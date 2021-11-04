#include "postgres.h"
#include "funcapi.h"
#include "jit/llvmjit.h"
#include "executor/spi.h"
#include "executor/execExpr.h"
#include "stdint.h"
#include "utils/builtins.h"
#include "parser/parser.h"
#include "nodes/nodes.h"
#include "catalog/pg_type.h"
#include "access/htup_details.h"
#include "nodes/print.h"
#include "portability/instr_time.h"
#include "utils/lsyscache.h"
#include "utils/hsearch.h"
#include "common/config_info.h"
#include <math.h>
#include <pthread.h>
#include "miscadmin.h"

extern TupleDesc gradient_descent_record_type(List *args)
{
    LambdaExpr *lambda = (LambdaExpr *)list_nth(args, 1);
    int num_atts = DatumGetInt32(((Const *)list_nth(args, 3))->constvalue);

    TupleDesc outDesc = CreateTemplateTupleDesc(num_atts + 1, false);

    for (int i = 0; i < num_atts; i++)
    {
        char buffer[12];
        sprintf(buffer, "a%d", i);         
        TupleDescInitEntry(outDesc, (AttrNumber)(i + 1), buffer,
                           lambda->rettype, lambda->rettypmod, 0);
    }

    char attrNamesMapped[2];
    strcpy(attrNamesMapped, "b");

    TupleDescInitEntry(outDesc, (AttrNumber)(num_atts + 1), attrNamesMapped,
                       lambda->rettype, lambda->rettypmod, 0);

    return outDesc;
}

PG_MODULE_MAGIC;
PG_FUNCTION_INFO_V1_RECTYPE(gradient_descent_l1_2, gradient_descent_record_type);
PG_FUNCTION_INFO_V1_RECTYPE(gradient_descent_l3, gradient_descent_record_type);
PG_FUNCTION_INFO_V1_RECTYPE(gradient_descent_l4, gradient_descent_record_type);

Datum gradient_descent_internal_l1_2(PG_FUNCTION_ARGS)
{
    MemoryContext oldcontext;
    FuncCallContext *funcctx;
    MemoryContext per_query_ctx;
    TupleDesc outDesc = NULL;
    Tuplestorestate *ttsIn;
    Datum *replVal;
    Datum *oldVal;
    bool *replIsNull;
    bool *oldIsNull;
    int batch_size = PG_GETARG_INT32(4); // amount of tuples to be loaded during grad_desc
    int num_atts = PG_GETARG_INT32(3);   // number of independent variables per run(DOES NOT INCLUDE b)
    int iterations = PG_GETARG_INT32(2); // Number of iterations for grad_desc algorithm

    ReturnSetInfo *rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;
    LLVMJitContext *jitContext = (LLVMJitContext *)(rsinfo->econtext->ecxt_estate->es_jit);

    if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("set-valued function called in context that cannot accept a set")));
    if (!(rsinfo->allowedModes & SFRM_Materialize))
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("materialize mode required, but it is not "
                        "allowed in this context")));

    LambdaExpr *lambda = PG_GETARG_LAMBDA(1);
    TupleDesc inDesc = (TupleDesc)list_nth(lambda->argtypes, 0);

    per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
    oldcontext = MemoryContextSwitchTo(per_query_ctx);

    PlanState *planState = (PlanState *)PG_GETARG_POINTER(0);
    ttsIn = ((TypedTuplestore *)PG_GETARG_POINTER(0))->tuplestorestate;
    TupleTableSlot *slot = MakeTupleTableSlot(NULL);
    Tuplestorestate *tsOut = tuplestore_begin_heap(true, false, work_mem);

    HeapTuple tuple;

    oldIsNull = (bool *)palloc(inDesc->natts * sizeof(bool));
    oldVal = (Datum *)palloc(inDesc->natts * sizeof(Datum));

    {
        outDesc = CreateTemplateTupleDesc(num_atts + 1, false);

        for (int i = 0; i < num_atts; i++)
        {
            char buffer[15];
            sprintf(buffer, "a%d", i);

            TupleDescInitEntry(outDesc, (AttrNumber)(i + 1), buffer,
                               lambda->rettype, lambda->rettypmod, 0);
        }

        char attrNamesMapped[2];
        strcpy(attrNamesMapped, "b");

        TupleDescInitEntry(outDesc, (AttrNumber)(num_atts + 1), attrNamesMapped,
                           lambda->rettype, lambda->rettypmod, 0);
    }

    replIsNull = (bool *)palloc((num_atts + 1) * sizeof(bool));
    replVal = (Datum *)palloc((num_atts + 1) * sizeof(Datum));

    Datum derivatives[inDesc->natts];                                              /* Returned derivatives from autodiff */
    Datum coefficients_per_iteration[num_atts + 1];                                /* coefficients of lambda_expr */
    float8 *derivatives_tally = (float8 *)palloc((num_atts + 1) * sizeof(float8)); // all derivatives for a single iterations tallied up

    for (int i = 0; i < num_atts + 1; i++)
    {
        coefficients_per_iteration[i] = Float8GetDatum(1.0);
    }

    for (int i = 0; i < iterations; i++)
    {
        while (tuplestore_gettupleslot(ttsIn, true, false, slot))
        {
            HeapTupleHeader hdr;
            hdr = slot->tts_tuple->t_data;
            bool isnull;
            heap_deform_tuple(slot->tts_tuple, inDesc, oldVal, oldIsNull);

            for (int it = 0; it < num_atts + 1; it++)
            {
                /* Set elements of tuple in slot to the correct cooefficients*/
                oldVal[it] = coefficients_per_iteration[it];
            }

            for (int it = 0; it < inDesc->natts; it++)
            {
                /* Reset derivatives to avoid undefined behaviour */
                derivatives[it] = Float8GetDatum(0.0);
            }

            PG_LAMBDA_SETARG(lambda, 0, HeapTupleHeaderGetDatum(hdr));
            Datum result = PG_LAMBDA_DERIVE(lambda, &isnull, derivatives);

            for (int it = 0; it < num_atts + 1; it++)
            {
                // tally up all elements per column
                derivatives_tally[it] += DatumGetFloat8(derivatives[it]);
            }
        }
        tuplestore_rescan(ttsIn);

        /* Calculate avg of tally and adjust coefficients */
        for (int it = 0; it < num_atts + 1; it++)
        {
            // 0.001 is the learning rate, can be adjusted or even a parameter to be given from the outside
            float8 tally = (0.001 * derivatives_tally[it]) / batch_size;
            coefficients_per_iteration[it] = Float8GetDatum(DatumGetFloat8(coefficients_per_iteration[it]) - tally);
        }
    }

    for (int i = 0; i < num_atts + 1; i++)
    {
        replVal[i] = coefficients_per_iteration[i];
        replIsNull[i] = false;
    }

    tuple = heap_form_tuple(outDesc, replVal, replIsNull);

    tuplestore_puttuple(tsOut, tuple);

    rsinfo->returnMode = SFRM_Materialize;
    rsinfo->setResult = tsOut;
    rsinfo->setDesc = outDesc;

    MemoryContextSwitchTo(oldcontext);
    return (Datum)0;
}

Datum gradient_descent_internal_l3(PG_FUNCTION_ARGS, Datum (*derivefunc)(Datum **arg, Datum *derivatives))
{
    MemoryContext oldcontext;
    FuncCallContext *funcctx;
    MemoryContext per_query_ctx;
    TupleDesc outDesc = NULL;
    Tuplestorestate *ttsIn;
    Datum *replVal;
    Datum *oldVal;
    bool *replIsNull;
    bool *oldIsNull;
    int batch_size = PG_GETARG_INT32(4); // amount of tuples to be loaded during grad_desc
    int num_atts = PG_GETARG_INT32(3);   // number of independent variables per run(DOES NOT INCLUDE b)
    int iterations = PG_GETARG_INT32(2); // Number of iterations for grad_desc algorithm

    ReturnSetInfo *rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;
    LLVMJitContext *jitContext = (LLVMJitContext *)(rsinfo->econtext->ecxt_estate->es_jit);

    if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("set-valued function called in context that cannot accept a set")));
    if (!(rsinfo->allowedModes & SFRM_Materialize))
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("materialize mode required, but it is not "
                        "allowed in this context")));

    LambdaExpr *lambda = PG_GETARG_LAMBDA(1);
    TupleDesc inDesc = (TupleDesc)list_nth(lambda->argtypes, 0);

    per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
    oldcontext = MemoryContextSwitchTo(per_query_ctx);

    PlanState *planState = (PlanState *)PG_GETARG_POINTER(0);
    ttsIn = ((TypedTuplestore *)PG_GETARG_POINTER(0))->tuplestorestate;
    TupleTableSlot *slot = MakeTupleTableSlot(NULL);
    Tuplestorestate *tsOut = tuplestore_begin_heap(true, false, work_mem);

    HeapTuple tuple;

    oldIsNull = (bool *)palloc(inDesc->natts * sizeof(bool));
    oldVal = (Datum *)palloc(inDesc->natts * sizeof(Datum));

    {
        outDesc = CreateTemplateTupleDesc(num_atts + 1, false);

        for (int i = 0; i < num_atts; i++)
        {
            char buffer[15];
            sprintf(buffer, "a%d", i);

            TupleDescInitEntry(outDesc, (AttrNumber)(i + 1), buffer,
                               lambda->rettype, lambda->rettypmod, 0);
        }

        char attrNamesMapped[2];
        strcpy(attrNamesMapped, "b");

        TupleDescInitEntry(outDesc, (AttrNumber)(num_atts + 1), attrNamesMapped,
                           lambda->rettype, lambda->rettypmod, 0);
    }

    replIsNull = (bool *)palloc((num_atts + 1) * sizeof(bool));
    replVal = (Datum *)palloc((num_atts + 1) * sizeof(Datum));

    Datum derivatives[inDesc->natts];                                                 /* Returned derivatives from autodiff */
    Datum coefficients_per_iteration[num_atts + 1];                                   /* coefficients of lambda_expr */
    float8 *derivatives_tally = (float8 *) palloc((num_atts + 1) * sizeof(float8));   // all derivatives for a single iterations tallied up

    for (int i = 0; i < num_atts + 1; i++)
    {
        coefficients_per_iteration[i] = Float8GetDatum(1.0);
    }

    for (int i = 0; i < iterations; i++)
    {
        while (tuplestore_gettupleslot(ttsIn, true, false, slot))
        {
            heap_deform_tuple(slot->tts_tuple, inDesc, oldVal, oldIsNull);

            for(int it = 0; it < num_atts + 1; it++) {
                /* Set elements of tuple in slot to the correct cooefficients*/
                oldVal[it] = coefficients_per_iteration[it];
            }

            for(int it = 0; it < inDesc->natts; it++) {
                /* Reset derivatives to avoid undefined behaviour */
                derivatives[it] = Float8GetDatum(0.0);
            }

            Datum result = derivefunc(&oldVal, derivatives);

            for (int it = 0; it < num_atts + 1; it++)
            {
                // tally up all elements per column
                derivatives_tally[it] += DatumGetFloat8(derivatives[it]);
            }
        }
        tuplestore_rescan(ttsIn);

        /* Calculate avg of tally and adjust coefficients */
        for (int it = 0; it < num_atts + 1; it++)
        {
            // 0.001 is the learning rate, can be adjusted or even a parameter to be given from the outside
            float8 tally = (0.001 * derivatives_tally[it]) / batch_size;
            coefficients_per_iteration[it] = Float8GetDatum(DatumGetFloat8(coefficients_per_iteration[it]) - tally);
        }
    }

    for (int i = 0; i < num_atts + 1; i++)
    {
        replVal[i] = coefficients_per_iteration[i];
        replIsNull[i] = false;
    }

    tuple = heap_form_tuple(outDesc, replVal, replIsNull);

    tuplestore_puttuple(tsOut, tuple);

    rsinfo->returnMode = SFRM_Materialize;
    rsinfo->setResult = tsOut;
    rsinfo->setDesc = outDesc;

    MemoryContextSwitchTo(oldcontext);
    return (Datum)0;
}

Datum gradient_descent_internal_l4(PG_FUNCTION_ARGS)
{
    MemoryContext oldcontext;
    FuncCallContext *funcctx;
    MemoryContext per_query_ctx;
    TupleDesc outDesc = NULL;
    Tuplestorestate *ttsIn;
    Datum *replVal;
    Datum *oldVal;
    bool *replIsNull;
    bool *oldIsNull;
    int batch_size = PG_GETARG_INT32(4); // amount of tuples to be loaded during grad_desc
    int num_atts = PG_GETARG_INT32(3);   // number of independent variables per run(DOES NOT INCLUDE b)
    int iterations = PG_GETARG_INT32(2); // Number of iterations for grad_desc algorithm

    ReturnSetInfo *rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;
    LLVMJitContext *jitContext = (LLVMJitContext *)(rsinfo->econtext->ecxt_estate->es_jit);

    if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("set-valued function called in context that cannot accept a set")));
    if (!(rsinfo->allowedModes & SFRM_Materialize))
        ereport(ERROR,
                (errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
                 errmsg("materialize mode required, but it is not "
                        "allowed in this context")));

    LambdaExpr *lambda = PG_GETARG_LAMBDA(1);
    TupleDesc inDesc = (TupleDesc)list_nth(lambda->argtypes, 0);

    per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
    oldcontext = MemoryContextSwitchTo(per_query_ctx);

    PlanState *planState = (PlanState *)PG_GETARG_POINTER(0);
    ttsIn = ((TypedTuplestore *)PG_GETARG_POINTER(0))->tuplestorestate;
    TupleTableSlot *slot = MakeTupleTableSlot(NULL);
    Tuplestorestate *tsOut = tuplestore_begin_heap(true, false, work_mem);

    HeapTuple tuple;

    oldIsNull = (bool *)palloc(inDesc->natts * sizeof(bool));
    oldVal = (Datum *)palloc(inDesc->natts * sizeof(Datum));

    {
        outDesc = CreateTemplateTupleDesc(num_atts + 1, false);

        for (int i = 0; i < num_atts; i++)
        {
            char buffer[15];
            sprintf(buffer, "a%d", i);

            TupleDescInitEntry(outDesc, (AttrNumber)(i + 1), buffer,
                               lambda->rettype, lambda->rettypmod, 0);
        }

        char attrNamesMapped[2];
        strcpy(attrNamesMapped, "b");

        TupleDescInitEntry(outDesc, (AttrNumber)(num_atts + 1), attrNamesMapped,
                           lambda->rettype, lambda->rettypmod, 0);
    }

    replIsNull = (bool *)palloc((num_atts + 1) * sizeof(bool));
    replVal = (Datum *)palloc((num_atts + 1) * sizeof(Datum));

    Datum derivatives[inDesc->natts];                                              /* Returned derivatives from autodiff */
    Datum coefficients_per_iteration[num_atts + 1];                                /* coefficients of lambda_expr */
    float8 *derivatives_tally = (float8 *)palloc((num_atts + 1) * sizeof(float8)); // all derivatives for a single iterations tallied up

    for (int i = 0; i < num_atts + 1; i++)
    {
        coefficients_per_iteration[i] = Float8GetDatum(1.0);
    }

    for (int i = 0; i < iterations; i++)
    {
        while (tuplestore_gettupleslot(ttsIn, true, false, slot))
        {
            heap_deform_tuple(slot->tts_tuple, inDesc, oldVal, oldIsNull);

            for (int it = 0; it < num_atts + 1; it++)
            {
                /* Set elements of tuple in slot to the correct cooefficients*/
                oldVal[it] = coefficients_per_iteration[it];
            }

            for (int it = 0; it < inDesc->natts; it++)
            {
                /* Reset derivatives to avoid undefined behaviour */
                derivatives[it] = Float8GetDatum(0.0);
            }

            Datum result = PG_SIMPLE_LAMBDA_INJECT_DERIV(&oldVal, derivatives, 0);

            for (int it = 0; it < num_atts + 1; it++)
            {
                // tally up all elements per column
                derivatives_tally[it] += DatumGetFloat8(derivatives[it]);
            }
        }
        tuplestore_rescan(ttsIn);

        /* Calculate avg of tally and adjust coefficients */
        for (int it = 0; it < num_atts + 1; it++)
        {
            // 0.001 is the learning rate, can be adjusted or even a parameter to be given from the outside
            float8 tally = (0.001 * derivatives_tally[it]) / batch_size;
            coefficients_per_iteration[it] = Float8GetDatum(DatumGetFloat8(coefficients_per_iteration[it]) - tally);
        }
    }

    for (int i = 0; i < num_atts + 1; i++)
    {
        replVal[i] = coefficients_per_iteration[i];
        replIsNull[i] = false;
    }

    tuple = heap_form_tuple(outDesc, replVal, replIsNull);

    tuplestore_puttuple(tsOut, tuple);

    rsinfo->returnMode = SFRM_Materialize;
    rsinfo->setResult = tsOut;
    rsinfo->setDesc = outDesc;

    MemoryContextSwitchTo(oldcontext);
    return (Datum)0;
}

Datum gradient_descent_l1_2(PG_FUNCTION_ARGS)
{
    ReturnSetInfo *rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;
    LambdaExpr *lambda = PG_GETARG_LAMBDA(1);

    llvm_enter_tmp_context(rsinfo->econtext->ecxt_estate);
    ExecInitLambdaExpr((Node *)lambda, false, true);
    llvm_leave_tmp_context(rsinfo->econtext->ecxt_estate);

    return gradient_descent_internal_l1_2(fcinfo);
}

Datum gradient_descent_l3(PG_FUNCTION_ARGS)
{
    ReturnSetInfo *rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;
    LambdaExpr *lambda = PG_GETARG_LAMBDA(1);

    llvm_enter_tmp_context(rsinfo->econtext->ecxt_estate);
    LLVMJitContext *jitContext = (LLVMJitContext *)(rsinfo->econtext->ecxt_estate->es_jit);

    ExecInitLambdaExpr((Node *)lambda, true, true);
    Datum (*compiled_func)(Datum **, Datum *);
    compiled_func = llvm_prepare_simple_expression_derivation(castNode(ExprState, lambda->exprstate));

    llvm_leave_tmp_context(rsinfo->econtext->ecxt_estate);

    return gradient_descent_internal_l3(fcinfo, compiled_func);
}

Datum gradient_descent_l4(PG_FUNCTION_ARGS)
{
    ReturnSetInfo *rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;
    LambdaExpr *lambda = PG_GETARG_LAMBDA(1);

    llvm_enter_tmp_context(rsinfo->econtext->ecxt_estate);
    LLVMJitContext *jitContext = (LLVMJitContext *)(rsinfo->econtext->ecxt_estate->es_jit);

    ExecInitLambdaExpr((Node *)lambda, true, true);
    Datum (*compiled_func)(FunctionCallInfo);
    compiled_func = llvm_prepare_lambda_tablefunc(jitContext, "ext/gradient_desc_ext.bc", "gradient_descent_internal_l4", 1);

    llvm_leave_tmp_context(rsinfo->econtext->ecxt_estate);

    return compiled_func(fcinfo);
}