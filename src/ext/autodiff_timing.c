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
#include <time.h>

extern TupleDesc autodiff_t_record_type(List *args)
{
    LambdaExpr *lambda = (LambdaExpr *)list_nth(args, 1);
    TupleDesc inDesc = (TupleDesc)list_nth(lambda->argtypes, 0);
    TupleDesc outDesc;
    outDesc = CreateTemplateTupleDesc(inDesc->natts * 2 + 1, false);

    for (int i = 0; i < inDesc->natts; i++)
    {
        TupleDescCopyEntry(outDesc, (AttrNumber)(i + 1), inDesc, (AttrNumber)(i + 1));
    }

    char attrNamesMapped[64];
    strcpy(attrNamesMapped, "Result");

    TupleDescInitEntry(outDesc, (AttrNumber)(inDesc->natts + 1), attrNamesMapped,
                       lambda->rettype, lambda->rettypmod, 0);

    for (int i = 0; i < inDesc->natts; i++)
    {
        char buffer[64];
        char *column_name = inDesc->attrs[i].attname.data;
        sprintf(buffer, "d_%s", column_name);

        TupleDescInitEntry(outDesc, (AttrNumber)(inDesc->natts + 2 + i), buffer,
                           lambda->rettype, lambda->rettypmod, 0);
    }

    return outDesc;
}

PG_MODULE_MAGIC;
PG_FUNCTION_INFO_V1_RECTYPE(autodiff_t_l2, autodiff_t_record_type);
PG_FUNCTION_INFO_V1_RECTYPE(autodiff_t_l3, autodiff_t_record_type);
PG_FUNCTION_INFO_V1_RECTYPE(autodiff_t_l4, autodiff_t_record_type);

Datum autodiff_t_l2_internal(PG_FUNCTION_ARGS, clock_t diff)
{
    MemoryContext oldcontext;
    FuncCallContext *funcctx;
    MemoryContext per_query_ctx;
    TupleDesc outDesc = NULL;
    Datum *replVal;
    Datum *oldVal;
    bool *replIsNull;
    bool *oldIsNull;

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
    TupleTableSlot *slot = MakeTupleTableSlot(NULL);
    Tuplestorestate *tsOut = tuplestore_begin_heap(true, false, work_mem);

    HeapTuple tuple;

    outDesc = CreateTupleDescCopy(inDesc);
    oldIsNull = (bool *)palloc(inDesc->natts * sizeof(bool));
    oldVal = (Datum *)palloc(outDesc->natts * sizeof(Datum));

    outDesc = CreateTemplateTupleDesc(inDesc->natts * 2 + 1, false);

    { // register each tuple-descriptor field with the corresponding field name
        for (int i = 0; i < inDesc->natts; i++)
        {
            TupleDescCopyEntry(outDesc, (AttrNumber)(i + 1), inDesc, (AttrNumber)(i + 1));
        }

        TupleDescInitEntry(outDesc, (AttrNumber)(inDesc->natts + 1), "Result",
                           lambda->rettype, lambda->rettypmod, 0);

        for (int i = 0; i < inDesc->natts; i++)
        {
            char buffer[64];
            char *column_name = inDesc->attrs[i].attname.data;
            sprintf(buffer, "d_%s", column_name);

            TupleDescInitEntry(outDesc, (AttrNumber)(inDesc->natts + 2 + i), buffer,
                               lambda->rettype, lambda->rettypmod, 0);
        }
    }

    replIsNull = (bool *)palloc((inDesc->natts * 2 + 1) * sizeof(bool));
    replVal = (Datum *)palloc((inDesc->natts * 2 + 1) * sizeof(Datum));

    Datum derivatives[inDesc->natts];
    int counter = 0;
    float8 time_compilation = 0.0, time_execution = 0.0;
    clock_t execution_tally = 0;

    for (slot = ExecProcNode(planState); !TupIsNull(slot); slot = ExecProcNode(planState))
    {
        bool isnull;
        Datum *val_ptr = oldVal;
        bool *null_ptr = oldIsNull;

        HeapTuple t;
        HeapTupleHeader hdr;

        if (slot->tts_mintuple)
        {
            hdr = slot->tts_tuple->t_data;
            heap_deform_tuple(slot->tts_tuple, inDesc, val_ptr, null_ptr);
        }
        else
        {
            slot_getallattrs(slot);
            t = heap_form_tuple(inDesc, slot->tts_values, slot->tts_isnull);
            val_ptr = slot->tts_values;
            null_ptr = slot->tts_isnull;
            hdr = t->t_data;
        }

        /* reset derivatives to avoid undefined behaviour */
        for (int i = 0; i < inDesc->natts; i++)
        {
            if (castNode(ExprState, lambda->exprstate)->lambdaContainsMatrix)
            {
                derivatives[i] = createScalar(0.0);
            }
            else
            {
                derivatives[i] = Float8GetDatum(0.0);
            }
        }

        clock_t begin = clock();

        PG_LAMBDA_SETARG(lambda, 0, HeapTupleHeaderGetDatum(hdr));
        Datum result = PG_LAMBDA_DERIVE(lambda, &isnull, derivatives);

        clock_t end = clock();

        clock_t diff_internal = end - begin;

        if (counter++ == 0) {
            time_compilation = ((float8)(diff_internal + diff)) / CLOCKS_PER_SEC;
            printf("the time taken for compilation was: %lf\n", time_compilation);
        }

        for (int i = 0; i < inDesc->natts; i++)
        {
            replVal[i] = val_ptr[i];
            replIsNull[i] = null_ptr[i];
        }
        replVal[inDesc->natts] = result;
        replIsNull[inDesc->natts] = false;

        for (int i = 0; i < inDesc->natts; i++)
        {
            replVal[inDesc->natts + 1 + i] = derivatives[i];
            replIsNull[inDesc->natts + 1 + i] = false;
        }

        tuple = heap_form_tuple(outDesc, replVal, replIsNull);
        tuplestore_puttuple(tsOut, tuple);
        //counter++;
    }

    // time_execution = ((float8)execution_tally) / CLOCKS_PER_SEC;
    // printf("all ticks combined for eval were: %ld\n", execution_tally);
    // //time_execution /= (float8) counter - 1;
    // printf("the time taken for execution was: %lf\n", time_execution);
    // printf("the number of runs: %d\n", counter);

    rsinfo->returnMode = SFRM_Materialize;
    rsinfo->setResult = tsOut;
    rsinfo->setDesc = outDesc;

    MemoryContextSwitchTo(oldcontext);

    return (Datum)0;
}

Datum autodiff_t_l3_internal(PG_FUNCTION_ARGS, Datum (*derivefunc)(Datum **arg, Datum *derivatives))
{
    MemoryContext oldcontext;
    FuncCallContext *funcctx;
    MemoryContext per_query_ctx;
    TupleDesc outDesc = NULL;
    Datum *replVal;
    Datum *oldVal;
    bool *replIsNull;
    bool *oldIsNull;

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
    TupleTableSlot *slot = MakeTupleTableSlot(NULL);
    Tuplestorestate *tsOut = tuplestore_begin_heap(true, false, work_mem);

    HeapTuple tuple;

    outDesc = CreateTupleDescCopy(inDesc);
    oldIsNull = (bool *)palloc(inDesc->natts * sizeof(bool));
    oldVal = (Datum *)palloc(outDesc->natts * sizeof(Datum));

    outDesc = CreateTemplateTupleDesc(inDesc->natts * 2 + 1, false);

    {
        for (int i = 0; i < inDesc->natts; i++)
        {
            TupleDescCopyEntry(outDesc, (AttrNumber)(i + 1), inDesc, (AttrNumber)(i + 1));
        }

        char attrNamesMapped[64];
        strcpy(attrNamesMapped, "Result");

        TupleDescInitEntry(outDesc, (AttrNumber)(inDesc->natts + 1), attrNamesMapped,
                           lambda->rettype, lambda->rettypmod, 0);

        for (int i = 0; i < inDesc->natts; i++)
        {
            char buffer[64];
            char *column_name = inDesc->attrs[i].attname.data;
            sprintf(buffer, "d_%s", column_name);

            TupleDescInitEntry(outDesc, (AttrNumber)(inDesc->natts + 2 + i), buffer,
                               lambda->rettype, lambda->rettypmod, 0);
        }
    }

    replIsNull = (bool *)palloc((inDesc->natts * 2 + 1) * sizeof(bool));
    replVal = (Datum *)palloc((inDesc->natts * 2 + 1) * sizeof(Datum));

    Datum derivatives[inDesc->natts];

    for (slot = ExecProcNode(planState); !TupIsNull(slot); slot = ExecProcNode(planState))
    {
        bool isnull;
        Datum *val_ptr = oldVal;
        bool *null_ptr = oldIsNull;

        HeapTuple t;
        HeapTupleHeader hdr;

        if (slot->tts_mintuple)
        {
            hdr = slot->tts_tuple->t_data;
            heap_deform_tuple(slot->tts_tuple, inDesc, val_ptr, null_ptr);
        }
        else
        {
            slot_getallattrs(slot);
            t = heap_form_tuple(inDesc, slot->tts_values, slot->tts_isnull);
            val_ptr = slot->tts_values;
            null_ptr = slot->tts_isnull;
            hdr = t->t_data;
        }
        /* reset derivatives to avoid undefined behaviour */
        for (int i = 0; i < inDesc->natts; i++)
        {
            derivatives[i] = Float8GetDatum(0.0);
        }

        Datum result = derivefunc(&val_ptr, derivatives);

        for (int i = 0; i < inDesc->natts; i++)
        {
            replVal[i] = val_ptr[i];
            replIsNull[i] = null_ptr[i];
        }

        replVal[inDesc->natts] = result;
        replIsNull[inDesc->natts] = false;

        for (int i = 0; i < inDesc->natts; i++)
        {
            replVal[inDesc->natts + 1 + i] = derivatives[i];
            replIsNull[inDesc->natts + 1 + i] = false;
        }

        tuple = heap_form_tuple(outDesc, replVal, replIsNull);

        tuplestore_puttuple(tsOut, tuple);
    }

    rsinfo->returnMode = SFRM_Materialize;
    rsinfo->setResult = tsOut;
    rsinfo->setDesc = outDesc;

    MemoryContextSwitchTo(oldcontext);

    return (Datum)0;
}

Datum autodiff_t_l4_internal(PG_FUNCTION_ARGS)
{
    MemoryContext oldcontext;
    FuncCallContext *funcctx;
    MemoryContext per_query_ctx;
    TupleDesc outDesc = NULL;
    Datum *replVal;
    Datum *oldVal;
    bool *replIsNull;
    bool *oldIsNull;

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
    TupleTableSlot *slot = MakeTupleTableSlot(NULL);
    Tuplestorestate *tsOut = tuplestore_begin_heap(true, false, work_mem);

    HeapTuple tuple;

    outDesc = CreateTupleDescCopy(inDesc);
    oldIsNull = (bool *)palloc(inDesc->natts * sizeof(bool));
    oldVal = (Datum *)palloc(outDesc->natts * sizeof(Datum));

    outDesc = CreateTemplateTupleDesc(inDesc->natts * 2 + 1, false);

    {
        for (int i = 0; i < inDesc->natts; i++)
        {
            TupleDescCopyEntry(outDesc, (AttrNumber)(i + 1), inDesc, (AttrNumber)(i + 1));
        }

        char attrNamesMapped[64];
        strcpy(attrNamesMapped, "Result");

        TupleDescInitEntry(outDesc, (AttrNumber)(inDesc->natts + 1), attrNamesMapped,
                           lambda->rettype, lambda->rettypmod, 0);

        for (int i = 0; i < inDesc->natts; i++)
        {
            char buffer[64];
            char *column_name = inDesc->attrs[i].attname.data;
            sprintf(buffer, "d_%s", column_name);

            TupleDescInitEntry(outDesc, (AttrNumber)(inDesc->natts + 2 + i), buffer,
                               lambda->rettype, lambda->rettypmod, 0);
        }
    }

    replIsNull = (bool *)palloc((inDesc->natts * 2 + 1) * sizeof(bool));
    replVal = (Datum *)palloc((inDesc->natts * 2 + 1) * sizeof(Datum));

    Datum derivatives[inDesc->natts];

    for (slot = ExecProcNode(planState); !TupIsNull(slot); slot = ExecProcNode(planState))
    {
        bool isnull;
        Datum *val_ptr = oldVal;
        bool *null_ptr = oldIsNull;

        HeapTuple t;
        HeapTupleHeader hdr;

        if (slot->tts_mintuple)
        {
            hdr = slot->tts_tuple->t_data;
            heap_deform_tuple(slot->tts_tuple, inDesc, val_ptr, null_ptr);
        }
        else
        {
            slot_getallattrs(slot);
            t = heap_form_tuple(inDesc, slot->tts_values, slot->tts_isnull);
            val_ptr = slot->tts_values;
            null_ptr = slot->tts_isnull;
            hdr = t->t_data;
        }
        /* reset derivatives to avoid undefined behaviour */
        for (int i = 0; i < inDesc->natts; i++)
        {
            derivatives[i] = Float8GetDatum(0.0);
        }

        Datum result = PG_SIMPLE_LAMBDA_INJECT_DERIV(&val_ptr, derivatives, 0);

        for (int i = 0; i < inDesc->natts; i++)
        {
            replVal[i] = val_ptr[i];
            replIsNull[i] = null_ptr[i];
        }

        replVal[inDesc->natts] = result;
        replIsNull[inDesc->natts] = false;

        for (int i = 0; i < inDesc->natts; i++)
        {
            replVal[inDesc->natts + 1 + i] = derivatives[i];
            replIsNull[inDesc->natts + 1 + i] = false;
        }

        tuple = heap_form_tuple(outDesc, replVal, replIsNull);

        tuplestore_puttuple(tsOut, tuple);
    }

    rsinfo->returnMode = SFRM_Materialize;
    rsinfo->setResult = tsOut;
    rsinfo->setDesc = outDesc;

    MemoryContextSwitchTo(oldcontext);

    return (Datum)0;
}

Datum autodiff_t_l2(PG_FUNCTION_ARGS)
{
    // ReturnSetInfo *rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;
    // LambdaExpr *lambda = PG_GETARG_LAMBDA(1);

    // clock_t t = clock();

    // llvm_enter_tmp_context(rsinfo->econtext->ecxt_estate);
    // ExecInitLambdaExpr((Node *)lambda, false, true);
    // llvm_leave_tmp_context(rsinfo->econtext->ecxt_estate);

    // t = clock() - t;
    // double time_taken_compile = ((double)t) / CLOCKS_PER_SEC;
    // printf("Time taken for compilation of L1|L2: %f s", time_taken_compile);
    // t = clock();

    // Datum result = autodiff_t_l2_internal(fcinfo);
    // t = clock() - t;
    // double time_taken_execute = ((double)t) / CLOCKS_PER_SEC;
    // printf("Time taken for execution of L1|L2: %f s", time_taken_execute);
    // return result;
    ReturnSetInfo *rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;
    LambdaExpr *lambda = PG_GETARG_LAMBDA(1);

    clock_t t = clock();

    llvm_enter_tmp_context(rsinfo->econtext->ecxt_estate);
    ExecInitLambdaExpr((Node *)lambda, false, true);
    llvm_leave_tmp_context(rsinfo->econtext->ecxt_estate);

    clock_t diff = clock() - t;
    Datum result = autodiff_t_l2_internal(fcinfo, diff);
    clock_t complete = clock() - t;
    double time_taken_execute = ((double)complete) / CLOCKS_PER_SEC;
    printf("Time taken for execution of L2: %lf s", time_taken_execute);
    return result;
}

Datum autodiff_t_l3(PG_FUNCTION_ARGS)
{
    ReturnSetInfo *rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;
    LambdaExpr *lambda = PG_GETARG_LAMBDA(1);

    clock_t t = clock();

    llvm_enter_tmp_context(rsinfo->econtext->ecxt_estate);
    LLVMJitContext *jitContext = (LLVMJitContext *)(rsinfo->econtext->ecxt_estate->es_jit);

    ExecInitLambdaExpr((Node *)lambda, true, true);
    Datum (*compiled_func)(Datum **, Datum *);
    compiled_func = llvm_prepare_simple_expression_derivation(castNode(ExprState, lambda->exprstate));

    llvm_leave_tmp_context(rsinfo->econtext->ecxt_estate);

    t = clock() - t;
    double time_taken_compile = ((double)t) / CLOCKS_PER_SEC;
    printf("Time taken for compilation of L3: %f s", time_taken_compile);
    t = clock();

    Datum result = autodiff_t_l3_internal(fcinfo, compiled_func);
    t = clock() - t;
    double time_taken_execute = ((double)t) / CLOCKS_PER_SEC;
    printf("Time taken for execution of L3: %f s", time_taken_execute);
    return result;
}

Datum autodiff_t_l4(PG_FUNCTION_ARGS)
{
    ReturnSetInfo *rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;
    LambdaExpr *lambda = PG_GETARG_LAMBDA(1);
    LLVMJitContext *jitContext;

    clock_t t = clock();

    llvm_enter_tmp_context(rsinfo->econtext->ecxt_estate);
    ExecInitLambdaExpr((Node *)lambda, true, true);
    jitContext = (LLVMJitContext *)(rsinfo->econtext->ecxt_estate->es_jit);

    Datum (*compiled_func)(FunctionCallInfo);
    compiled_func = llvm_prepare_lambda_tablefunc(jitContext, "ext/autodiff_timing.bc", "autodiff_t_l4_internal", 1);
    llvm_leave_tmp_context(rsinfo->econtext->ecxt_estate);
    
    t = clock() - t;
    double time_taken_compile = ((double)t) / CLOCKS_PER_SEC;
    printf("Time taken for compilation of L4: %f s", time_taken_compile);
    t = clock();

    Datum result = compiled_func(fcinfo);
    t = clock() - t;
    double time_taken_execute = ((double)t) / CLOCKS_PER_SEC;
    printf("Time taken for execution of L4: %f s", time_taken_execute);
    return result;
}
