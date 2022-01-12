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

extern TupleDesc gradient_descent_m_record_type(List *args)
{
    LambdaExpr *lambda = (LambdaExpr *)list_nth(args, 1);
    TupleDesc inDesc = (TupleDesc)list_nth(lambda->argtypes, 0);
    int num_atts = DatumGetInt32(((Const *)list_nth(args, 3))->constvalue);

    TupleDesc outDesc = CreateTemplateTupleDesc(num_atts, false);

    for (int i = 0; i < num_atts; i++)
    {
        TupleDescCopyEntry(outDesc, (AttrNumber)(i + 1), inDesc, (AttrNumber)(i + 1));
    }

    return outDesc;
}

PG_MODULE_MAGIC;
PG_FUNCTION_INFO_V1_RECTYPE(gradient_descent_m_l1_2, gradient_descent_m_record_type);
PG_FUNCTION_INFO_V1_RECTYPE(gradient_descent_m_l3, gradient_descent_m_record_type);
PG_FUNCTION_INFO_V1_RECTYPE(gradient_descent_m_l4, gradient_descent_m_record_type);

Datum gradient_descent_m_internal_l1_2(PG_FUNCTION_ARGS)
{
    float8 learning_rate = PG_GETARG_FLOAT8(5); // learning rate for gradient desc
    int batch_size = PG_GETARG_INT32(4);        // amount of tuples to be loaded during grad_desc
    int num_atts = PG_GETARG_INT32(3);          // number of independent variables per run
    int iterations = PG_GETARG_INT32(2);        // Number of iterations for grad_desc algorithm

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

    MemoryContext per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
    MemoryContext per_expr_eval = rsinfo->econtext->ecxt_per_tuple_memory;
    MemoryContext oldcontext = MemoryContextSwitchTo(per_query_ctx);

    Tuplestorestate *ttsIn = ((TypedTuplestore *)PG_GETARG_POINTER(0))->tuplestorestate;
    TupleTableSlot *slot = MakeTupleTableSlot(NULL);
    Tuplestorestate *tsOut = tuplestore_begin_heap(true, false, work_mem);
    int tupleStoreCount = (int)tuplestore_tuple_count(ttsIn);
    tuplestore_rescan(ttsIn);

    if (batch_size < 1 || batch_size > tupleStoreCount)
    {
        batch_size = tupleStoreCount;
    }

    TupleDesc outDesc = CreateTemplateTupleDesc(num_atts, false);
    for (int i = 0; i < num_atts; i++)
    {
        TupleDescCopyEntry(outDesc, (AttrNumber)(i + 1), inDesc, (AttrNumber)(i + 1));
    }

    HeapTuple tuple;

    bool *oldIsNull = (bool *)palloc_extended(inDesc->natts * sizeof(bool), (MCXT_ALLOC_ZERO));
    Datum *oldVal = (Datum *)palloc_extended(inDesc->natts * sizeof(Datum), (MCXT_ALLOC_ZERO));

    bool *replIsNull = (bool *)palloc_extended((num_atts) * sizeof(bool), (MCXT_ALLOC_ZERO));
    Datum *replVal = (Datum *)palloc_extended((num_atts) * sizeof(Datum), (MCXT_ALLOC_ZERO));

    Datum *derivatives = (Datum *)palloc_extended(inDesc->natts * sizeof(Datum), (MCXT_ALLOC_ZERO));             // Returned derivatives from autodiff
    Datum *coefficients_per_iteration = (Datum *)palloc_extended((num_atts) * sizeof(Datum), (MCXT_ALLOC_ZERO)); // coefficients of lambda_expr
    Datum *derivatives_tally = (Datum *)palloc_extended((num_atts) * sizeof(Datum), (MCXT_ALLOC_ZERO));          // all derivatives for a single iterations tallied up(indricetion through ArrayType pointers)

    if (tuplestore_gettupleslot(ttsIn, true, false, slot))
    {
        heap_deform_tuple(slot->tts_tuple, inDesc, oldVal, oldIsNull);
        tuplestore_rescan(ttsIn);
    }
    else
    {
        ereport(ERROR,
                (errcode(ERRCODE_ASSERT_FAILURE),
                 errmsg("Gradient Descent cannot work on empty tables!")));
    }

    for (int i = 0; i < num_atts; i++)
    {
        int ndim = ARR_NDIM(DatumGetArrayTypeP(oldVal[i]));
        int *dims = ARR_DIMS(DatumGetArrayTypeP(oldVal[i]));
        int *lbs = ARR_LBOUND(DatumGetArrayTypeP(oldVal[i]));

        coefficients_per_iteration[i] = PointerGetDatum(copyArray(oldVal[i]));
        derivatives_tally[i] = PointerGetDatum(initResult(ndim, dims, lbs));
    }

    MemoryContextSwitchTo(per_expr_eval);

    // printf("debug info: iterations: %d, LR: %lf, batchsize: %d, size of tupleStore: %d, num_atts: %d\n", iterations, learning_rate, batch_size, tupleStoreCount, num_atts);
    for (int it = 0; it < iterations; it++)
    {
        int internal_counter = 0;
        while (tuplestore_gettupleslot(ttsIn, true, false, slot))
        {
            bool isnull;
            heap_deform_tuple(slot->tts_tuple, inDesc, oldVal, oldIsNull);

            for (int i = 0; i < num_atts; i++)
            {
                oldVal[i] = PointerGetDatum(copyArray(coefficients_per_iteration[i]));
            }

            for (int i = 0; i < inDesc->natts; i++)
            {
                derivatives[i] = createScalar(0.0);
            }

            HeapTuple newTup = heap_form_tuple(inDesc, oldVal, oldIsNull); // only way to feed the coefficients into the HeapTupleHeader struct-format
            HeapTupleHeader newHdr = newTup->t_data;
            heap_deform_tuple(newTup, inDesc, oldVal, oldIsNull);

            PG_LAMBDA_SETARG(lambda, 0, HeapTupleHeaderGetDatum(newHdr));
            Datum result = PG_LAMBDA_DERIVE(lambda, &isnull, derivatives);

            for (int i = 0; i < num_atts; i++)
            {
                derivatives_tally[i] = matrix_add_inplace(derivatives_tally[i], derivatives[i]);
                // matrixPrint(derivatives[i]);
            }
            MemoryContextResetAndDeleteChildren(per_expr_eval);
        }
        tuplestore_rescan(ttsIn);

        for (int i = 0; i < num_atts; i++)
        {
            coefficients_per_iteration[i] = mat_apply_gradient(coefficients_per_iteration[i], derivatives_tally[i], learning_rate, batch_size);
            matrixSetValue(derivatives_tally[i], (float8)0.0);
        }
    }
    MemoryContextSwitchTo(per_query_ctx);

    for (int i = 0; i < num_atts; i++)
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

Datum gradient_descent_m_internal_l3(PG_FUNCTION_ARGS, Datum (*derivefunc)(Datum **arg, Datum *derivatives))
{
    float8 learning_rate = PG_GETARG_FLOAT8(5); // learning rate for gradient desc
    int batch_size = PG_GETARG_INT32(4);        // amount of tuples to be loaded during grad_desc
    int num_atts = PG_GETARG_INT32(3);          // number of independent variables per run
    int iterations = PG_GETARG_INT32(2);        // Number of iterations for grad_desc algorithm

    // printf("Grad_desc_l3_internal begin\n");

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

    MemoryContext per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
    MemoryContext per_expr_eval = rsinfo->econtext->ecxt_per_tuple_memory;
    MemoryContext oldcontext = MemoryContextSwitchTo(per_query_ctx);

    // printf("Grad_desc_l3_internal switched MCTX\n");

    Tuplestorestate *ttsIn = ((TypedTuplestore *)PG_GETARG_POINTER(0))->tuplestorestate;
    TupleTableSlot *slot = MakeTupleTableSlot(NULL);
    Tuplestorestate *tsOut = tuplestore_begin_heap(true, false, work_mem);
    int tupleStoreCount = (int)tuplestore_tuple_count(ttsIn);
    tuplestore_rescan(ttsIn);

    if (batch_size < 1 || batch_size > tupleStoreCount)
    {
        batch_size = tupleStoreCount;
    }

    // printf("Grad_desc_l3_internal got batch size\n");

    TupleDesc outDesc = CreateTemplateTupleDesc(num_atts, false);
    for (int i = 0; i < num_atts; i++)
    {
        TupleDescCopyEntry(outDesc, (AttrNumber)(i + 1), inDesc, (AttrNumber)(i + 1));
    }

    HeapTuple tuple;

    bool *oldIsNull = (bool *)palloc_extended(inDesc->natts * sizeof(bool), (MCXT_ALLOC_ZERO));
    Datum *oldVal = (Datum *)palloc_extended(inDesc->natts * sizeof(Datum), (MCXT_ALLOC_ZERO));

    bool *replIsNull = (bool *)palloc_extended((num_atts) * sizeof(bool), (MCXT_ALLOC_ZERO));
    Datum *replVal = (Datum *)palloc_extended((num_atts) * sizeof(Datum), (MCXT_ALLOC_ZERO));

    Datum *derivatives = (Datum *)palloc_extended(inDesc->natts * sizeof(Datum), (MCXT_ALLOC_ZERO));             // Returned derivatives from autodiff
    Datum *coefficients_per_iteration = (Datum *)palloc_extended((num_atts) * sizeof(Datum), (MCXT_ALLOC_ZERO)); // coefficients of lambda_expr
    Datum *derivatives_tally = (Datum *)palloc_extended((num_atts) * sizeof(Datum), (MCXT_ALLOC_ZERO));          // all derivatives for a single iterations tallied up(indricetion through ArrayType pointers)

    // printf("Grad_desc_l3_internal alloc'ed mem\n");

    if (tuplestore_gettupleslot(ttsIn, true, false, slot))
    {
        heap_deform_tuple(slot->tts_tuple, inDesc, oldVal, oldIsNull);
        tuplestore_rescan(ttsIn);
    }
    else
    {
        ereport(ERROR,
                (errcode(ERRCODE_ASSERT_FAILURE),
                 errmsg("Gradient Descent cannot work on empty tables!")));
    }

    // printf("Grad_desc_l3_internal got first tuple\n");

    for (int i = 0; i < num_atts; i++)
    {
        int ndim = ARR_NDIM(DatumGetArrayTypeP(oldVal[i]));
        int *dims = ARR_DIMS(DatumGetArrayTypeP(oldVal[i]));
        int *lbs = ARR_LBOUND(DatumGetArrayTypeP(oldVal[i]));

        coefficients_per_iteration[i] = PointerGetDatum(copyArray(oldVal[i]));
        derivatives_tally[i] = PointerGetDatum(initResult(ndim, dims, lbs));
    }

    // printf("Grad_desc_l3_internal copied first tuples\n");

    MemoryContextSwitchTo(per_expr_eval);

    // printf("Grad_desc_l3_internal switched MCTX to per_tuple\n");

    for (int it = 0; it < iterations; it++)
    {
        while (tuplestore_gettupleslot(ttsIn, true, false, slot))
        {
            heap_deform_tuple(slot->tts_tuple, inDesc, oldVal, oldIsNull);
            // printf("Grad_desc_l3_internal tuple deform done\n");

            for (int i = 0; i < num_atts; i++)
            {
                oldVal[i] = PointerGetDatum(copyArray(coefficients_per_iteration[i]));
            }

            // printf("Grad_desc_l3_internal oldVal filed with coefficients\n");

            for (int i = 0; i < inDesc->natts; i++)
            {
                derivatives[i] = createScalar(0.0);
            }

            // printf("Grad_desc_l3_internal created scalars\n");

            Datum result = derivefunc(&oldVal, derivatives);

            // printf("Grad_desc_l3_internal calculated lambda and derivs\n");

            for (int i = 0; i < num_atts; i++)
            {
                derivatives_tally[i] = matrix_add_inplace(derivatives_tally[i], derivatives[i]);
            }
            // printf("Grad_desc_l3_internal derive_tally add in place\n");

            MemoryContextResetAndDeleteChildren(per_expr_eval);
            // printf("Grad_desc_l3_internal reset MCTX\n");
        }
        tuplestore_rescan(ttsIn);

        // printf("Grad_desc_l3_internal finished all tuples for this iteration\n");

        for (int i = 0; i < num_atts; i++)
        {
            coefficients_per_iteration[i] = mat_apply_gradient(coefficients_per_iteration[i], derivatives_tally[i], learning_rate, batch_size);
            matrixSetValue(derivatives_tally[i], (float8)0.0);
        }
        // printf("Grad_desc_l3_internal mat_apply gradients for this iteration done\n");
    }
    MemoryContextResetAndDeleteChildren(per_expr_eval);
    MemoryContextSwitchTo(per_query_ctx);
    printf("Grad_desc_l3_internal switched back to MCTX per query\n");

    for (int i = 0; i < num_atts; i++)
    {
        replVal[i] = PointerGetDatum(copyArray(coefficients_per_iteration[i]));
        replIsNull[i] = false;
    }

    printf("Grad_desc_l3_internal before last heap_form\n");

    tuple = heap_form_tuple(outDesc, replVal, replIsNull);

    printf("Grad_desc_l3_internal heap_form worked\n");

    tuplestore_puttuple(tsOut, tuple);

    printf("Grad_desc_l3_internal tupleStore put tuple\n");

    rsinfo->returnMode = SFRM_Materialize;
    rsinfo->setResult = tsOut;
    rsinfo->setDesc = outDesc;

    printf("Grad_desc_l3_internal rsinfo was filled\n");

    MemoryContextSwitchTo(oldcontext);

    printf("Grad_desc_l3_internal MCTX was switched to old\n");

    return (Datum)0;
}

Datum gradient_descent_m_internal_l4(PG_FUNCTION_ARGS)
{
    float8 learning_rate = PG_GETARG_FLOAT8(5); // learning rate for gradient desc
    int batch_size = PG_GETARG_INT32(4);        // amount of tuples to be loaded during grad_desc
    int num_atts = PG_GETARG_INT32(3);          // number of independent variables per run
    int iterations = PG_GETARG_INT32(2);        // Number of iterations for grad_desc algorithm

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

    MemoryContext per_query_ctx = rsinfo->econtext->ecxt_per_query_memory;
    MemoryContext per_expr_eval = rsinfo->econtext->ecxt_per_tuple_memory;
    MemoryContext oldcontext = MemoryContextSwitchTo(per_query_ctx);

    Tuplestorestate *ttsIn = ((TypedTuplestore *)PG_GETARG_POINTER(0))->tuplestorestate;
    TupleTableSlot *slot = MakeTupleTableSlot(NULL);
    Tuplestorestate *tsOut = tuplestore_begin_heap(true, false, work_mem);
    int tupleStoreCount = (int)tuplestore_tuple_count(ttsIn);
    tuplestore_rescan(ttsIn);

    if (batch_size < 1 || batch_size > tupleStoreCount)
    {
        batch_size = tupleStoreCount;
    }

    TupleDesc outDesc = CreateTemplateTupleDesc(num_atts, false);
    for (int i = 0; i < num_atts; i++)
    {
        TupleDescCopyEntry(outDesc, (AttrNumber)(i + 1), inDesc, (AttrNumber)(i + 1));
    }

    HeapTuple tuple;

    bool *oldIsNull = (bool *)palloc_extended(inDesc->natts * sizeof(bool), (MCXT_ALLOC_ZERO));
    Datum *oldVal = (Datum *)palloc_extended(inDesc->natts * sizeof(Datum), (MCXT_ALLOC_ZERO));

    bool *replIsNull = (bool *)palloc_extended((num_atts) * sizeof(bool), (MCXT_ALLOC_ZERO));
    Datum *replVal = (Datum *)palloc_extended((num_atts) * sizeof(Datum), (MCXT_ALLOC_ZERO));

    Datum *derivatives = (Datum *)palloc_extended(inDesc->natts * sizeof(Datum), (MCXT_ALLOC_ZERO));             // Returned derivatives from autodiff
    Datum *coefficients_per_iteration = (Datum *)palloc_extended((num_atts) * sizeof(Datum), (MCXT_ALLOC_ZERO)); // coefficients of lambda_expr
    Datum *derivatives_tally = (Datum *)palloc_extended((num_atts) * sizeof(Datum), (MCXT_ALLOC_ZERO));          // all derivatives for a single iterations tallied up(indricetion through ArrayType pointers)

    if (tuplestore_gettupleslot(ttsIn, true, false, slot))
    {
        heap_deform_tuple(slot->tts_tuple, inDesc, oldVal, oldIsNull);
        tuplestore_rescan(ttsIn);
    }
    else
    {
        ereport(ERROR,
                (errcode(ERRCODE_ASSERT_FAILURE),
                 errmsg("Gradient Descent cannot work on empty tables!")));
    }

    for (int i = 0; i < num_atts; i++)
    {
        int ndim = ARR_NDIM(DatumGetArrayTypeP(oldVal[i]));
        int *dims = ARR_DIMS(DatumGetArrayTypeP(oldVal[i]));
        int *lbs = ARR_LBOUND(DatumGetArrayTypeP(oldVal[i]));

        coefficients_per_iteration[i] = PointerGetDatum(copyArray(oldVal[i]));
        derivatives_tally[i] = PointerGetDatum(initResult(ndim, dims, lbs));
    }

    MemoryContextSwitchTo(per_expr_eval);

    for (int it = 0; it < iterations; it++)
    {
        int internal_counter = 0;
        while (tuplestore_gettupleslot(ttsIn, true, false, slot))
        {
            heap_deform_tuple(slot->tts_tuple, inDesc, oldVal, oldIsNull);

            for (int i = 0; i < num_atts; i++)
            {
                oldVal[i] = PointerGetDatum(copyArray(coefficients_per_iteration[i]));
            }

            for (int i = 0; i < inDesc->natts; i++)
            {
                derivatives[i] = createScalar(0.0);
                // matrixPrint(DatumGetArrayTypeP(oldVal[i]));
            }

            Datum result = PG_SIMPLE_LAMBDA_INJECT_DERIV(&oldVal, derivatives, 0);

            for (int i = 0; i < num_atts; i++)
            {
                derivatives_tally[i] = matrix_add_inplace(derivatives_tally[i], derivatives[i]);
                // matrixPrint(derivatives[i]);
            }
            MemoryContextResetAndDeleteChildren(per_expr_eval);
        }
        tuplestore_rescan(ttsIn);

        for (int i = 0; i < num_atts; i++)
        {
            coefficients_per_iteration[i] = mat_apply_gradient(coefficients_per_iteration[i], derivatives_tally[i], learning_rate, batch_size);
            matrixSetValue(derivatives_tally[i], (float8)0.0);
        }
    }
    MemoryContextSwitchTo(per_query_ctx);

    for (int i = 0; i < num_atts; i++)
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

Datum gradient_descent_m_l1_2(PG_FUNCTION_ARGS)
{
    ReturnSetInfo *rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;
    LambdaExpr *lambda = PG_GETARG_LAMBDA(1);

    llvm_enter_tmp_context(rsinfo->econtext->ecxt_estate);
    ExecInitLambdaExpr((Node *)lambda, false, true);
    llvm_leave_tmp_context(rsinfo->econtext->ecxt_estate);

    return gradient_descent_m_internal_l1_2(fcinfo);
}

Datum gradient_descent_m_l3(PG_FUNCTION_ARGS)
{
    printf("Grad_desc_l3 begin\n");
    ReturnSetInfo *rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;
    LambdaExpr *lambda = PG_GETARG_LAMBDA(1);
    printf("Grad_desc_l3 got lambda\n");

    llvm_enter_tmp_context(rsinfo->econtext->ecxt_estate);
    LLVMJitContext *jitContext = (LLVMJitContext *)(rsinfo->econtext->ecxt_estate->es_jit);

    ExecInitLambdaExpr((Node *)lambda, true, true);
    printf("Grad_desc_l3 lambda was initiated\n");

    Datum (*compiled_func)(Datum **, Datum *);
    compiled_func = llvm_prepare_simple_expression_derivation(castNode(ExprState, lambda->exprstate));
    printf("Grad_desc_l3 lambda was compiled\n");

    llvm_leave_tmp_context(rsinfo->econtext->ecxt_estate);
    return gradient_descent_m_internal_l3(fcinfo, compiled_func);
}

Datum gradient_descent_m_l4(PG_FUNCTION_ARGS)
{
    ReturnSetInfo *rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;
    LambdaExpr *lambda = PG_GETARG_LAMBDA(1);

    llvm_enter_tmp_context(rsinfo->econtext->ecxt_estate);
    LLVMJitContext *jitContext = (LLVMJitContext *)(rsinfo->econtext->ecxt_estate->es_jit);

    ExecInitLambdaExpr((Node *)lambda, true, true);
    Datum (*compiled_func)(FunctionCallInfo);
    compiled_func = llvm_prepare_lambda_tablefunc(jitContext, "ext/gradient_desc_m_ext.bc", "gradient_descent_m_internal_l4", 1);

    llvm_leave_tmp_context(rsinfo->econtext->ecxt_estate);

    return compiled_func(fcinfo);
}

/*
int tuple_counter = 0; // number of already calculated samples in batch
        while (tuplestore_gettupleslot(ttsIn, true, false, slot))
        {
            if (batch_size <= tuple_counter)
            {
                // Calculate avg of tally and adjust coefficients
for (int it = 0; it < num_atts; it++)
{
    // updates inplace, but if scalar coefficients, returns new array
    coefficients_per_iteration[it] = mat_apply_gradient(coefficients_per_iteration[it], derivatives_tally[it], learning_rate, batch_size);
    derivatives_tally[it] = createScalar(0.0);
}
tuple_counter = 0;
}

heap_deform_tuple(slot->tts_tuple, inDesc, oldVal, oldIsNull);
if (unlikely(firstRun)) // will only be executed once, so 'unlikely' should improve perf
{
    firstRun = false;
    for (int it = 0; it < num_atts; it++)
    {
        // Set cooefficients/weights to input, so size does not have to be determined
        coefficients_per_iteration[it] = matrix_add_inplace(coefficients_per_iteration[it], oldVal[it]);
    }
}

for (int it = 0; it < num_atts; it++)
{
    // Set elements of tuple in slot to the correct cooefficients
    Datum tmp = createScalar(0.0);
    oldVal[it] = matrix_add_inplace(tmp, coefficients_per_iteration[it]);
}

for (int it = 0; it < inDesc->natts; it++)
{
    // Reset derivatives to avoid undefined behaviour
    derivatives[it] = createScalar(0.0);
}

Datum result = derivefunc(&oldVal, derivatives);

for (int it = 0; it < num_atts; it++)
{
    // tally up all elements per column
    derivatives_tally[it] = matrix_add_inplace(derivatives_tally[it], derivatives[it]);
}
tuple_counter++;
}
tuplestore_rescan(ttsIn);

// Calculate avg of tally and adjust coefficients
for (int it = 0; it < num_atts; it++)
{
    coefficients_per_iteration[it] = mat_apply_gradient(coefficients_per_iteration[it], derivatives_tally[it], learning_rate, batch_size);
    derivatives_tally[it] = createScalar(0.0);
}
*/