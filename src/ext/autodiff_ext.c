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

extern TupleDesc autodiff_record_type(List *args)
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
    strcpy(attrNamesMapped, "f(");
    for (int i = 0; i < inDesc->natts; i++)
    {
        strcat(attrNamesMapped, inDesc->attrs[i].attname.data);
        if (i < inDesc->natts - 1)
        {
            strcat(attrNamesMapped, ",");
        }
    }
    strcat(attrNamesMapped, ")");

    TupleDescInitEntry(outDesc, (AttrNumber)(inDesc->natts + 1), attrNamesMapped,
                       lambda->rettype, lambda->rettypmod, 0);

    for (int i = 0; i < inDesc->natts; i++)
    {
        char buffer[64];
        char *column_name = inDesc->attrs[i].attname.data;
        sprintf(buffer, "df/d%s", column_name);

        TupleDescInitEntry(outDesc, (AttrNumber)(inDesc->natts + 2 + i), buffer,
                           lambda->rettype, lambda->rettypmod, 0);
    }

    return outDesc;
}

PG_MODULE_MAGIC;
PG_FUNCTION_INFO_V1_RECTYPE(autodiff, autodiff_record_type);

Datum autodiff_internal(PG_FUNCTION_ARGS)
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

    for (int i = 0; i < inDesc->natts; i++)
    {
        TupleDescCopyEntry(outDesc, (AttrNumber)(i + 1), inDesc, (AttrNumber)(i + 1));
    }

    char attrNamesMapped[64];
    strcpy(attrNamesMapped, "f(");
    for (int i = 0; i < inDesc->natts; i++)
    {
        strcat(attrNamesMapped, inDesc->attrs[i].attname.data);
        if (i < inDesc->natts - 1)
        {
            strcat(attrNamesMapped, ",");
        }
    }
    strcat(attrNamesMapped, ")");

    TupleDescInitEntry(outDesc, (AttrNumber)(inDesc->natts + 1), attrNamesMapped,
                       lambda->rettype, lambda->rettypmod, 0);

    for (int i = 0; i < inDesc->natts; i++)
    {
        char buffer[64];
        char *column_name = inDesc->attrs[i].attname.data;
        sprintf(buffer, "df/d%s", column_name);

        TupleDescInitEntry(outDesc, (AttrNumber)(inDesc->natts + 2 + i), buffer,
                           lambda->rettype, lambda->rettypmod, 0);
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

        PG_LAMBDA_SETARG(lambda, 0, HeapTupleHeaderGetDatum(hdr));
        Datum result = PG_LAMBDA_DERIVE(lambda, &isnull, derivatives, inDesc->natts);

        //Begin testing
        // const char enumsAsNames[87][30] = {"EEOP_DONE", "EEOP_INNER_FETCHSOME", "EEOP_OUTER_FETCHSOME", "EEOP_SCAN_FETCHSOME", "EEOP_INNER_VAR", "EEOP_OUTER_VAR", "EEOP_SCAN_VAR", "EEOP_INNER_SYSVAR", "EEOP_OUTER_SYSVAR", "EEOP_SCAN_SYSVAR", "EEOP_WHOLEROW", "EEOP_ASSIGN_INNER_VAR", "EEOP_ASSIGN_OUTER_VAR", "EEOP_ASSIGN_SCAN_VAR", "EEOP_ASSIGN_TMP", "EEOP_ASSIGN_TMP_MAKE_RO", "EEOP_CONST", "EEOP_FUNCEXPR", "EEOP_FUNCEXPR_STRICT", "EEOP_FUNCEXPR_FUSAGE", "EEOP_FUNCEXPR_STRICT_FUSAGE", "EEOP_BOOL_AND_STEP_FIRST", "EEOP_BOOL_AND_STEP", "EEOP_BOOL_AND_STEP_LAST", "EEOP_BOOL_OR_STEP_FIRST", "EEOP_BOOL_OR_STEP", "EEOP_BOOL_OR_STEP_LAST", "EEOP_BOOL_NOT_STEP", "EEOP_QUAL", "EEOP_JUMP", "EEOP_JUMP_IF_NULL", "EEOP_JUMP_IF_NOT_NULL", "EEOP_JUMP_IF_NOT_TRUE", "EEOP_NULLTEST_ISNULL", "EEOP_NULLTEST_ISNOTNULL", "EEOP_NULLTEST_ROWISNULL", "EEOP_NULLTEST_ROWISNOTNULL", "EEOP_BOOLTEST_IS_TRUE", "EEOP_BOOLTEST_IS_NOT_TRUE", "EEOP_BOOLTEST_IS_FALSE", "EEOP_BOOLTEST_IS_NOT_FALSE", "EEOP_PARAM_EXEC", "EEOP_PARAM_EXTERN", "EEOP_PARAM_CALLBACK", "EEOP_CASE_TESTVAL", "EEOP_MAKE_READONLY", "EEOP_IOCOERCE", "EEOP_DISTINCT", "EEOP_NOT_DISTINCT", "EEOP_NULLIF", "EEOP_SQLVALUEFUNCTION", "EEOP_CURRENTOFEXPR", "EEOP_NEXTVALUEEXPR", "EEOP_ARRAYEXPR", "EEOP_ARRAYCOERCE", "EEOP_ROW", "EEOP_ROWCOMPARE_STEP", "EEOP_ROWCOMPARE_FINAL", "EEOP_MINMAX", "EEOP_FIELDSELECT", "EEOP_FIELDSTORE_DEFORM", "EEOP_FIELDSTORE_FORM", "EEOP_ARRAYREF_SUBSCRIPT", "EEOP_ARRAYREF_OLD", "EEOP_ARRAYREF_ASSIGN", "EEOP_ARRAYREF_FETCH", "EEOP_DOMAIN_TESTVAL", "EEOP_DOMAIN_NOTNULL", "EEOP_DOMAIN_CHECK", "EEOP_CONVERT_ROWTYPE", "EEOP_SCALARARRAYOP", "EEOP_XMLEXPR", "EEOP_AGGREF", "EEOP_GROUPING_FUNC", "EEOP_WINDOW_FUNC", "EEOP_SUBPLAN", "EEOP_ALTERNATIVE_SUBPLAN", "EEOP_AGG_STRICT_DESERIALIZE", "EEOP_AGG_DESERIALIZE", "EEOP_AGG_STRICT_INPUT_CHECK", "EEOP_AGG_INIT_TRANS", "EEOP_AGG_STRICT_TRANS_CHECK", "EEOP_AGG_PLAIN_TRANS_BYVAL", "EEOP_AGG_PLAIN_TRANS", "EEOP_AGG_ORDERED_TRANS_DATUM", "EEOP_AGG_ORDERED_TRANS_TUPLE", "EEOP_LAST"};
        // ExprState *state = castNode(ExprState, lambda->exprstate);
        // for (int i = 0; i < state->steps_len; i++)
        // {
        //     printf("\n%i: %s, %lu", i, enumsAsNames[(long)state->steps[i].opcode], (long)state->steps[i].opcode);
        //     if ((long)state->steps[i].opcode != 0) {
        //         printf("\t\tThis is the resultValue: ");
        //         printf("%f", DatumGetFloat8(*state->steps[i].resvalue));
        //     }
        //     if ((long)state->steps[i].opcode == 18) //EEOP_FUNEXPR_STRICT
        //     {
        //         ExprEvalStep evalStep = state->steps[i];
        //         printf("\n%u <- OId of FuncExpr", evalStep.d.func.finfo->fn_oid);
        //     }
        // }
        // printf("\nEnd of tupel\n\n");
        //End testing

        for (int i = 0; i < inDesc->natts; i++)
        {
            replVal[i] = val_ptr[i];
            replIsNull[i] = null_ptr[i];
        }

        replVal[inDesc->natts] = result;
        replIsNull[inDesc->natts] = false;

        for(int i = 0; i < inDesc->natts; i++) {
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

Datum autodiff(PG_FUNCTION_ARGS)
{
    ReturnSetInfo *rsinfo = (ReturnSetInfo *)fcinfo->resultinfo;
    LambdaExpr *lambda = PG_GETARG_LAMBDA(1);

    llvm_enter_tmp_context(rsinfo->econtext->ecxt_estate);
    ExecInitLambdaExpr((Node *)lambda, false);
    llvm_leave_tmp_context(rsinfo->econtext->ecxt_estate);

    return autodiff_internal(fcinfo);
}