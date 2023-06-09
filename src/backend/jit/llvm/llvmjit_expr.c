/*-------------------------------------------------------------------------
 *
 * llvmjit_expr.c
 *	  JIT compile expressions.
 *
 * Portions Copyright (c) 1996-2017, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 *
 * IDENTIFICATION
 *	  src/backend/jit/llvm/llvmjit_expr.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include <llvm-c/Core.h>
#include <llvm-c/Target.h>

#include "access/htup_details.h"
#include "access/nbtree.h"
#include "access/tupconvert.h"
#include "catalog/objectaccess.h"
#include "catalog/pg_type.h"
#include "executor/execdebug.h"
#include "executor/nodeAgg.h"
#include "executor/nodeSubplan.h"
#include "executor/execExpr.h"
#include "funcapi.h"
#include "jit/llvmjit.h"
#include "jit/llvmjit_emit.h"
#include "miscadmin.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/planner.h"
#include "parser/parse_coerce.h"
#include "parser/parsetree.h"
#include "pgstat.h"
#include "utils/acl.h"
#include "utils/builtins.h"
#include "utils/date.h"
#include "utils/fmgrtab.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/timestamp.h"
#include "utils/typcache.h"
#include "utils/xml.h"

static Datum ExecRunCompiledExpr(ExprState *state, ExprContext *econtext, bool *isNull);
static Datum ExecRunCompiledExprDeriv(ExprState *state, ExprContext *econtext, bool *isNull, Datum *derivatives);

static LLVMValueRef BuildV1Call(LLVMJitContext *context, LLVMBuilderRef b,
								LLVMModuleRef mod, FunctionCallInfo fcinfo,
								LLVMValueRef *v_fcinfo_isnull);
static void build_EvalXFunc(LLVMBuilderRef b, LLVMModuleRef mod,
							const char *funcname,
							LLVMValueRef v_state, LLVMValueRef v_econtext,
							ExprEvalStep *op);
static LLVMValueRef build_EvalCFunc(LLVMBuilderRef b, LLVMModuleRef mod,
									const char *funcname, LLVMValueRef *params,
									LLVMTypeRef *param_types, LLVMTypeRef rettype, int nparams);
static LLVMValueRef create_LifetimeEnd(LLVMModuleRef mod);
static int llvm_compile_expr_deriv_subtree(LLVMBuilderRef b, LLVMModuleRef mod, ExprState *state, 
										   int fetchIndex, LLVMValueRef seed, LLVMValueRef derivatives);
static int llvm_compile_simple_deriv_subtree(LLVMBuilderRef b, LLVMModuleRef mod, ExprState *state,
											 int fetchIndex, LLVMValueRef seed,
											 LLVMValueRef derivatives, LLVMValueRef *funcVals, int *intermediates_pointer);

#define CASE_FLOAT8_CFUNC_2ARG(oid, fname)                                                                   \
	case oid:                                                                                                \
	{                                                                                                        \
		LLVMTypeRef types[2];                                                                                \
		LLVMValueRef params[2];                                                                              \
		numparams = 2;                                                                                       \
		types[0] = LLVMDoubleType();                                                                         \
		types[1] = LLVMDoubleType();                                                                         \
		params[0] = l_as_float8(b, registers[registerPointer - 2]);                                          \
		params[1] = l_as_float8(b, registers[registerPointer - 1]);                                          \
                                                                                                             \
		opres = build_EvalCFunc(b, mod, fname, (LLVMValueRef *)&params, (LLVMTypeRef *)&types, types[0], 2); \
		break;                                                                                               \
	}

#define CASE_INT4_CFUNC_2ARG(oid, fname)                                                                     \
	case oid:                                                                                                \
	{                                                                                                        \
		LLVMTypeRef types[2];                                                                                \
		LLVMValueRef params[2];                                                                              \
		numparams = 2;                                                                                       \
		types[0] = LLVMInt32Type();                                                                          \
		types[1] = LLVMInt32Type();                                                                          \
		params[0] = l_as_int4(b, registers[registerPointer - 2]);                                            \
		params[1] = l_as_int4(b, registers[registerPointer - 1]);                                            \
                                                                                                             \
		opres = build_EvalCFunc(b, mod, fname, (LLVMValueRef *)&params, (LLVMTypeRef *)&types, types[0], 2); \
		break;                                                                                               \
	}

#define CASE_FLOAT8_CFUNC_1ARG(oid, fname)                              \
	case oid:                                                           \
	{                                                                   \
		LLVMTypeRef types[1];                                           \
		LLVMValueRef params[1];                                         \
		numparams = 1;                                                  \
		types[0] = LLVMDoubleType();                                    \
		params[0] = l_as_float8(b, registers[registerPointer - 1]);     \
                                                                        \
		opres = build_EvalCFunc(b, mod, fname, (LLVMValueRef *)&params, \
								(LLVMTypeRef *)&types, types[0], 1);    \
		break;                                                          \
	}

#define CASE_INT4_CFUNC_1ARG(oid, fname)                                \
	case oid:                                                           \
	{                                                                   \
		LLVMTypeRef types[1];                                           \
		LLVMValueRef params[1];                                         \
		numparams = 1;                                                  \
		types[0] = LLVMInt32Type();                                     \
		params[0] = l_as_int4(b, registers[registerPointer - 1]);       \
                                                                        \
		opres = build_EvalCFunc(b, mod, fname, (LLVMValueRef *)&params, \
								(LLVMTypeRef *)&types, types[0], 1);    \
		break;                                                          \
	}

/*
 * JIT compile expression.
 */
bool llvm_compile_expr(ExprState *state)
{
	PlanState *parent = state->parent;
	int i;
	char *funcname;

	LLVMJitContext *context = NULL;

	LLVMBuilderRef b;
	LLVMModuleRef mod;
	LLVMTypeRef eval_sig;
	LLVMValueRef eval_fn;
	LLVMBasicBlockRef entry;
	LLVMBasicBlockRef *opblocks;

	/* state itself */
	LLVMValueRef v_state;
	LLVMValueRef v_econtext;

	/* returnvalue */
	LLVMValueRef v_isnullp;

	/* tmp vars in state */
	LLVMValueRef v_tmpvaluep;
	LLVMValueRef v_tmpisnullp;

	/* slots */
	LLVMValueRef v_innerslot;
	LLVMValueRef v_outerslot;
	LLVMValueRef v_scanslot;
	LLVMValueRef v_resultslot;

	/* nulls/values of slots */
	LLVMValueRef v_innervalues;
	LLVMValueRef v_innernulls;
	LLVMValueRef v_outervalues;
	LLVMValueRef v_outernulls;
	LLVMValueRef v_scanvalues;
	LLVMValueRef v_scannulls;
	LLVMValueRef v_resultvalues;
	LLVMValueRef v_resultnulls;

	/* stuff in econtext */
	LLVMValueRef v_aggvalues;
	LLVMValueRef v_aggnulls;

	instr_time starttime;
	instr_time endtime;

	llvm_enter_fatal_on_oom();

	/* get or create JIT context */
	if (parent && parent->state->es_jit)
	{
		context = (LLVMJitContext *)parent->state->es_jit;
	}
	else
	{
		context = llvm_create_context(parent->state->es_jit_flags);

		if (parent)
		{
			parent->state->es_jit = &context->base;
		}
	}

	INSTR_TIME_SET_CURRENT(starttime);

	mod = llvm_mutable_module(context);

	b = LLVMCreateBuilder();

	funcname = llvm_expand_funcname(context, "evalexpr", false);

	/* Create the signature and function */
	{
		LLVMTypeRef param_types[3];

		param_types[0] = l_ptr(StructExprState);   /* state */
		param_types[1] = l_ptr(StructExprContext); /* econtext */
		param_types[2] = l_ptr(TypeParamBool);	   /* isnull */

		eval_sig = LLVMFunctionType(TypeSizeT,
									param_types, lengthof(param_types),
									false);
	}
	eval_fn = LLVMAddFunction(mod, funcname, eval_sig);
	LLVMSetLinkage(eval_fn, LLVMExternalLinkage);
	LLVMSetVisibility(eval_fn, LLVMDefaultVisibility);
	llvm_copy_attributes(AttributeTemplate, eval_fn);

	entry = LLVMAppendBasicBlock(eval_fn, "entry");

	/* build state */
	v_state = LLVMGetParam(eval_fn, 0);
	v_econtext = LLVMGetParam(eval_fn, 1);
	v_isnullp = LLVMGetParam(eval_fn, 2);

	LLVMPositionBuilderAtEnd(b, entry);

	v_tmpvaluep = LLVMBuildStructGEP(b, v_state,
									 FIELDNO_EXPRSTATE_RESVALUE,
									 "v.state.resvalue");
	v_tmpisnullp = LLVMBuildStructGEP(b, v_state,
									  FIELDNO_EXPRSTATE_RESNULL,
									  "v.state.resnull");

	/* build global slots */
	v_scanslot = l_load_struct_gep(b, v_econtext,
								   FIELDNO_EXPRCONTEXT_SCANTUPLE,
								   "v_scanslot");
	v_innerslot = l_load_struct_gep(b, v_econtext,
									FIELDNO_EXPRCONTEXT_INNERTUPLE,
									"v_innerslot");
	v_outerslot = l_load_struct_gep(b, v_econtext,
									FIELDNO_EXPRCONTEXT_OUTERTUPLE,
									"v_outerslot");
	v_resultslot = l_load_struct_gep(b, v_state,
									 FIELDNO_EXPRSTATE_RESULTSLOT,
									 "v_resultslot");

	/* build global values/isnull pointers */
	v_scanvalues = l_load_struct_gep(b, v_scanslot,
									 FIELDNO_TUPLETABLESLOT_VALUES,
									 "v_scanvalues");
	v_scannulls = l_load_struct_gep(b, v_scanslot,
									FIELDNO_TUPLETABLESLOT_ISNULL,
									"v_scannulls");
	v_innervalues = l_load_struct_gep(b, v_innerslot,
									  FIELDNO_TUPLETABLESLOT_VALUES,
									  "v_innervalues");
	v_innernulls = l_load_struct_gep(b, v_innerslot,
									 FIELDNO_TUPLETABLESLOT_ISNULL,
									 "v_innernulls");
	v_outervalues = l_load_struct_gep(b, v_outerslot,
									  FIELDNO_TUPLETABLESLOT_VALUES,
									  "v_outervalues");
	v_outernulls = l_load_struct_gep(b, v_outerslot,
									 FIELDNO_TUPLETABLESLOT_ISNULL,
									 "v_outernulls");
	v_resultvalues = l_load_struct_gep(b, v_resultslot,
									   FIELDNO_TUPLETABLESLOT_VALUES,
									   "v_resultvalues");
	v_resultnulls = l_load_struct_gep(b, v_resultslot,
									  FIELDNO_TUPLETABLESLOT_ISNULL,
									  "v_resultnulls");

	/* aggvalues/aggnulls */
	v_aggvalues = l_load_struct_gep(b, v_econtext,
									FIELDNO_EXPRCONTEXT_AGGVALUES,
									"v.econtext.aggvalues");
	v_aggnulls = l_load_struct_gep(b, v_econtext,
								   FIELDNO_EXPRCONTEXT_AGGNULLS,
								   "v.econtext.aggnulls");

	/* allocate blocks for each op upfront, so we can do jumps easily */
	opblocks = palloc(sizeof(LLVMBasicBlockRef) * state->steps_len);
	for (i = 0; i < state->steps_len; i++)
		opblocks[i] = l_bb_append_v(eval_fn, "b.op.%d.start", i);

	/* jump from entry to first block */
	LLVMBuildBr(b, opblocks[0]);

	for (i = 0; i < state->steps_len; i++)
	{
		ExprEvalStep *op;
		ExprEvalOp opcode;
		LLVMValueRef v_resvaluep;
		LLVMValueRef v_resnullp;

		LLVMPositionBuilderAtEnd(b, opblocks[i]);

		op = &state->steps[i];
		opcode = ExecEvalStepOp(state, op);

		v_resvaluep = l_ptr_const(op->resvalue, l_ptr(TypeSizeT));
		v_resnullp = l_ptr_const(op->resnull, l_ptr(TypeStorageBool));

		switch (opcode)
		{
		case EEOP_DONE:
		{
			LLVMValueRef v_tmpisnull,
				v_tmpvalue;

			v_tmpvalue = LLVMBuildLoad(b, v_tmpvaluep, "");
			v_tmpisnull = LLVMBuildLoad(b, v_tmpisnullp, "");
			v_tmpisnull =
				LLVMBuildTrunc(b, v_tmpisnull, TypeParamBool, "");

			LLVMBuildStore(b, v_tmpisnull, v_isnullp);

			LLVMBuildRet(b, v_tmpvalue);
			break;
		}

		case EEOP_INNER_FETCHSOME:
		case EEOP_OUTER_FETCHSOME:
		case EEOP_SCAN_FETCHSOME:
		{
			TupleDesc desc = NULL;
			LLVMValueRef v_slot;
			LLVMBasicBlockRef b_fetch;
			LLVMValueRef v_nvalid;

			b_fetch = l_bb_before_v(opblocks[i + 1],
									"op.%d.fetch", i);

			if (op->d.fetch.known_desc)
				desc = op->d.fetch.known_desc;

			if (opcode == EEOP_INNER_FETCHSOME)
			{
				PlanState *is = innerPlanState(parent);

				v_slot = v_innerslot;

				if (!desc &&
					is &&
					is->ps_ResultTupleSlot &&
					is->ps_ResultTupleSlot->tts_fixedTupleDescriptor)
					desc = is->ps_ResultTupleSlot->tts_tupleDescriptor;
			}
			else if (opcode == EEOP_OUTER_FETCHSOME)
			{
				PlanState *os = outerPlanState(parent);

				v_slot = v_outerslot;

				if (!desc &&
					os &&
					os->ps_ResultTupleSlot &&
					os->ps_ResultTupleSlot->tts_fixedTupleDescriptor)
					desc = os->ps_ResultTupleSlot->tts_tupleDescriptor;
			}
			else
			{
				v_slot = v_scanslot;
				if (!desc && parent)
					desc = parent->scandesc;
			}

			/*
					 * Check if all required attributes are available, or
					 * whether deforming is required.
					 */
			v_nvalid =
				l_load_struct_gep(b, v_slot,
								  FIELDNO_TUPLETABLESLOT_NVALID,
								  "");
			LLVMBuildCondBr(b,
							LLVMBuildICmp(b, LLVMIntUGE, v_nvalid,
										  l_int32_const(op->d.fetch.last_var),
										  ""),
							opblocks[i + 1], b_fetch);

			LLVMPositionBuilderAtEnd(b, b_fetch);

			/*
					 * If the tupledesc of the to-be-deformed tuple is known,
					 * and JITing of deforming is enabled, build deform
					 * function specific to tupledesc and the exact number of
					 * to-be-extracted attributes.
					 */
			if (desc && (context->base.flags & PGJIT_DEFORM))
			{
				LLVMValueRef params[1];
				LLVMValueRef l_jit_deform;

				l_jit_deform =
					slot_compile_deform(context, desc,
										op->d.fetch.last_var);
				params[0] = v_slot;

				LLVMBuildCall(b, l_jit_deform,
							  params, lengthof(params), "");
			}
			else
			{
				LLVMValueRef params[2];

				params[0] = v_slot;
				params[1] = l_int32_const(op->d.fetch.last_var);

				LLVMBuildCall(b,
							  llvm_get_decl(mod, FuncSlotGetsomeattrs),
							  params, lengthof(params), "");
			}

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_INNER_VAR:
		case EEOP_OUTER_VAR:
		case EEOP_SCAN_VAR:
		{
			LLVMValueRef value,
				isnull;
			LLVMValueRef v_attnum;
			LLVMValueRef v_values;
			LLVMValueRef v_nulls;

			if (opcode == EEOP_INNER_VAR)
			{
				v_values = v_innervalues;
				v_nulls = v_innernulls;
			}
			else if (opcode == EEOP_OUTER_VAR)
			{
				v_values = v_outervalues;
				v_nulls = v_outernulls;
			}
			else
			{
				v_values = v_scanvalues;
				v_nulls = v_scannulls;
			}

			v_attnum = l_int32_const(op->d.var.attnum);
			value = l_load_gep1(b, v_values, v_attnum, "");
			isnull = l_load_gep1(b, v_nulls, v_attnum, "");
			LLVMBuildStore(b, value, v_resvaluep);
			LLVMBuildStore(b, isnull, v_resnullp);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_INNER_SYSVAR:
		case EEOP_OUTER_SYSVAR:
		case EEOP_SCAN_SYSVAR:
		{
			int attnum = op->d.var.attnum;
			LLVMValueRef v_attnum;
			LLVMValueRef v_tuple;
			LLVMValueRef v_tupleDescriptor;
			LLVMValueRef v_params[4];
			LLVMValueRef v_syscol;
			LLVMValueRef v_slot;

			if (opcode == EEOP_INNER_SYSVAR)
				v_slot = v_innerslot;
			else if (opcode == EEOP_OUTER_SYSVAR)
				v_slot = v_outerslot;
			else
				v_slot = v_scanslot;

			Assert(op->d.var.attnum < 0);

			v_tuple = l_load_struct_gep(b, v_slot,
										FIELDNO_TUPLETABLESLOT_TUPLE,
										"v.tuple");

			/*
					 * Could optimize this a bit for fixed descriptors, but
					 * this shouldn't be that critical a path.
					 */
			v_tupleDescriptor =
				l_load_struct_gep(b, v_slot,
								  FIELDNO_TUPLETABLESLOT_TUPLEDESCRIPTOR,
								  "v.tupledesc");
			v_attnum = l_int32_const(attnum);

			v_params[0] = v_tuple;
			v_params[1] = v_attnum;
			v_params[2] = v_tupleDescriptor;
			v_params[3] = v_resnullp;
			v_syscol = LLVMBuildCall(b,
									 llvm_get_decl(mod, FuncHeapGetsysattr),
									 v_params, lengthof(v_params),
									 "");
			LLVMBuildStore(b, v_syscol, v_resvaluep);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_WHOLEROW:
			build_EvalXFunc(b, mod, "ExecEvalWholeRowVar",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_ASSIGN_INNER_VAR:
		case EEOP_ASSIGN_OUTER_VAR:
		case EEOP_ASSIGN_SCAN_VAR:
		{
			LLVMValueRef v_value,
				v_isnull;
			LLVMValueRef v_rvaluep,
				v_risnullp;
			LLVMValueRef v_attnum,
				v_resultnum;
			LLVMValueRef v_values;
			LLVMValueRef v_nulls;

			if (opcode == EEOP_ASSIGN_INNER_VAR)
			{
				v_values = v_innervalues;
				v_nulls = v_innernulls;
			}
			else if (opcode == EEOP_ASSIGN_OUTER_VAR)
			{
				v_values = v_outervalues;
				v_nulls = v_outernulls;
			}
			else
			{
				v_values = v_scanvalues;
				v_nulls = v_scannulls;
			}

			/* load data */
			v_attnum = l_int32_const(op->d.assign_var.attnum);
			v_value = l_load_gep1(b, v_values, v_attnum, "");
			v_isnull = l_load_gep1(b, v_nulls, v_attnum, "");

			/* compute addresses of targets */
			v_resultnum = l_int32_const(op->d.assign_var.resultnum);
			v_rvaluep = LLVMBuildGEP(b, v_resultvalues,
									 &v_resultnum, 1, "");
			v_risnullp = LLVMBuildGEP(b, v_resultnulls,
									  &v_resultnum, 1, "");

			/* and store */
			LLVMBuildStore(b, v_value, v_rvaluep);
			LLVMBuildStore(b, v_isnull, v_risnullp);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_ASSIGN_TMP:
		{
			LLVMValueRef v_value,
				v_isnull;
			LLVMValueRef v_rvaluep,
				v_risnullp;
			LLVMValueRef v_resultnum;
			size_t resultnum = op->d.assign_tmp.resultnum;

			/* load data */
			v_value = LLVMBuildLoad(b, v_tmpvaluep, "");
			v_isnull = LLVMBuildLoad(b, v_tmpisnullp, "");

			/* compute addresses of targets */
			v_resultnum = l_int32_const(resultnum);
			v_rvaluep =
				LLVMBuildGEP(b, v_resultvalues, &v_resultnum, 1, "");
			v_risnullp =
				LLVMBuildGEP(b, v_resultnulls, &v_resultnum, 1, "");

			/* and store */
			LLVMBuildStore(b, v_value, v_rvaluep);
			LLVMBuildStore(b, v_isnull, v_risnullp);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_ASSIGN_TMP_MAKE_RO:
		{
			LLVMBasicBlockRef b_notnull;
			LLVMValueRef v_params[1];
			LLVMValueRef v_ret;
			LLVMValueRef v_value,
				v_isnull;
			LLVMValueRef v_rvaluep,
				v_risnullp;
			LLVMValueRef v_resultnum;
			size_t resultnum = op->d.assign_tmp.resultnum;

			b_notnull = l_bb_before_v(opblocks[i + 1],
									  "op.%d.assign_tmp.notnull", i);

			/* load data */
			v_value = LLVMBuildLoad(b, v_tmpvaluep, "");
			v_isnull = LLVMBuildLoad(b, v_tmpisnullp, "");

			/* compute addresses of targets */
			v_resultnum = l_int32_const(resultnum);
			v_rvaluep = LLVMBuildGEP(b, v_resultvalues,
									 &v_resultnum, 1, "");
			v_risnullp = LLVMBuildGEP(b, v_resultnulls,
									  &v_resultnum, 1, "");

			/* store nullness */
			LLVMBuildStore(b, v_isnull, v_risnullp);

			/* check if value is NULL */
			LLVMBuildCondBr(b,
							LLVMBuildICmp(b, LLVMIntEQ, v_isnull,
										  l_sbool_const(0), ""),
							b_notnull, opblocks[i + 1]);

			/* if value is not null, convert to RO datum */
			LLVMPositionBuilderAtEnd(b, b_notnull);
			v_params[0] = v_value;
			v_ret =
				LLVMBuildCall(b,
							  llvm_get_decl(mod, FuncMakeExpandedObjectReadOnlyInternal),
							  v_params, lengthof(v_params), "");

			/* store value */
			LLVMBuildStore(b, v_ret, v_rvaluep);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_CONST:
		{
			LLVMValueRef v_constvalue,
				v_constnull;

			v_constvalue = l_sizet_const(op->d.constval.value);
			v_constnull = l_sbool_const(op->d.constval.isnull);

			LLVMBuildStore(b, v_constvalue, v_resvaluep);
			LLVMBuildStore(b, v_constnull, v_resnullp);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_FUNCEXPR_STRICT:
		{
			FunctionCallInfo fcinfo = op->d.func.fcinfo_data;
			LLVMBasicBlockRef b_nonull;
			int argno;
			LLVMValueRef v_fcinfo;
			LLVMValueRef v_argnullp;
			LLVMBasicBlockRef *b_checkargnulls;

			/*
					 * Block for the actual function call, if args are
					 * non-NULL.
					 */
			b_nonull = l_bb_before_v(opblocks[i + 1],
									 "b.%d.no-null-args", i);

			/* should make sure they're optimized beforehand */
			if (op->d.func.nargs == 0)
				elog(ERROR, "argumentless strict functions are pointless");

			v_fcinfo =
				l_ptr_const(fcinfo, l_ptr(StructFunctionCallInfoData));

			v_argnullp =
				LLVMBuildStructGEP(b,
								   v_fcinfo,
								   FIELDNO_FUNCTIONCALLINFODATA_ARGNULL,
								   "v_argnullp");

			/*
					 * set resnull to true, if the function is actually
					 * called, it'll be reset
					 */
			LLVMBuildStore(b, l_sbool_const(1), v_resnullp);

			/* create blocks for checking args, one for each */
			b_checkargnulls =
				palloc(sizeof(LLVMBasicBlockRef *) * op->d.func.nargs);
			for (argno = 0; argno < op->d.func.nargs; argno++)
				b_checkargnulls[argno] =
					l_bb_before_v(b_nonull, "b.%d.isnull.%d", i, argno);

			/* jump to check of first argument */
			LLVMBuildBr(b, b_checkargnulls[0]);

			/* check each arg for NULLness */
			for (argno = 0; argno < op->d.func.nargs; argno++)
			{
				LLVMValueRef v_argisnull;
				LLVMBasicBlockRef b_argnotnull;

				LLVMPositionBuilderAtEnd(b, b_checkargnulls[argno]);

				/* compute block to jump to if argument is not null */
				if (argno + 1 == op->d.func.nargs)
					b_argnotnull = b_nonull;
				else
					b_argnotnull = b_checkargnulls[argno + 1];

				/* and finally load & check NULLness of arg */
				v_argisnull = l_load_struct_gep(b, v_argnullp,
												argno, "");
				LLVMBuildCondBr(b,
								LLVMBuildICmp(b, LLVMIntEQ,
											  v_argisnull,
											  l_sbool_const(1),
											  ""),
								opblocks[i + 1],
								b_argnotnull);
			}

			LLVMPositionBuilderAtEnd(b, b_nonull);
		}
			/* FALLTHROUGH */

		case EEOP_FUNCEXPR:
		{
			FunctionCallInfo fcinfo = op->d.func.fcinfo_data;
			LLVMValueRef v_fcinfo_isnull;
			LLVMValueRef v_retval;

			v_retval = BuildV1Call(context, b, mod, fcinfo,
								   &v_fcinfo_isnull);
			LLVMBuildStore(b, v_retval, v_resvaluep);
			LLVMBuildStore(b, v_fcinfo_isnull, v_resnullp);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_FUNCEXPR_FUSAGE:
			build_EvalXFunc(b, mod, "ExecEvalFuncExprFusage",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_FUNCEXPR_STRICT_FUSAGE:
			build_EvalXFunc(b, mod, "ExecEvalFuncExprStrictFusage",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_BOOL_AND_STEP_FIRST:
		{
			LLVMValueRef v_boolanynullp;

			v_boolanynullp = l_ptr_const(op->d.boolexpr.anynull,
										 l_ptr(TypeStorageBool));
			LLVMBuildStore(b, l_sbool_const(0), v_boolanynullp);
		}
			/* FALLTHROUGH */

			/*
				 * Treat them the same for now, optimizer can remove
				 * redundancy. Could be worthwhile to optimize during emission
				 * though.
				 */
		case EEOP_BOOL_AND_STEP_LAST:
		case EEOP_BOOL_AND_STEP:
		{
			LLVMValueRef v_boolvalue;
			LLVMValueRef v_boolnull;
			LLVMValueRef v_boolanynullp,
				v_boolanynull;
			LLVMBasicBlockRef b_boolisnull;
			LLVMBasicBlockRef b_boolcheckfalse;
			LLVMBasicBlockRef b_boolisfalse;
			LLVMBasicBlockRef b_boolcont;
			LLVMBasicBlockRef b_boolisanynull;

			b_boolisnull = l_bb_before_v(opblocks[i + 1],
										 "b.%d.boolisnull", i);
			b_boolcheckfalse = l_bb_before_v(opblocks[i + 1],
											 "b.%d.boolcheckfalse", i);
			b_boolisfalse = l_bb_before_v(opblocks[i + 1],
										  "b.%d.boolisfalse", i);
			b_boolisanynull = l_bb_before_v(opblocks[i + 1],
											"b.%d.boolisanynull", i);
			b_boolcont = l_bb_before_v(opblocks[i + 1],
									   "b.%d.boolcont", i);

			v_boolanynullp = l_ptr_const(op->d.boolexpr.anynull,
										 l_ptr(TypeStorageBool));

			v_boolnull = LLVMBuildLoad(b, v_resnullp, "");
			v_boolvalue = LLVMBuildLoad(b, v_resvaluep, "");

			/* set resnull to boolnull */
			LLVMBuildStore(b, v_boolnull, v_resnullp);
			/* set revalue to boolvalue */
			LLVMBuildStore(b, v_boolvalue, v_resvaluep);

			/* check if current input is NULL */
			LLVMBuildCondBr(b,
							LLVMBuildICmp(b, LLVMIntEQ, v_boolnull,
										  l_sbool_const(1), ""),
							b_boolisnull,
							b_boolcheckfalse);

			/* build block that sets anynull */
			LLVMPositionBuilderAtEnd(b, b_boolisnull);
			/* set boolanynull to true */
			LLVMBuildStore(b, l_sbool_const(1), v_boolanynullp);
			/* and jump to next block */
			LLVMBuildBr(b, b_boolcont);

			/* build block checking for false */
			LLVMPositionBuilderAtEnd(b, b_boolcheckfalse);
			LLVMBuildCondBr(b,
							LLVMBuildICmp(b, LLVMIntEQ, v_boolvalue,
										  l_sizet_const(0), ""),
							b_boolisfalse,
							b_boolcont);

			/*
					 * Build block handling FALSE. Value is false, so short
					 * circuit.
					 */
			LLVMPositionBuilderAtEnd(b, b_boolisfalse);
			/* result is already set to FALSE, need not change it */
			/* and jump to the end of the AND expression */
			LLVMBuildBr(b, opblocks[op->d.boolexpr.jumpdone]);

			/* Build block that continues if bool is TRUE. */
			LLVMPositionBuilderAtEnd(b, b_boolcont);

			v_boolanynull = LLVMBuildLoad(b, v_boolanynullp, "");

			/* set value to NULL if any previous values were NULL */
			LLVMBuildCondBr(b,
							LLVMBuildICmp(b, LLVMIntEQ, v_boolanynull,
										  l_sbool_const(0), ""),
							opblocks[i + 1], b_boolisanynull);

			LLVMPositionBuilderAtEnd(b, b_boolisanynull);
			/* set resnull to true */
			LLVMBuildStore(b, l_sbool_const(1), v_resnullp);
			/* reset resvalue */
			LLVMBuildStore(b, l_sizet_const(0), v_resvaluep);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}
		case EEOP_BOOL_OR_STEP_FIRST:
		{
			LLVMValueRef v_boolanynullp;

			v_boolanynullp = l_ptr_const(op->d.boolexpr.anynull,
										 l_ptr(TypeStorageBool));
			LLVMBuildStore(b, l_sbool_const(0), v_boolanynullp);
		}
			/* FALLTHROUGH */

			/*
				 * Treat them the same for now, optimizer can remove
				 * redundancy. Could be worthwhile to optimize during emission
				 * though.
				 */
		case EEOP_BOOL_OR_STEP_LAST:
		case EEOP_BOOL_OR_STEP:
		{
			LLVMValueRef v_boolvalue;
			LLVMValueRef v_boolnull;
			LLVMValueRef v_boolanynullp,
				v_boolanynull;

			LLVMBasicBlockRef b_boolisnull;
			LLVMBasicBlockRef b_boolchecktrue;
			LLVMBasicBlockRef b_boolistrue;
			LLVMBasicBlockRef b_boolcont;
			LLVMBasicBlockRef b_boolisanynull;

			b_boolisnull = l_bb_before_v(opblocks[i + 1],
										 "b.%d.boolisnull", i);
			b_boolchecktrue = l_bb_before_v(opblocks[i + 1],
											"b.%d.boolchecktrue", i);
			b_boolistrue = l_bb_before_v(opblocks[i + 1],
										 "b.%d.boolistrue", i);
			b_boolisanynull = l_bb_before_v(opblocks[i + 1],
											"b.%d.boolisanynull", i);
			b_boolcont = l_bb_before_v(opblocks[i + 1],
									   "b.%d.boolcont", i);

			v_boolanynullp = l_ptr_const(op->d.boolexpr.anynull,
										 l_ptr(TypeStorageBool));

			v_boolnull = LLVMBuildLoad(b, v_resnullp, "");
			v_boolvalue = LLVMBuildLoad(b, v_resvaluep, "");

			/* set resnull to boolnull */
			LLVMBuildStore(b, v_boolnull, v_resnullp);
			/* set revalue to boolvalue */
			LLVMBuildStore(b, v_boolvalue, v_resvaluep);

			LLVMBuildCondBr(b,
							LLVMBuildICmp(b, LLVMIntEQ, v_boolnull,
										  l_sbool_const(1), ""),
							b_boolisnull,
							b_boolchecktrue);

			/* build block that sets anynull */
			LLVMPositionBuilderAtEnd(b, b_boolisnull);
			/* set boolanynull to true */
			LLVMBuildStore(b, l_sbool_const(1), v_boolanynullp);
			/* and jump to next block */
			LLVMBuildBr(b, b_boolcont);

			/* build block checking for true */
			LLVMPositionBuilderAtEnd(b, b_boolchecktrue);
			LLVMBuildCondBr(b,
							LLVMBuildICmp(b, LLVMIntEQ, v_boolvalue,
										  l_sizet_const(1), ""),
							b_boolistrue,
							b_boolcont);

			/*
					 * Build block handling True. Value is true, so short
					 * circuit.
					 */
			LLVMPositionBuilderAtEnd(b, b_boolistrue);
			/* result is already set to TRUE, need not change it */
			/* and jump to the end of the OR expression */
			LLVMBuildBr(b, opblocks[op->d.boolexpr.jumpdone]);

			/* build block that continues if bool is FALSE */
			LLVMPositionBuilderAtEnd(b, b_boolcont);

			v_boolanynull = LLVMBuildLoad(b, v_boolanynullp, "");

			/* set value to NULL if any previous values were NULL */
			LLVMBuildCondBr(b,
							LLVMBuildICmp(b, LLVMIntEQ, v_boolanynull,
										  l_sbool_const(0), ""),
							opblocks[i + 1], b_boolisanynull);

			LLVMPositionBuilderAtEnd(b, b_boolisanynull);
			/* set resnull to true */
			LLVMBuildStore(b, l_sbool_const(1), v_resnullp);
			/* reset resvalue */
			LLVMBuildStore(b, l_sizet_const(0), v_resvaluep);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_BOOL_NOT_STEP:
		{
			LLVMValueRef v_boolvalue;
			LLVMValueRef v_boolnull;
			LLVMValueRef v_negbool;

			v_boolnull = LLVMBuildLoad(b, v_resnullp, "");
			v_boolvalue = LLVMBuildLoad(b, v_resvaluep, "");

			v_negbool = LLVMBuildZExt(b,
									  LLVMBuildICmp(b, LLVMIntEQ,
													v_boolvalue,
													l_sizet_const(0),
													""),
									  TypeSizeT, "");
			/* set resnull to boolnull */
			LLVMBuildStore(b, v_boolnull, v_resnullp);
			/* set revalue to !boolvalue */
			LLVMBuildStore(b, v_negbool, v_resvaluep);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_QUAL:
		{
			LLVMValueRef v_resnull;
			LLVMValueRef v_resvalue;
			LLVMValueRef v_nullorfalse;
			LLVMBasicBlockRef b_qualfail;

			b_qualfail = l_bb_before_v(opblocks[i + 1],
									   "op.%d.qualfail", i);

			v_resvalue = LLVMBuildLoad(b, v_resvaluep, "");
			v_resnull = LLVMBuildLoad(b, v_resnullp, "");

			v_nullorfalse =
				LLVMBuildOr(b,
							LLVMBuildICmp(b, LLVMIntEQ, v_resnull,
										  l_sbool_const(1), ""),
							LLVMBuildICmp(b, LLVMIntEQ, v_resvalue,
										  l_sizet_const(0), ""),
							"");

			LLVMBuildCondBr(b,
							v_nullorfalse,
							b_qualfail,
							opblocks[i + 1]);

			/* build block handling NULL or false */
			LLVMPositionBuilderAtEnd(b, b_qualfail);
			/* set resnull to false */
			LLVMBuildStore(b, l_sbool_const(0), v_resnullp);
			/* set resvalue to false */
			LLVMBuildStore(b, l_sizet_const(0), v_resvaluep);
			/* and jump out */
			LLVMBuildBr(b, opblocks[op->d.qualexpr.jumpdone]);
			break;
		}

		case EEOP_JUMP:
		{
			LLVMBuildBr(b, opblocks[op->d.jump.jumpdone]);
			break;
		}

		case EEOP_JUMP_IF_NULL:
		{
			LLVMValueRef v_resnull;

			/* Transfer control if current result is null */

			v_resnull = LLVMBuildLoad(b, v_resnullp, "");

			LLVMBuildCondBr(b,
							LLVMBuildICmp(b, LLVMIntEQ, v_resnull,
										  l_sbool_const(1), ""),
							opblocks[op->d.jump.jumpdone],
							opblocks[i + 1]);
			break;
		}

		case EEOP_JUMP_IF_NOT_NULL:
		{
			LLVMValueRef v_resnull;

			/* Transfer control if current result is non-null */

			v_resnull = LLVMBuildLoad(b, v_resnullp, "");

			LLVMBuildCondBr(b,
							LLVMBuildICmp(b, LLVMIntEQ, v_resnull,
										  l_sbool_const(0), ""),
							opblocks[op->d.jump.jumpdone],
							opblocks[i + 1]);
			break;
		}

		case EEOP_JUMP_IF_NOT_TRUE:
		{
			LLVMValueRef v_resnull;
			LLVMValueRef v_resvalue;
			LLVMValueRef v_nullorfalse;

			/* Transfer control if current result is null or false */

			v_resvalue = LLVMBuildLoad(b, v_resvaluep, "");
			v_resnull = LLVMBuildLoad(b, v_resnullp, "");

			v_nullorfalse =
				LLVMBuildOr(b,
							LLVMBuildICmp(b, LLVMIntEQ, v_resnull,
										  l_sbool_const(1), ""),
							LLVMBuildICmp(b, LLVMIntEQ, v_resvalue,
										  l_sizet_const(0), ""),
							"");

			LLVMBuildCondBr(b,
							v_nullorfalse,
							opblocks[op->d.jump.jumpdone],
							opblocks[i + 1]);
			break;
		}

		case EEOP_NULLTEST_ISNULL:
		{
			LLVMValueRef v_resnull = LLVMBuildLoad(b, v_resnullp, "");
			LLVMValueRef v_resvalue;

			v_resvalue =
				LLVMBuildSelect(b,
								LLVMBuildICmp(b, LLVMIntEQ, v_resnull,
											  l_sbool_const(1), ""),
								l_sizet_const(1),
								l_sizet_const(0),
								"");
			LLVMBuildStore(b, v_resvalue, v_resvaluep);
			LLVMBuildStore(b, l_sbool_const(0), v_resnullp);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_NULLTEST_ISNOTNULL:
		{
			LLVMValueRef v_resnull = LLVMBuildLoad(b, v_resnullp, "");
			LLVMValueRef v_resvalue;

			v_resvalue =
				LLVMBuildSelect(b,
								LLVMBuildICmp(b, LLVMIntEQ, v_resnull,
											  l_sbool_const(1), ""),
								l_sizet_const(0),
								l_sizet_const(1),
								"");
			LLVMBuildStore(b, v_resvalue, v_resvaluep);
			LLVMBuildStore(b, l_sbool_const(0), v_resnullp);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_NULLTEST_ROWISNULL:
			build_EvalXFunc(b, mod, "ExecEvalRowNull",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_NULLTEST_ROWISNOTNULL:
			build_EvalXFunc(b, mod, "ExecEvalRowNotNull",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_BOOLTEST_IS_TRUE:
		case EEOP_BOOLTEST_IS_NOT_FALSE:
		case EEOP_BOOLTEST_IS_FALSE:
		case EEOP_BOOLTEST_IS_NOT_TRUE:
		{
			LLVMBasicBlockRef b_isnull,
				b_notnull;
			LLVMValueRef v_resnull = LLVMBuildLoad(b, v_resnullp, "");

			b_isnull = l_bb_before_v(opblocks[i + 1],
									 "op.%d.isnull", i);
			b_notnull = l_bb_before_v(opblocks[i + 1],
									  "op.%d.isnotnull", i);

			/* check if value is NULL */
			LLVMBuildCondBr(b,
							LLVMBuildICmp(b, LLVMIntEQ, v_resnull,
										  l_sbool_const(1), ""),
							b_isnull, b_notnull);

			/* if value is NULL, return false */
			LLVMPositionBuilderAtEnd(b, b_isnull);

			/* result is not null */
			LLVMBuildStore(b, l_sbool_const(0), v_resnullp);

			if (opcode == EEOP_BOOLTEST_IS_TRUE ||
				opcode == EEOP_BOOLTEST_IS_FALSE)
			{
				LLVMBuildStore(b, l_sizet_const(0), v_resvaluep);
			}
			else
			{
				LLVMBuildStore(b, l_sizet_const(1), v_resvaluep);
			}

			LLVMBuildBr(b, opblocks[i + 1]);

			LLVMPositionBuilderAtEnd(b, b_notnull);

			if (opcode == EEOP_BOOLTEST_IS_TRUE ||
				opcode == EEOP_BOOLTEST_IS_NOT_FALSE)
			{
				/*
						 * if value is not null NULL, return value (already
						 * set)
						 */
			}
			else
			{
				LLVMValueRef v_value =
					LLVMBuildLoad(b, v_resvaluep, "");

				v_value = LLVMBuildZExt(b,
										LLVMBuildICmp(b, LLVMIntEQ,
													  v_value,
													  l_sizet_const(0),
													  ""),
										TypeSizeT, "");
				LLVMBuildStore(b, v_value, v_resvaluep);
			}
			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_PARAM_EXEC:
			build_EvalXFunc(b, mod, "ExecEvalParamExec",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_PARAM_EXTERN:
			if (op->d.param.lambda)
			{
				build_EvalXFunc(b, mod, "ExecEvalFastParamExtern",
								v_state, v_econtext, op);
			}
			else
			{
				build_EvalXFunc(b, mod, "ExecEvalParamExtern",
								v_state, v_econtext, op);
			}

			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_PARAM_CALLBACK:
		{
			LLVMTypeRef param_types[3];
			LLVMValueRef v_params[3];
			LLVMTypeRef v_functype;
			LLVMValueRef v_func;

			param_types[0] = l_ptr(StructExprState);
			param_types[1] = l_ptr(TypeSizeT);
			param_types[2] = l_ptr(StructExprContext);

			v_functype = LLVMFunctionType(LLVMVoidType(),
										  param_types,
										  lengthof(param_types),
										  false);
			v_func = l_ptr_const(op->d.cparam.paramfunc,
								 l_ptr(v_functype));

			v_params[0] = v_state;
			v_params[1] = l_ptr_const(op, l_ptr(TypeSizeT));
			v_params[2] = v_econtext;
			LLVMBuildCall(b,
						  v_func,
						  v_params, lengthof(v_params), "");

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_ARRAYREF_OLD:
			build_EvalXFunc(b, mod, "ExecEvalArrayRefOld",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_ARRAYREF_ASSIGN:
			build_EvalXFunc(b, mod, "ExecEvalArrayRefAssign",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_ARRAYREF_FETCH:
			build_EvalXFunc(b, mod, "ExecEvalArrayRefFetch",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_CASE_TESTVAL:
		{
			LLVMBasicBlockRef b_avail,
				b_notavail;
			LLVMValueRef v_casevaluep,
				v_casevalue;
			LLVMValueRef v_casenullp,
				v_casenull;
			LLVMValueRef v_casevaluenull;

			b_avail = l_bb_before_v(opblocks[i + 1],
									"op.%d.avail", i);
			b_notavail = l_bb_before_v(opblocks[i + 1],
									   "op.%d.notavail", i);

			v_casevaluep = l_ptr_const(op->d.casetest.value,
									   l_ptr(TypeSizeT));
			v_casenullp = l_ptr_const(op->d.casetest.isnull,
									  l_ptr(TypeStorageBool));

			v_casevaluenull =
				LLVMBuildICmp(b, LLVMIntEQ,
							  LLVMBuildPtrToInt(b, v_casevaluep,
												TypeSizeT, ""),
							  l_sizet_const(0), "");
			LLVMBuildCondBr(b, v_casevaluenull, b_notavail, b_avail);

			/* if casetest != NULL */
			LLVMPositionBuilderAtEnd(b, b_avail);
			v_casevalue = LLVMBuildLoad(b, v_casevaluep, "");
			v_casenull = LLVMBuildLoad(b, v_casenullp, "");
			LLVMBuildStore(b, v_casevalue, v_resvaluep);
			LLVMBuildStore(b, v_casenull, v_resnullp);
			LLVMBuildBr(b, opblocks[i + 1]);

			/* if casetest == NULL */
			LLVMPositionBuilderAtEnd(b, b_notavail);
			v_casevalue =
				l_load_struct_gep(b, v_econtext,
								  FIELDNO_EXPRCONTEXT_CASEDATUM, "");
			v_casenull =
				l_load_struct_gep(b, v_econtext,
								  FIELDNO_EXPRCONTEXT_CASENULL, "");
			LLVMBuildStore(b, v_casevalue, v_resvaluep);
			LLVMBuildStore(b, v_casenull, v_resnullp);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_MAKE_READONLY:
		{
			LLVMBasicBlockRef b_notnull;
			LLVMValueRef v_params[1];
			LLVMValueRef v_ret;
			LLVMValueRef v_nullp;
			LLVMValueRef v_valuep;
			LLVMValueRef v_null;
			LLVMValueRef v_value;

			b_notnull = l_bb_before_v(opblocks[i + 1],
									  "op.%d.readonly.notnull", i);

			v_nullp = l_ptr_const(op->d.make_readonly.isnull,
								  l_ptr(TypeStorageBool));

			v_null = LLVMBuildLoad(b, v_nullp, "");

			/* store null isnull value in result */
			LLVMBuildStore(b, v_null, v_resnullp);

			/* check if value is NULL */
			LLVMBuildCondBr(b,
							LLVMBuildICmp(b, LLVMIntEQ, v_null,
										  l_sbool_const(1), ""),
							opblocks[i + 1], b_notnull);

			/* if value is not null, convert to RO datum */
			LLVMPositionBuilderAtEnd(b, b_notnull);

			v_valuep = l_ptr_const(op->d.make_readonly.value,
								   l_ptr(TypeSizeT));

			v_value = LLVMBuildLoad(b, v_valuep, "");

			v_params[0] = v_value;
			v_ret =
				LLVMBuildCall(b,
							  llvm_get_decl(mod, FuncMakeExpandedObjectReadOnlyInternal),
							  v_params, lengthof(v_params), "");
			LLVMBuildStore(b, v_ret, v_resvaluep);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_IOCOERCE:
		{
			FunctionCallInfo fcinfo_out,
				fcinfo_in;
			LLVMValueRef v_fcinfo_out,
				v_fcinfo_in;
			LLVMValueRef v_fn_addr_out,
				v_fn_addr_in;
			LLVMValueRef v_fcinfo_in_isnullp;
			LLVMValueRef v_in_argp,
				v_out_argp;
			LLVMValueRef v_in_argnullp,
				v_out_argnullp;
			LLVMValueRef v_retval;
			LLVMValueRef v_resvalue;
			LLVMValueRef v_resnull;

			LLVMValueRef v_output_skip;
			LLVMValueRef v_output;

			LLVMBasicBlockRef b_skipoutput;
			LLVMBasicBlockRef b_calloutput;
			LLVMBasicBlockRef b_input;
			LLVMBasicBlockRef b_inputcall;

			fcinfo_out = op->d.iocoerce.fcinfo_data_out;
			fcinfo_in = op->d.iocoerce.fcinfo_data_in;

			b_skipoutput = l_bb_before_v(opblocks[i + 1],
										 "op.%d.skipoutputnull", i);
			b_calloutput = l_bb_before_v(opblocks[i + 1],
										 "op.%d.calloutput", i);
			b_input = l_bb_before_v(opblocks[i + 1],
									"op.%d.input", i);
			b_inputcall = l_bb_before_v(opblocks[i + 1],
										"op.%d.inputcall", i);

			v_fcinfo_out = l_ptr_const(fcinfo_out, l_ptr(StructFunctionCallInfoData));
			v_fcinfo_in = l_ptr_const(fcinfo_in, l_ptr(StructFunctionCallInfoData));
			v_fn_addr_out = l_ptr_const(fcinfo_out->flinfo->fn_addr, TypePGFunction);
			v_fn_addr_in = l_ptr_const(fcinfo_in->flinfo->fn_addr, TypePGFunction);

			v_fcinfo_in_isnullp =
				LLVMBuildStructGEP(b, v_fcinfo_in,
								   FIELDNO_FUNCTIONCALLINFODATA_ISNULL,
								   "v_fcinfo_in_isnull");
			v_out_argnullp =
				LLVMBuildStructGEP(b, v_fcinfo_out,
								   FIELDNO_FUNCTIONCALLINFODATA_ARGNULL,
								   "v_fcinfo_out_argnullp");
			v_in_argnullp =
				LLVMBuildStructGEP(b, v_fcinfo_in,
								   FIELDNO_FUNCTIONCALLINFODATA_ARGNULL,
								   "v_fcinfo_in_argnullp");
			v_out_argp =
				LLVMBuildStructGEP(b, v_fcinfo_out,
								   FIELDNO_FUNCTIONCALLINFODATA_ARG,
								   "v_fcinfo_out_argp");
			v_in_argp =
				LLVMBuildStructGEP(b, v_fcinfo_in,
								   FIELDNO_FUNCTIONCALLINFODATA_ARG,
								   "v_fcinfo_in_argp");

			/* output functions are not called on nulls */
			v_resnull = LLVMBuildLoad(b, v_resnullp, "");
			LLVMBuildCondBr(b,
							LLVMBuildICmp(b, LLVMIntEQ, v_resnull,
										  l_sbool_const(1), ""),
							b_skipoutput,
							b_calloutput);

			LLVMPositionBuilderAtEnd(b, b_skipoutput);
			v_output_skip = l_sizet_const(0);
			LLVMBuildBr(b, b_input);

			LLVMPositionBuilderAtEnd(b, b_calloutput);
			v_resvalue = LLVMBuildLoad(b, v_resvaluep, "");

			/* set arg[0] */
			LLVMBuildStore(b,
						   v_resvalue,
						   LLVMBuildStructGEP(b, v_out_argp, 0, ""));
			LLVMBuildStore(b,
						   l_sbool_const(0),
						   LLVMBuildStructGEP(b, v_out_argnullp,
											  0, ""));
			/* and call output function (can never return NULL) */
			v_output = LLVMBuildCall(b, v_fn_addr_out, &v_fcinfo_out,
									 1, "funccall_coerce_out");
			LLVMBuildBr(b, b_input);

			/* build block handling input function call */
			LLVMPositionBuilderAtEnd(b, b_input);

			/* phi between resnull and output function call branches */
			{
				LLVMValueRef incoming_values[2];
				LLVMBasicBlockRef incoming_blocks[2];

				incoming_values[0] = v_output_skip;
				incoming_blocks[0] = b_skipoutput;

				incoming_values[1] = v_output;
				incoming_blocks[1] = b_calloutput;

				v_output = LLVMBuildPhi(b, TypeSizeT, "output");
				LLVMAddIncoming(v_output,
								incoming_values, incoming_blocks,
								lengthof(incoming_blocks));
			}

			/*
					 * If input function is strict, skip if input string is
					 * NULL.
					 */
			if (op->d.iocoerce.finfo_in->fn_strict)
			{
				LLVMBuildCondBr(b,
								LLVMBuildICmp(b, LLVMIntEQ, v_output,
											  l_sizet_const(0), ""),
								opblocks[i + 1],
								b_inputcall);
			}
			else
			{
				LLVMBuildBr(b, b_inputcall);
			}

			LLVMPositionBuilderAtEnd(b, b_inputcall);
			/* set arguments */
			/* arg0: output */
			LLVMBuildStore(b, v_output,
						   LLVMBuildStructGEP(b, v_in_argp, 0, ""));
			LLVMBuildStore(b, v_resnull,
						   LLVMBuildStructGEP(b, v_in_argnullp, 0, ""));

			/* arg1: ioparam: preset in execExpr.c */
			/* arg2: typmod: preset in execExpr.c  */

			/* reset fcinfo_in->isnull */
			LLVMBuildStore(b, l_sbool_const(0), v_fcinfo_in_isnullp);
			/* and call function */
			v_retval = LLVMBuildCall(b, v_fn_addr_in, &v_fcinfo_in, 1,
									 "funccall_iocoerce_in");

			LLVMBuildStore(b, v_retval, v_resvaluep);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_DISTINCT:
		case EEOP_NOT_DISTINCT:
		{
			FunctionCallInfo fcinfo = op->d.func.fcinfo_data;

			LLVMValueRef v_fcinfo;
			LLVMValueRef v_fcinfo_isnull;

			LLVMValueRef v_argnullp;
			LLVMValueRef v_argnull0,
				v_argisnull0;
			LLVMValueRef v_argnull1,
				v_argisnull1;

			LLVMValueRef v_anyargisnull;
			LLVMValueRef v_bothargisnull;

			LLVMValueRef v_result;

			LLVMBasicBlockRef b_noargnull;
			LLVMBasicBlockRef b_checkbothargnull;
			LLVMBasicBlockRef b_bothargnull;
			LLVMBasicBlockRef b_anyargnull;

			b_noargnull = l_bb_before_v(opblocks[i + 1], "op.%d.noargnull", i);
			b_checkbothargnull = l_bb_before_v(opblocks[i + 1], "op.%d.checkbothargnull", i);
			b_bothargnull = l_bb_before_v(opblocks[i + 1], "op.%d.bothargnull", i);
			b_anyargnull = l_bb_before_v(opblocks[i + 1], "op.%d.anyargnull", i);

			v_fcinfo = l_ptr_const(fcinfo, l_ptr(StructFunctionCallInfoData));

			v_argnullp =
				LLVMBuildStructGEP(b,
								   v_fcinfo,
								   FIELDNO_FUNCTIONCALLINFODATA_ARGNULL,
								   "v_argnullp");

			/* load argnull[0|1] for both arguments */
			v_argnull0 = l_load_struct_gep(b, v_argnullp, 0, "");
			v_argisnull0 = LLVMBuildICmp(b, LLVMIntEQ, v_argnull0,
										 l_sbool_const(1), "");

			v_argnull1 = l_load_struct_gep(b, v_argnullp, 1, "");
			v_argisnull1 = LLVMBuildICmp(b, LLVMIntEQ, v_argnull1,
										 l_sbool_const(1), "");

			v_anyargisnull = LLVMBuildOr(b, v_argisnull0, v_argisnull1, "");
			v_bothargisnull = LLVMBuildAnd(b, v_argisnull0, v_argisnull1, "");

			/*
					 * Check function arguments for NULLness: If either is
					 * NULL, we check if both args are NULL. Otherwise call
					 * comparator.
					 */
			LLVMBuildCondBr(b, v_anyargisnull, b_checkbothargnull,
							b_noargnull);

			/*
					 * build block checking if any arg is null
					 */
			LLVMPositionBuilderAtEnd(b, b_checkbothargnull);
			LLVMBuildCondBr(b, v_bothargisnull, b_bothargnull,
							b_anyargnull);

			/* Both NULL? Then is not distinct... */
			LLVMPositionBuilderAtEnd(b, b_bothargnull);
			LLVMBuildStore(b, l_sbool_const(0), v_resnullp);
			if (opcode == EEOP_NOT_DISTINCT)
				LLVMBuildStore(b, l_sizet_const(1), v_resvaluep);
			else
				LLVMBuildStore(b, l_sizet_const(0), v_resvaluep);

			LLVMBuildBr(b, opblocks[i + 1]);

			/* Only one is NULL? Then is distinct... */
			LLVMPositionBuilderAtEnd(b, b_anyargnull);
			LLVMBuildStore(b, l_sbool_const(0), v_resnullp);
			if (opcode == EEOP_NOT_DISTINCT)
				LLVMBuildStore(b, l_sizet_const(0), v_resvaluep);
			else
				LLVMBuildStore(b, l_sizet_const(1), v_resvaluep);
			LLVMBuildBr(b, opblocks[i + 1]);

			/* neither argument is null: compare */
			LLVMPositionBuilderAtEnd(b, b_noargnull);

			v_result = BuildV1Call(context, b, mod, fcinfo,
								   &v_fcinfo_isnull);

			if (opcode == EEOP_DISTINCT)
			{
				/* Must invert result of "=" */
				v_result =
					LLVMBuildZExt(b,
								  LLVMBuildICmp(b, LLVMIntEQ,
												v_result,
												l_sizet_const(0), ""),
								  TypeSizeT, "");
			}

			LLVMBuildStore(b, v_fcinfo_isnull, v_resnullp);
			LLVMBuildStore(b, v_result, v_resvaluep);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_NULLIF:
		{
			FunctionCallInfo fcinfo = op->d.func.fcinfo_data;

			LLVMValueRef v_fcinfo;
			LLVMValueRef v_fcinfo_isnull;
			LLVMValueRef v_argnullp;
			LLVMValueRef v_argnull0;
			LLVMValueRef v_argnull1;
			LLVMValueRef v_anyargisnull;
			LLVMValueRef v_argp;
			LLVMValueRef v_arg0;
			LLVMBasicBlockRef b_hasnull;
			LLVMBasicBlockRef b_nonull;
			LLVMBasicBlockRef b_argsequal;
			LLVMValueRef v_retval;
			LLVMValueRef v_argsequal;

			b_hasnull = l_bb_before_v(opblocks[i + 1],
									  "b.%d.null-args", i);
			b_nonull = l_bb_before_v(opblocks[i + 1],
									 "b.%d.no-null-args", i);
			b_argsequal = l_bb_before_v(opblocks[i + 1],
										"b.%d.argsequal", i);

			v_fcinfo = l_ptr_const(fcinfo, l_ptr(StructFunctionCallInfoData));

			v_argnullp =
				LLVMBuildStructGEP(b,
								   v_fcinfo,
								   FIELDNO_FUNCTIONCALLINFODATA_ARGNULL,
								   "v_argnullp");

			v_argp =
				LLVMBuildStructGEP(b,
								   v_fcinfo,
								   FIELDNO_FUNCTIONCALLINFODATA_ARG,
								   "v_argp");

			/* if either argument is NULL they can't be equal */
			v_argnull0 = l_load_struct_gep(b, v_argnullp, 0, "");
			v_argnull1 = l_load_struct_gep(b, v_argnullp, 1, "");

			v_anyargisnull =
				LLVMBuildOr(b,
							LLVMBuildICmp(b, LLVMIntEQ, v_argnull0,
										  l_sbool_const(1), ""),
							LLVMBuildICmp(b, LLVMIntEQ, v_argnull1,
										  l_sbool_const(1), ""),
							"");

			LLVMBuildCondBr(b, v_anyargisnull, b_hasnull, b_nonull);

			/* one (or both) of the arguments are null, return arg[0] */
			LLVMPositionBuilderAtEnd(b, b_hasnull);
			v_arg0 = l_load_struct_gep(b, v_argp, 0, "");
			LLVMBuildStore(b, v_argnull0, v_resnullp);
			LLVMBuildStore(b, v_arg0, v_resvaluep);
			LLVMBuildBr(b, opblocks[i + 1]);

			/* build block to invoke function and check result */
			LLVMPositionBuilderAtEnd(b, b_nonull);

			v_retval = BuildV1Call(context, b, mod, fcinfo, &v_fcinfo_isnull);

			/*
					 * If result not null, and arguments are equal return null
					 * (same result as if there'd been NULLs, hence reuse
					 * b_hasnull).
					 */
			v_argsequal = LLVMBuildAnd(b,
									   LLVMBuildICmp(b, LLVMIntEQ,
													 v_fcinfo_isnull,
													 l_sbool_const(0),
													 ""),
									   LLVMBuildICmp(b, LLVMIntEQ,
													 v_retval,
													 l_sizet_const(1),
													 ""),
									   "");
			LLVMBuildCondBr(b, v_argsequal, b_argsequal, b_hasnull);

			/* build block setting result to NULL, if args are equal */
			LLVMPositionBuilderAtEnd(b, b_argsequal);
			LLVMBuildStore(b, l_sbool_const(1), v_resnullp);
			LLVMBuildStore(b, l_sizet_const(0), v_resvaluep);
			LLVMBuildStore(b, v_retval, v_resvaluep);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_SQLVALUEFUNCTION:
			build_EvalXFunc(b, mod, "ExecEvalSQLValueFunction",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_CURRENTOFEXPR:
			build_EvalXFunc(b, mod, "ExecEvalCurrentOfExpr",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_NEXTVALUEEXPR:
			build_EvalXFunc(b, mod, "ExecEvalNextValueExpr",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_ARRAYEXPR:
			build_EvalXFunc(b, mod, "ExecEvalArrayExpr",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_ARRAYCOERCE:
			build_EvalXFunc(b, mod, "ExecEvalArrayCoerce",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_ROW:
			build_EvalXFunc(b, mod, "ExecEvalRow",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_ROWCOMPARE_STEP:
		{
			FunctionCallInfo fcinfo = op->d.rowcompare_step.fcinfo_data;
			LLVMValueRef v_fcinfo_isnull;
			LLVMBasicBlockRef b_null;
			LLVMBasicBlockRef b_compare;
			LLVMBasicBlockRef b_compare_result;

			LLVMValueRef v_retval;

			b_null = l_bb_before_v(opblocks[i + 1],
								   "op.%d.row-null", i);
			b_compare = l_bb_before_v(opblocks[i + 1],
									  "op.%d.row-compare", i);
			b_compare_result =
				l_bb_before_v(opblocks[i + 1],
							  "op.%d.row-compare-result",
							  i);

			/*
					 * If function is strict, and either arg is null, we're
					 * done.
					 */
			if (op->d.rowcompare_step.finfo->fn_strict)
			{
				LLVMValueRef v_fcinfo;
				LLVMValueRef v_argnullp;
				LLVMValueRef v_argnull0;
				LLVMValueRef v_argnull1;
				LLVMValueRef v_anyargisnull;

				v_fcinfo = l_ptr_const(fcinfo,
									   l_ptr(StructFunctionCallInfoData));

				v_argnullp =
					LLVMBuildStructGEP(b, v_fcinfo,
									   FIELDNO_FUNCTIONCALLINFODATA_ARGNULL,
									   "v_argnullp");

				v_argnull0 = l_load_struct_gep(b, v_argnullp, 0, "");
				v_argnull1 = l_load_struct_gep(b, v_argnullp, 1, "");

				v_anyargisnull =
					LLVMBuildOr(b,
								LLVMBuildICmp(b,
											  LLVMIntEQ,
											  v_argnull0,
											  l_sbool_const(1),
											  ""),
								LLVMBuildICmp(b, LLVMIntEQ,
											  v_argnull1,
											  l_sbool_const(1), ""),
								"");

				LLVMBuildCondBr(b, v_anyargisnull, b_null, b_compare);
			}
			else
			{
				LLVMBuildBr(b, b_compare);
			}

			/* build block invoking comparison function */
			LLVMPositionBuilderAtEnd(b, b_compare);

			/* call function */
			v_retval = BuildV1Call(context, b, mod, fcinfo,
								   &v_fcinfo_isnull);
			LLVMBuildStore(b, v_retval, v_resvaluep);

			/* if result of function is NULL, force NULL result */
			LLVMBuildCondBr(b,
							LLVMBuildICmp(b,
										  LLVMIntEQ,
										  v_fcinfo_isnull,
										  l_sbool_const(0),
										  ""),
							b_compare_result,
							b_null);

			/* build block analyzing the !NULL comparator result */
			LLVMPositionBuilderAtEnd(b, b_compare_result);

			/* if results equal, compare next, otherwise done */
			LLVMBuildCondBr(b,
							LLVMBuildICmp(b,
										  LLVMIntEQ,
										  v_retval,
										  l_sizet_const(0), ""),
							opblocks[i + 1],
							opblocks[op->d.rowcompare_step.jumpdone]);

			/*
					 * Build block handling NULL input or NULL comparator
					 * result.
					 */
			LLVMPositionBuilderAtEnd(b, b_null);
			LLVMBuildStore(b, l_sbool_const(1), v_resnullp);
			LLVMBuildBr(b, opblocks[op->d.rowcompare_step.jumpnull]);

			break;
		}

		case EEOP_ROWCOMPARE_FINAL:
		{
			RowCompareType rctype = op->d.rowcompare_final.rctype;

			LLVMValueRef v_cmpresult;
			LLVMValueRef v_result;
			LLVMIntPredicate predicate;

			/*
					 * Btree comparators return 32 bit results, need to be
					 * careful about sign (used as a 64 bit value it's
					 * otherwise wrong).
					 */
			v_cmpresult =
				LLVMBuildTrunc(b,
							   LLVMBuildLoad(b, v_resvaluep, ""),
							   LLVMInt32Type(), "");

			switch (rctype)
			{
			case ROWCOMPARE_LT:
				predicate = LLVMIntSLT;
				break;
			case ROWCOMPARE_LE:
				predicate = LLVMIntSLE;
				break;
			case ROWCOMPARE_GT:
				predicate = LLVMIntSGT;
				break;
			case ROWCOMPARE_GE:
				predicate = LLVMIntSGE;
				break;
			default:
				/* EQ and NE cases aren't allowed here */
				Assert(false);
				predicate = 0; /* prevent compiler warning */
				break;
			}

			v_result = LLVMBuildICmp(b,
									 predicate,
									 v_cmpresult,
									 l_int32_const(0),
									 "");
			v_result = LLVMBuildZExt(b, v_result, TypeSizeT, "");

			LLVMBuildStore(b, l_sbool_const(0), v_resnullp);
			LLVMBuildStore(b, v_result, v_resvaluep);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_MINMAX:
			build_EvalXFunc(b, mod, "ExecEvalMinMax",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_FIELDSELECT:
			if (!op->d.fieldselect.lambda)
			{
				build_EvalXFunc(b, mod, "ExecEvalFieldSelect",
								v_state, v_econtext, op);
			}
			else
			{
				build_EvalXFunc(b, mod, "ExecEvalFastFieldSelect",
								v_state, v_econtext, op);
			}

			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_FIELDSTORE_DEFORM:
			build_EvalXFunc(b, mod, "ExecEvalFieldStoreDeForm",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_FIELDSTORE_FORM:
			build_EvalXFunc(b, mod, "ExecEvalFieldStoreForm",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_ARRAYREF_SUBSCRIPT:
		{
			LLVMValueRef v_fn;
			int jumpdone = op->d.arrayref_subscript.jumpdone;
			LLVMValueRef v_params[2];
			LLVMValueRef v_ret;

			v_fn = llvm_get_decl(mod, FuncExecEvalArrayRefSubscript);

			v_params[0] = v_state;
			v_params[1] = l_ptr_const(op, l_ptr(StructExprEvalStep));
			v_ret = LLVMBuildCall(b, v_fn,
								  v_params, lengthof(v_params), "");
			v_ret = LLVMBuildZExt(b, v_ret, TypeStorageBool, "");

			LLVMBuildCondBr(b,
							LLVMBuildICmp(b, LLVMIntEQ, v_ret,
										  l_sbool_const(1), ""),
							opblocks[i + 1],
							opblocks[jumpdone]);
			break;
		}

		case EEOP_DOMAIN_TESTVAL:
		{
			LLVMBasicBlockRef b_avail,
				b_notavail;
			LLVMValueRef v_casevaluep,
				v_casevalue;
			LLVMValueRef v_casenullp,
				v_casenull;
			LLVMValueRef v_casevaluenull;

			b_avail = l_bb_before_v(opblocks[i + 1],
									"op.%d.avail", i);
			b_notavail = l_bb_before_v(opblocks[i + 1],
									   "op.%d.notavail", i);

			v_casevaluep = l_ptr_const(op->d.casetest.value,
									   l_ptr(TypeSizeT));
			v_casenullp = l_ptr_const(op->d.casetest.isnull,
									  l_ptr(TypeStorageBool));

			v_casevaluenull =
				LLVMBuildICmp(b, LLVMIntEQ,
							  LLVMBuildPtrToInt(b, v_casevaluep,
												TypeSizeT, ""),
							  l_sizet_const(0), "");
			LLVMBuildCondBr(b,
							v_casevaluenull,
							b_notavail, b_avail);

			/* if casetest != NULL */
			LLVMPositionBuilderAtEnd(b, b_avail);
			v_casevalue = LLVMBuildLoad(b, v_casevaluep, "");
			v_casenull = LLVMBuildLoad(b, v_casenullp, "");
			LLVMBuildStore(b, v_casevalue, v_resvaluep);
			LLVMBuildStore(b, v_casenull, v_resnullp);
			LLVMBuildBr(b, opblocks[i + 1]);

			/* if casetest == NULL */
			LLVMPositionBuilderAtEnd(b, b_notavail);
			v_casevalue =
				l_load_struct_gep(b, v_econtext,
								  FIELDNO_EXPRCONTEXT_DOMAINDATUM,
								  "");
			v_casenull =
				l_load_struct_gep(b, v_econtext,
								  FIELDNO_EXPRCONTEXT_DOMAINNULL,
								  "");
			LLVMBuildStore(b, v_casevalue, v_resvaluep);
			LLVMBuildStore(b, v_casenull, v_resnullp);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_DOMAIN_NOTNULL:
			build_EvalXFunc(b, mod, "ExecEvalConstraintNotNull",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_DOMAIN_CHECK:
			build_EvalXFunc(b, mod, "ExecEvalConstraintCheck",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_CONVERT_ROWTYPE:
			build_EvalXFunc(b, mod, "ExecEvalConvertRowtype",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_SCALARARRAYOP:
			build_EvalXFunc(b, mod, "ExecEvalScalarArrayOp",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_XMLEXPR:
			build_EvalXFunc(b, mod, "ExecEvalXmlExpr",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_AGGREF:
		{
			AggrefExprState *aggref = op->d.aggref.astate;
			LLVMValueRef v_aggnop;
			LLVMValueRef v_aggno;
			LLVMValueRef value,
				isnull;

			/*
					 * At this point aggref->aggno is not yet set (it's set up
					 * in ExecInitAgg() after initializing the expression). So
					 * load it from memory each time round.
					 */
			v_aggnop = l_ptr_const(&aggref->aggno,
								   l_ptr(LLVMInt32Type()));
			v_aggno = LLVMBuildLoad(b, v_aggnop, "v_aggno");

			/* load agg value / null */
			value = l_load_gep1(b, v_aggvalues, v_aggno, "aggvalue");
			isnull = l_load_gep1(b, v_aggnulls, v_aggno, "aggnull");

			/* and store result */
			LLVMBuildStore(b, value, v_resvaluep);
			LLVMBuildStore(b, isnull, v_resnullp);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_GROUPING_FUNC:
			build_EvalXFunc(b, mod, "ExecEvalGroupingFunc",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_WINDOW_FUNC:
		{
			WindowFuncExprState *wfunc = op->d.window_func.wfstate;
			LLVMValueRef v_wfuncnop;
			LLVMValueRef v_wfuncno;
			LLVMValueRef value,
				isnull;

			/*
					 * At this point aggref->wfuncno is not yet set (it's set
					 * up in ExecInitWindowAgg() after initializing the
					 * expression). So load it from memory each time round.
					 */
			v_wfuncnop = l_ptr_const(&wfunc->wfuncno,
									 l_ptr(LLVMInt32Type()));
			v_wfuncno = LLVMBuildLoad(b, v_wfuncnop, "v_wfuncno");

			/* load window func value / null */
			value = l_load_gep1(b, v_aggvalues, v_wfuncno,
								"windowvalue");
			isnull = l_load_gep1(b, v_aggnulls, v_wfuncno,
								 "windownull");

			LLVMBuildStore(b, value, v_resvaluep);
			LLVMBuildStore(b, isnull, v_resnullp);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_SUBPLAN:
			build_EvalXFunc(b, mod, "ExecEvalSubPlan",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_ALTERNATIVE_SUBPLAN:
			build_EvalXFunc(b, mod, "ExecEvalAlternativeSubPlan",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_AGG_STRICT_DESERIALIZE:
		{
			FunctionCallInfo fcinfo = op->d.agg_deserialize.fcinfo_data;
			LLVMValueRef v_fcinfo;
			LLVMValueRef v_argnullp;
			LLVMValueRef v_argnull0;
			LLVMBasicBlockRef b_deserialize;

			b_deserialize = l_bb_before_v(opblocks[i + 1],
										  "op.%d.deserialize", i);

			v_fcinfo = l_ptr_const(fcinfo,
								   l_ptr(StructFunctionCallInfoData));

			v_argnullp =
				LLVMBuildStructGEP(b,
								   v_fcinfo,
								   FIELDNO_FUNCTIONCALLINFODATA_ARGNULL,
								   "v_argnullp");
			v_argnull0 =
				l_load_struct_gep(b, v_argnullp, 0, "v_argnull0");

			LLVMBuildCondBr(b,
							LLVMBuildICmp(b,
										  LLVMIntEQ,
										  v_argnull0,
										  l_sbool_const(1),
										  ""),
							opblocks[op->d.agg_deserialize.jumpnull],
							b_deserialize);
			LLVMPositionBuilderAtEnd(b, b_deserialize);
		}
			/* FALLTHROUGH */

		case EEOP_AGG_DESERIALIZE:
		{
			AggState *aggstate;
			FunctionCallInfo fcinfo;

			LLVMValueRef v_retval;
			LLVMValueRef v_fcinfo_isnull;
			LLVMValueRef v_tmpcontext;
			LLVMValueRef v_oldcontext;

			aggstate = op->d.agg_deserialize.aggstate;
			fcinfo = op->d.agg_deserialize.fcinfo_data;

			v_tmpcontext =
				l_ptr_const(aggstate->tmpcontext->ecxt_per_tuple_memory,
							l_ptr(StructMemoryContextData));
			v_oldcontext = l_mcxt_switch(mod, b, v_tmpcontext);
			v_retval = BuildV1Call(context, b, mod, fcinfo,
								   &v_fcinfo_isnull);
			l_mcxt_switch(mod, b, v_oldcontext);

			LLVMBuildStore(b, v_retval, v_resvaluep);
			LLVMBuildStore(b, v_fcinfo_isnull, v_resnullp);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_AGG_STRICT_INPUT_CHECK:
		{
			int nargs = op->d.agg_strict_input_check.nargs;
			bool *nulls = op->d.agg_strict_input_check.nulls;
			int jumpnull;
			int argno;

			LLVMValueRef v_nullp;
			LLVMBasicBlockRef *b_checknulls;

			Assert(nargs > 0);

			jumpnull = op->d.agg_strict_input_check.jumpnull;
			v_nullp = l_ptr_const(nulls, l_ptr(TypeStorageBool));

			/* create blocks for checking args */
			b_checknulls = palloc(sizeof(LLVMBasicBlockRef *) * nargs);
			for (argno = 0; argno < nargs; argno++)
			{
				b_checknulls[argno] =
					l_bb_before_v(opblocks[i + 1],
								  "op.%d.check-null.%d",
								  i, argno);
			}

			LLVMBuildBr(b, b_checknulls[0]);

			/* strict function, check for NULL args */
			for (argno = 0; argno < nargs; argno++)
			{
				LLVMValueRef v_argno = l_int32_const(argno);
				LLVMValueRef v_argisnull;
				LLVMBasicBlockRef b_argnotnull;

				LLVMPositionBuilderAtEnd(b, b_checknulls[argno]);

				if (argno + 1 == nargs)
					b_argnotnull = opblocks[i + 1];
				else
					b_argnotnull = b_checknulls[argno + 1];

				v_argisnull = l_load_gep1(b, v_nullp, v_argno, "");

				LLVMBuildCondBr(b,
								LLVMBuildICmp(b,
											  LLVMIntEQ,
											  v_argisnull,
											  l_sbool_const(1), ""),
								opblocks[jumpnull],
								b_argnotnull);
			}

			break;
		}

		case EEOP_AGG_INIT_TRANS:
		{
			AggState *aggstate;
			AggStatePerTrans pertrans;

			LLVMValueRef v_aggstatep;
			LLVMValueRef v_pertransp;

			LLVMValueRef v_allpergroupsp;

			LLVMValueRef v_pergroupp;

			LLVMValueRef v_setoff,
				v_transno;

			LLVMValueRef v_notransvalue;

			LLVMBasicBlockRef b_init;

			aggstate = op->d.agg_init_trans.aggstate;
			pertrans = op->d.agg_init_trans.pertrans;

			v_aggstatep = l_ptr_const(aggstate,
									  l_ptr(StructAggState));
			v_pertransp = l_ptr_const(pertrans,
									  l_ptr(StructAggStatePerTransData));

			/*
					 * pergroup = &aggstate->all_pergroups
					 * [op->d.agg_init_trans_check.setoff]
					 * [op->d.agg_init_trans_check.transno];
					 */
			v_allpergroupsp =
				l_load_struct_gep(b, v_aggstatep,
								  FIELDNO_AGGSTATE_ALL_PERGROUPS,
								  "aggstate.all_pergroups");
			v_setoff = l_int32_const(op->d.agg_init_trans.setoff);
			v_transno = l_int32_const(op->d.agg_init_trans.transno);
			v_pergroupp =
				LLVMBuildGEP(b,
							 l_load_gep1(b, v_allpergroupsp, v_setoff, ""),
							 &v_transno, 1, "");

			v_notransvalue =
				l_load_struct_gep(b, v_pergroupp,
								  FIELDNO_AGGSTATEPERGROUPDATA_NOTRANSVALUE,
								  "notransvalue");

			b_init = l_bb_before_v(opblocks[i + 1],
								   "op.%d.inittrans", i);

			LLVMBuildCondBr(b,
							LLVMBuildICmp(b, LLVMIntEQ, v_notransvalue,
										  l_sbool_const(1), ""),
							b_init,
							opblocks[i + 1]);

			LLVMPositionBuilderAtEnd(b, b_init);

			{
				LLVMValueRef params[3];
				LLVMValueRef v_curaggcontext;
				LLVMValueRef v_current_set;
				LLVMValueRef v_aggcontext;

				v_aggcontext = l_ptr_const(op->d.agg_init_trans.aggcontext,
										   l_ptr(StructExprContext));

				v_current_set =
					LLVMBuildStructGEP(b,
									   v_aggstatep,
									   FIELDNO_AGGSTATE_CURRENT_SET,
									   "aggstate.current_set");
				v_curaggcontext =
					LLVMBuildStructGEP(b,
									   v_aggstatep,
									   FIELDNO_AGGSTATE_CURAGGCONTEXT,
									   "aggstate.curaggcontext");

				LLVMBuildStore(b, l_int32_const(op->d.agg_init_trans.setno),
							   v_current_set);
				LLVMBuildStore(b, v_aggcontext,
							   v_curaggcontext);

				params[0] = v_aggstatep;
				params[1] = v_pertransp;
				params[2] = v_pergroupp;

				LLVMBuildCall(b,
							  llvm_get_decl(mod, FuncExecAggInitGroup),
							  params, lengthof(params),
							  "");
			}
			LLVMBuildBr(b, opblocks[op->d.agg_init_trans.jumpnull]);

			break;
		}

		case EEOP_AGG_STRICT_TRANS_CHECK:
		{
			AggState *aggstate;
			LLVMValueRef v_setoff,
				v_transno;

			LLVMValueRef v_aggstatep;
			LLVMValueRef v_allpergroupsp;

			LLVMValueRef v_transnull;
			LLVMValueRef v_pergroupp;

			int jumpnull = op->d.agg_strict_trans_check.jumpnull;

			aggstate = op->d.agg_strict_trans_check.aggstate;
			v_aggstatep = l_ptr_const(aggstate, l_ptr(StructAggState));

			/*
					 * pergroup = &aggstate->all_pergroups
					 * [op->d.agg_strict_trans_check.setoff]
					 * [op->d.agg_init_trans_check.transno];
					 */
			v_allpergroupsp =
				l_load_struct_gep(b, v_aggstatep,
								  FIELDNO_AGGSTATE_ALL_PERGROUPS,
								  "aggstate.all_pergroups");
			v_setoff =
				l_int32_const(op->d.agg_strict_trans_check.setoff);
			v_transno =
				l_int32_const(op->d.agg_strict_trans_check.transno);
			v_pergroupp =
				LLVMBuildGEP(b,
							 l_load_gep1(b, v_allpergroupsp, v_setoff, ""),
							 &v_transno, 1, "");

			v_transnull =
				l_load_struct_gep(b, v_pergroupp,
								  FIELDNO_AGGSTATEPERGROUPDATA_TRANSVALUEISNULL,
								  "transnull");

			LLVMBuildCondBr(b,
							LLVMBuildICmp(b, LLVMIntEQ, v_transnull,
										  l_sbool_const(1), ""),
							opblocks[jumpnull],
							opblocks[i + 1]);

			break;
		}

		case EEOP_AGG_PLAIN_TRANS_BYVAL:
		case EEOP_AGG_PLAIN_TRANS:
		{
			AggState *aggstate;
			AggStatePerTrans pertrans;
			FunctionCallInfo fcinfo;

			LLVMValueRef v_aggstatep;
			LLVMValueRef v_fcinfo;
			LLVMValueRef v_fcinfo_isnull;
			LLVMValueRef v_argp,
				v_argnullp;

			LLVMValueRef v_transvaluep;
			LLVMValueRef v_transnullp;

			LLVMValueRef v_setoff;
			LLVMValueRef v_transno;

			LLVMValueRef v_aggcontext;

			LLVMValueRef v_allpergroupsp;
			LLVMValueRef v_current_setp;
			LLVMValueRef v_current_pertransp;
			LLVMValueRef v_curaggcontext;

			LLVMValueRef v_pertransp;

			LLVMValueRef v_pergroupp;

			LLVMValueRef v_retval;

			LLVMValueRef v_tmpcontext;
			LLVMValueRef v_oldcontext;

			aggstate = op->d.agg_trans.aggstate;
			pertrans = op->d.agg_trans.pertrans;

			fcinfo = &pertrans->transfn_fcinfo;

			v_aggstatep = l_ptr_const(aggstate,
									  l_ptr(StructAggState));
			v_pertransp = l_ptr_const(pertrans,
									  l_ptr(StructAggStatePerTransData));

			/*
					 * pergroup = &aggstate->all_pergroups
					 * [op->d.agg_strict_trans_check.setoff]
					 * [op->d.agg_init_trans_check.transno];
					 */
			v_allpergroupsp =
				l_load_struct_gep(b, v_aggstatep,
								  FIELDNO_AGGSTATE_ALL_PERGROUPS,
								  "aggstate.all_pergroups");
			v_setoff = l_int32_const(op->d.agg_trans.setoff);
			v_transno = l_int32_const(op->d.agg_trans.transno);
			v_pergroupp =
				LLVMBuildGEP(b,
							 l_load_gep1(b, v_allpergroupsp, v_setoff, ""),
							 &v_transno, 1, "");

			v_fcinfo = l_ptr_const(fcinfo,
								   l_ptr(StructFunctionCallInfoData));

			v_argnullp =
				LLVMBuildStructGEP(b,
								   v_fcinfo,
								   FIELDNO_FUNCTIONCALLINFODATA_ARGNULL,
								   "v_argnullp");
			v_argp =
				LLVMBuildStructGEP(b,
								   v_fcinfo,
								   FIELDNO_FUNCTIONCALLINFODATA_ARG,
								   "v_argp");

			v_aggcontext = l_ptr_const(op->d.agg_trans.aggcontext,
									   l_ptr(StructExprContext));

			v_current_setp =
				LLVMBuildStructGEP(b,
								   v_aggstatep,
								   FIELDNO_AGGSTATE_CURRENT_SET,
								   "aggstate.current_set");
			v_curaggcontext =
				LLVMBuildStructGEP(b,
								   v_aggstatep,
								   FIELDNO_AGGSTATE_CURAGGCONTEXT,
								   "aggstate.curaggcontext");
			v_current_pertransp =
				LLVMBuildStructGEP(b,
								   v_aggstatep,
								   FIELDNO_AGGSTATE_CURPERTRANS,
								   "aggstate.curpertrans");

			/* set aggstate globals */
			LLVMBuildStore(b, v_aggcontext, v_curaggcontext);
			LLVMBuildStore(b, l_int32_const(op->d.agg_trans.setno),
						   v_current_setp);
			LLVMBuildStore(b, v_pertransp, v_current_pertransp);

			/* invoke transition function in per-tuple context */
			v_tmpcontext =
				l_ptr_const(aggstate->tmpcontext->ecxt_per_tuple_memory,
							l_ptr(StructMemoryContextData));
			v_oldcontext = l_mcxt_switch(mod, b, v_tmpcontext);

			/* store transvalue in fcinfo->arg/argnull[0] */
			v_transvaluep =
				LLVMBuildStructGEP(b, v_pergroupp,
								   FIELDNO_AGGSTATEPERGROUPDATA_TRANSVALUE,
								   "transvalue");
			v_transnullp =
				LLVMBuildStructGEP(b, v_pergroupp,
								   FIELDNO_AGGSTATEPERGROUPDATA_TRANSVALUEISNULL,
								   "transnullp");
			LLVMBuildStore(b,
						   LLVMBuildLoad(b, v_transvaluep,
										 "transvalue"),
						   LLVMBuildStructGEP(b, v_argp, 0, ""));
			LLVMBuildStore(b,
						   LLVMBuildLoad(b, v_transnullp, "transnull"),
						   LLVMBuildStructGEP(b, v_argnullp, 0, ""));

			/* and invoke transition function */
			v_retval = BuildV1Call(context, b, mod, fcinfo,
								   &v_fcinfo_isnull);

			/*
					 * For pass-by-ref datatype, must copy the new value into
					 * aggcontext and free the prior transValue.  But if
					 * transfn returned a pointer to its first input, we don't
					 * need to do anything.  Also, if transfn returned a
					 * pointer to a R/W expanded object that is already a
					 * child of the aggcontext, assume we can adopt that value
					 * without copying it.
					 */
			if (opcode == EEOP_AGG_PLAIN_TRANS)
			{
				LLVMBasicBlockRef b_call;
				LLVMBasicBlockRef b_nocall;
				LLVMValueRef v_fn;
				LLVMValueRef v_transvalue;
				LLVMValueRef v_transnull;
				LLVMValueRef v_newval;
				LLVMValueRef params[6];

				b_call = l_bb_before_v(opblocks[i + 1],
									   "op.%d.transcall", i);
				b_nocall = l_bb_before_v(opblocks[i + 1],
										 "op.%d.transnocall", i);

				v_transvalue = LLVMBuildLoad(b, v_transvaluep, "");
				v_transnull = LLVMBuildLoad(b, v_transnullp, "");

				/*
						 * DatumGetPointer(newVal) !=
						 * DatumGetPointer(pergroup->transValue))
						 */
				LLVMBuildCondBr(b,
								LLVMBuildICmp(b, LLVMIntEQ,
											  v_transvalue,
											  v_retval, ""),
								b_nocall, b_call);

				/* returned datum not passed datum, reparent */
				LLVMPositionBuilderAtEnd(b, b_call);

				params[0] = v_aggstatep;
				params[1] = v_pertransp;
				params[2] = v_retval;
				params[3] = LLVMBuildTrunc(b, v_fcinfo_isnull,
										   TypeParamBool, "");
				params[4] = v_transvalue;
				params[5] = LLVMBuildTrunc(b, v_transnull,
										   TypeParamBool, "");

				v_fn = llvm_get_decl(mod, FuncExecAggTransReparent);
				v_newval =
					LLVMBuildCall(b, v_fn,
								  params, lengthof(params),
								  "");

				/* store trans value */
				LLVMBuildStore(b, v_newval, v_transvaluep);
				LLVMBuildStore(b, v_fcinfo_isnull, v_transnullp);

				l_mcxt_switch(mod, b, v_oldcontext);
				LLVMBuildBr(b, opblocks[i + 1]);

				/* returned datum passed datum, no need to reparent */
				LLVMPositionBuilderAtEnd(b, b_nocall);
			}

			/* store trans value */
			LLVMBuildStore(b, v_retval, v_transvaluep);
			LLVMBuildStore(b, v_fcinfo_isnull, v_transnullp);

			l_mcxt_switch(mod, b, v_oldcontext);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_AGG_ORDERED_TRANS_DATUM:
			build_EvalXFunc(b, mod, "ExecEvalAggOrderedTransDatum",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_AGG_ORDERED_TRANS_TUPLE:
			build_EvalXFunc(b, mod, "ExecEvalAggOrderedTransTuple",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;

		case EEOP_LAST:
			Assert(false);
			break;
		}
	}

	LLVMDisposeBuilder(b);

	/*
	 * Don't immediately emit function, instead do so the first time the
	 * expression is actually evaluated. That allows to emit a lot of
	 * functions together, avoiding a lot of repeated llvm and memory
	 * remapping overhead.
	 */
	{

		CompiledExprState *cstate = palloc0(sizeof(CompiledExprState));

		cstate->context = context;
		cstate->funcname = funcname;

		state->evalfunc = ExecRunCompiledExpr;
		state->evalfunc_private = cstate;
	}

	llvm_leave_fatal_on_oom();

	INSTR_TIME_SET_CURRENT(endtime);
	INSTR_TIME_ACCUM_DIFF(context->base.instr.generation_counter,
						  endtime, starttime);

	return true;
}

/*
 * Fast JIT compilation for a small subset of opcodes.
 * 
 * To reduce overhead caused by the various error reporting and type checking
 * routines in the internal Postgres functions, this procedure performs some
 * very efficient low-level JIT-compilation for integers and doubles with
 * basic arithmetic operators and common functions such as sqrt, abs and pow.
 *
 * More complex datatypes, such as strings, arrays, numeric, etc. are not supported.
 * 
 * The generated function takes a **Datum parameter: The first indirection selects
 * the rowtype argument and the second indirection refers to an actual value of 
 * the row.
 */
bool llvm_compile_simple_expr(ExprState *state)
{
	PlanState *parent = state->parent;
	int i;
	char *funcname;

	LLVMJitContext *context = NULL;

	LLVMBuilderRef b;
	LLVMModuleRef mod;
	LLVMTypeRef eval_sig;
	LLVMValueRef eval_fn;
	LLVMValueRef datum_param;
	LLVMBasicBlockRef entry;
	LLVMBasicBlockRef *opblocks;
	LLVMValueRef registers[50];
	int registerPointer = 0;

	instr_time starttime;
	instr_time endtime;

	llvm_enter_fatal_on_oom();

	/* get or create JIT context */
	if (parent && parent->state->es_jit)
	{
		context = (LLVMJitContext *)parent->state->es_jit;
	}
	else
	{
		context = llvm_create_context(parent->state->es_jit_flags);

		if (parent)
		{
			parent->state->es_jit = &context->base;
		}
	}

	INSTR_TIME_SET_CURRENT(starttime);

	mod = llvm_mutable_module(context);

	b = LLVMCreateBuilder();

	funcname = llvm_expand_funcname(context, "evalexpr", true);

	/* Create the signature and function */
	{
		LLVMTypeRef param_types[1];

		param_types[0] = l_ptr(l_ptr(TypeDatum)); /* state */

		eval_sig = LLVMFunctionType(TypeDatum,
									param_types, lengthof(param_types),
									false);
	}

	eval_fn = LLVMAddFunction(mod, funcname, eval_sig);
	LLVMSetLinkage(eval_fn, LLVMExternalLinkage);
	LLVMSetVisibility(eval_fn, LLVMDefaultVisibility);

	entry = LLVMAppendBasicBlock(eval_fn, "entry");

	LLVMPositionBuilderAtEnd(b, entry);

	/* allocate blocks for each op upfront, so we can do jumps easily */
	opblocks = palloc(sizeof(LLVMBasicBlockRef));
	opblocks[0] = l_bb_append_v(eval_fn, "b.op.start");

	/* jump from entry to first block */
	LLVMBuildBr(b, opblocks[0]);

	datum_param = LLVMGetParam(eval_fn, 0);
	LLVMPositionBuilderAtEnd(b, opblocks[0]);

	for (i = 0; i < state->steps_len; i++)
	{
		ExprEvalStep *op;
		ExprEvalOp opcode;
		LLVMValueRef v_opval;

		op = &state->steps[i];
		opcode = ExecEvalStepOp(state, op);

		switch (opcode)
		{
		case EEOP_DONE:
		{
			LLVMBuildRet(b, LLVMBuildZExtOrBitCast(b, registers[registerPointer - 1], TypeDatum, ""));
			break;
		}

		case EEOP_CONST:
		{
			printf("Const: %lu %i %f\n", DatumGetInt64(op->d.constval.value), DatumGetInt32(op->d.constval.value), DatumGetFloat8(op->d.constval.value));
			registers[registerPointer++] = l_sizet_const(op->d.constval.value);
			break;
		}

		case EEOP_FUNCEXPR_STRICT:
		case EEOP_FUNCEXPR:
		{
			char *funcname;
			int numparams;

			LLVMValueRef opres;
			FunctionCallInfo fcinfo = op->d.func.fcinfo_data;
			printf("Op: %i\n", fcinfo->flinfo->fn_oid);
			switch (fcinfo->flinfo->fn_oid)
			{
			case 141:
				numparams = 2;
				opres = LLVMBuildBinOp(b, LLVMMul,
									   l_as_int4(b, registers[registerPointer - 2]),
									   l_as_int4(b, registers[registerPointer - 1]),
									   "imul");
				break;

			case 154:
				numparams = 2;
				opres = LLVMBuildBinOp(b, LLVMSDiv,
									   l_as_int4(b, registers[registerPointer - 2]),
									   l_as_int4(b, registers[registerPointer - 1]),
									   "idiv");
				break;

			case 177:
				numparams = 2;
				opres = LLVMBuildBinOp(b, LLVMAdd,
									   l_as_int4(b, registers[registerPointer - 2]),
									   l_as_int4(b, registers[registerPointer - 1]),
									   "iadd");
				break;

			case 181:
				numparams = 2;
				opres = LLVMBuildBinOp(b, LLVMSub,
									   l_as_int4(b, registers[registerPointer - 2]),
									   l_as_int4(b, registers[registerPointer - 1]),
									   "isub");
				break;

			case 212:
				numparams = 1;
				opres = LLVMBuildNeg(b, l_as_int4(b, registers[registerPointer - 1]),
									 "int4um");
				break;

			case 1726:
			case 216:
				numparams = 2;
				opres = LLVMBuildBinOp(b, LLVMFMul,
									   l_as_float8(b, registers[registerPointer - 2]),
									   l_as_float8(b, registers[registerPointer - 1]),
									   "fmul");
				printf("Ptr: %i\n", registerPointer);
				break;

			case 1727:
			case 217:
				numparams = 2;
				opres = LLVMBuildBinOp(b, LLVMFDiv,
									   l_as_float8(b, registers[registerPointer - 2]),
									   l_as_float8(b, registers[registerPointer - 1]),
									   "fdiv");
				break;

			case 1724:
			case 218:
				numparams = 2;
				opres = LLVMBuildBinOp(b, LLVMFAdd,
									   l_as_float8(b, registers[registerPointer - 2]),
									   l_as_float8(b, registers[registerPointer - 1]),
									   "fadd");
				break;

			case 1725:
			case 219:
				numparams = 2;
				opres = LLVMBuildBinOp(b, LLVMFSub,
									   l_as_float8(b, registers[registerPointer - 2]),
									   l_as_float8(b, registers[registerPointer - 1]),
									   "fsub");
				break;

			case 220:
				numparams = 1;
				opres = LLVMBuildNeg(b, l_as_float8(b, registers[registerPointer - 1]),
									 "float_um");
				break;

			case 316:
			case 1740:
				numparams = 1;
				opres = LLVMBuildCast(b, LLVMSIToFP, l_as_int4(b, registers[registerPointer - 1]),
									  LLVMDoubleType(), "int4_to_float8");
				break;

			case 317:
			case 1744:
				numparams = 1;
				opres = LLVMBuildCast(b, LLVMFPToSI, l_as_float8(b, registers[registerPointer - 1]),
									  LLVMInt32Type(), "float_to_int4");
				break;

			case 483:
			case 1781:
				numparams = 1;
				opres = LLVMBuildCast(b, LLVMFPToSI, l_as_float8(b, registers[registerPointer - 1]),
									  LLVMInt64Type(), "float8_to_int8");
				break;

			case 1746:
			case 1743:
				numparams = 1;
				opres = registers[registerPointer - 1];
				break;

				CASE_FLOAT8_CFUNC_1ARG(221, "fabs")
				CASE_FLOAT8_CFUNC_1ARG(1395, "fabs")
				CASE_FLOAT8_CFUNC_1ARG(230, "sqrt")
				CASE_FLOAT8_CFUNC_2ARG(232, "pow")
				CASE_FLOAT8_CFUNC_1ARG(1600, "asin")
				CASE_FLOAT8_CFUNC_1ARG(1601, "acos")
				CASE_FLOAT8_CFUNC_1ARG(1602, "atan")
				CASE_FLOAT8_CFUNC_2ARG(1603, "atan2")
				CASE_FLOAT8_CFUNC_1ARG(1604, "sin")
				CASE_FLOAT8_CFUNC_1ARG(1605, "cos")
				CASE_FLOAT8_CFUNC_1ARG(1606, "tan")
				CASE_FLOAT8_CFUNC_1ARG(1607, "cot")
				CASE_FLOAT8_CFUNC_1ARG(1339, "log10")
				CASE_FLOAT8_CFUNC_1ARG(1341, "log")
				CASE_FLOAT8_CFUNC_1ARG(1344, "sqrt")
				CASE_FLOAT8_CFUNC_2ARG(1346, "pow")
				CASE_FLOAT8_CFUNC_1ARG(1347, "exp")

			default:
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("Function with Oid %i has no fast JIT implementation.", fcinfo->flinfo->fn_oid)));
				break;
			}

			registerPointer -= numparams;
			registers[registerPointer++] = opres;

			break;
		}

		case EEOP_PARAM_EXTERN:
		{
			printf("Extern\n");
			LLVMValueRef addr = l_int32_const(op->d.param.paramid - 1);

			registers[registerPointer++] = LLVMBuildLoad(b, LLVMBuildGEP(b, datum_param, &addr, 1, ""), "");
			break;
		}

		case EEOP_FIELDSELECT:
		{
			printf("FieldSelect\n");
			LLVMValueRef addr = l_int32_const(op->d.fieldselect.fieldnum - 1);

			registers[registerPointer - 1] = LLVMBuildLoad(b, LLVMBuildGEP(b, registers[registerPointer - 1], &addr, 1, ""), "");
			break;
		}

		case EEOP_LAST:
			Assert(false);
			break;

		default:
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("Opcode %i cannot be JIT-compiled with llvm_compile_simple_expr", opcode)));
			break;
		}
	}

	LLVMDisposeBuilder(b);

	/*
	 * Don't immediately emit function, instead do so the first time the
	 * expression is actually evaluated. That allows to emit a lot of
	 * functions together, avoiding a lot of repeated llvm and memory
	 * remapping overhead.
	 */
	{
		CompiledExprState *cstate = palloc0(sizeof(CompiledExprState));

		cstate->context = context;
		cstate->funcname = funcname;

		state->evalfunc_simple_private = cstate;
	}

	llvm_leave_fatal_on_oom();

	INSTR_TIME_SET_CURRENT(endtime);
	INSTR_TIME_ACCUM_DIFF(context->base.instr.generation_counter,
						  endtime, starttime);

	return true;
}

/*
 * Run compiled expression.
 *
 * This will only be called the first time a JITed expression is called. We
 * first make sure the expression is still up2date, and then get a pointer to
 * the emitted function. The latter can be the first thing that triggers
 * optimizing and emitting all the generated functions.
 */
static Datum
ExecRunCompiledExpr(ExprState *state, ExprContext *econtext, bool *isNull)
{
	CompiledExprState *cstate = state->evalfunc_private;
	ExprStateEvalFunc func;

	CheckExprStillValid(state, econtext);

	llvm_enter_fatal_on_oom();
	func = (ExprStateEvalFunc)llvm_get_function(cstate->context,
												cstate->funcname);
	llvm_leave_fatal_on_oom();
	Assert(func);

	/* remove indirection via this function for future calls */
	state->evalfunc = func;

	return func(state, econtext, isNull);
}

/*
 * Run compiled expression derivation.
 */
static Datum
ExecRunCompiledExprDeriv(ExprState *state, ExprContext *econtext, bool *isNull, Datum *derivatives)
{
	//printf("\nExecRunCompiledExprDeriv: Begin of func\n\n");
	CompiledExprState *cstate = state->derivefunc_private;
	ExprStateDeriveFunc func;
	
	CheckExprStillValid(state, econtext);
	//printf("\nExecRunCompiledExprDeriv: Before llvm_get_func\n\n");

	llvm_enter_fatal_on_oom();
	func = (ExprStateDeriveFunc)llvm_get_function(cstate->context,
												  cstate->funcname);
	llvm_leave_fatal_on_oom();
	Assert(func);
	
	/* remove indirection via this function for future calls */
	state->derivefunc = func;
	
	return func(state, econtext, isNull, derivatives);
}

static LLVMValueRef
BuildV1Call(LLVMJitContext *context, LLVMBuilderRef b,
			LLVMModuleRef mod, FunctionCallInfo fcinfo,
			LLVMValueRef *v_fcinfo_isnull)
{
	LLVMValueRef v_fn;
	LLVMValueRef v_fcinfo_isnullp;
	LLVMValueRef v_retval;
	LLVMValueRef v_fcinfo;

	v_fn = llvm_function_reference(context, b, mod, fcinfo);

	v_fcinfo = l_ptr_const(fcinfo, l_ptr(StructFunctionCallInfoData));
	v_fcinfo_isnullp = LLVMBuildStructGEP(b, v_fcinfo,
										  FIELDNO_FUNCTIONCALLINFODATA_ISNULL,
										  "v_fcinfo_isnull");
	LLVMBuildStore(b, l_sbool_const(0), v_fcinfo_isnullp);

	v_retval = LLVMBuildCall(b, v_fn, &v_fcinfo, 1, "funccall");

	if (v_fcinfo_isnull)
		*v_fcinfo_isnull = LLVMBuildLoad(b, v_fcinfo_isnullp, "");

	/*
	 * Add lifetime-end annotation, signalling that writes to memory don't
	 * have to be retained (important for inlining potential).
	 */
	{
		LLVMValueRef v_lifetime = create_LifetimeEnd(mod);
		LLVMValueRef params[2];

		params[0] = l_int64_const(sizeof(fcinfo->arg));
		params[1] = l_ptr_const(fcinfo->arg, l_ptr(LLVMInt8Type()));
		LLVMBuildCall(b, v_lifetime, params, lengthof(params), "");

		params[0] = l_int64_const(sizeof(fcinfo->argnull));
		params[1] = l_ptr_const(fcinfo->argnull, l_ptr(LLVMInt8Type()));
		LLVMBuildCall(b, v_lifetime, params, lengthof(params), "");

		params[0] = l_int64_const(sizeof(fcinfo->isnull));
		params[1] = l_ptr_const(&fcinfo->isnull, l_ptr(LLVMInt8Type()));
		LLVMBuildCall(b, v_lifetime, params, lengthof(params), "");
	}

	return v_retval;
}

/*
 * Implement an expression step by calling the function funcname.
 */
static void
build_EvalXFunc(LLVMBuilderRef b, LLVMModuleRef mod, const char *funcname,
				LLVMValueRef v_state, LLVMValueRef v_econtext,
				ExprEvalStep *op)
{
	LLVMTypeRef sig;
	LLVMValueRef v_fn;
	LLVMTypeRef param_types[3];
	LLVMValueRef params[3];

	v_fn = LLVMGetNamedFunction(mod, funcname);
	if (!v_fn)
	{
		param_types[0] = l_ptr(StructExprState);
		param_types[1] = l_ptr(StructExprEvalStep);
		param_types[2] = l_ptr(StructExprContext);

		sig = LLVMFunctionType(LLVMVoidType(),
							   param_types, lengthof(param_types),
							   false);
		v_fn = LLVMAddFunction(mod, funcname, sig);
	}

	params[0] = v_state;
	params[1] = l_ptr_const(op, l_ptr(StructExprEvalStep));
	params[2] = v_econtext;

	LLVMBuildCall(b,
				  v_fn,
				  params, lengthof(params), "");
}

/*
 * Implement an expression step by calling the function funcname
 * with custom params.
 */
static LLVMValueRef
build_EvalCFunc(LLVMBuilderRef b, LLVMModuleRef mod,
				const char *funcname, LLVMValueRef *params,
				LLVMTypeRef *param_types, LLVMTypeRef rettype, int nparams)
{
	LLVMTypeRef sig;
	LLVMValueRef v_fn;

	v_fn = LLVMGetNamedFunction(mod, funcname);
	if (!v_fn)
	{
		sig = LLVMFunctionType(rettype,
							   param_types, nparams,
							   false);
		v_fn = LLVMAddFunction(mod, funcname, sig);
	}

	return LLVMBuildCall(b,
						 v_fn,
						 params, nparams, "");
}

static LLVMValueRef
create_LifetimeEnd(LLVMModuleRef mod)
{
	LLVMTypeRef sig;
	LLVMValueRef fn;
	LLVMTypeRef param_types[2];

	/* LLVM 5+ has a variadic pointer argument */
#if LLVM_VERSION_MAJOR < 5
	const char *nm = "llvm.lifetime.end";
#else
	const char *nm = "llvm.lifetime.end.p0i8";
#endif

	fn = LLVMGetNamedFunction(mod, nm);
	if (fn)
		return fn;

	param_types[0] = LLVMInt64Type();
	param_types[1] = l_ptr(LLVMInt8Type());

	sig = LLVMFunctionType(LLVMVoidType(),
						   param_types, lengthof(param_types),
						   false);
	fn = LLVMAddFunction(mod, nm, sig);

	LLVMSetFunctionCallConv(fn, LLVMCCallConv);

	Assert(LLVMGetIntrinsicID(fn));

	return fn;
}

/*
 * This function jit compiles LambdaExpr-ExprStates into 
 * usable LLVM-IR derivations, which will not be emitted,  
 * until the derive-function is called for the first time 
 * through the ExprState-Function pointer
 */
bool llvm_compile_expr_derive(ExprState *state)
{
	PlanState *parent = state->parent;
	int i;
	char *funcname;

	LLVMJitContext *context = NULL;

	LLVMBuilderRef b;
	LLVMModuleRef mod;
	LLVMTypeRef eval_sig;
	LLVMValueRef eval_fn;
	LLVMBasicBlockRef entry;
	LLVMBasicBlockRef *opblocks;

	/* state itself */
	LLVMValueRef v_state;
	LLVMValueRef v_econtext;

	/* returnvalue */
	LLVMValueRef v_isnullp;

	/* tmp vars in state */
	LLVMValueRef v_tmpvaluep;
	LLVMValueRef v_tmpisnullp;

	LLVMValueRef v_derivatives;

	instr_time starttime;
	instr_time endtime;

	llvm_enter_fatal_on_oom();

	/* get or create JIT context */
	if (parent && parent->state->es_jit)
	{
		context = (LLVMJitContext *)parent->state->es_jit;
	}
	else
	{
		context = llvm_create_context(parent->state->es_jit_flags);

		if (parent)
		{
			parent->state->es_jit = &context->base;
		}
	}

	INSTR_TIME_SET_CURRENT(starttime);

	mod = llvm_mutable_module(context);

	b = LLVMCreateBuilder();

	funcname = llvm_expand_funcname(context, "diffexpr", false);

	/* Create the signature and function */
	{
		LLVMTypeRef param_types[4];

		param_types[0] = l_ptr(StructExprState);   /* state */
		param_types[1] = l_ptr(StructExprContext); /* econtext */
		param_types[2] = l_ptr(TypeParamBool);	   /* isnull */
		param_types[3] = l_ptr(TypeDatum);		   /* derivatives */

		eval_sig = LLVMFunctionType(TypeSizeT,
									param_types, lengthof(param_types),
									false);
	}
	eval_fn = LLVMAddFunction(mod, funcname, eval_sig);
	LLVMSetLinkage(eval_fn, LLVMExternalLinkage);
	LLVMSetVisibility(eval_fn, LLVMDefaultVisibility);
	llvm_copy_attributes(AttributeTemplate, eval_fn);

	entry = LLVMAppendBasicBlock(eval_fn, "entry");

	/* build state */
	v_state = LLVMGetParam(eval_fn, 0);
	v_econtext = LLVMGetParam(eval_fn, 1);
	v_isnullp = LLVMGetParam(eval_fn, 2);
	v_derivatives = LLVMGetParam(eval_fn, 3);

	LLVMPositionBuilderAtEnd(b, entry);

	v_tmpvaluep = LLVMBuildStructGEP(b, v_state,
									 FIELDNO_EXPRSTATE_RESVALUE,
									 "v.state.resvalue");
	v_tmpisnullp = LLVMBuildStructGEP(b, v_state,
									  FIELDNO_EXPRSTATE_RESNULL,
									  "v.state.resnull");

	/* allocate blocks for each op upfront, so we can do jumps easily */
	opblocks = palloc(sizeof(LLVMBasicBlockRef) * (state->steps_len));
	for (i = 0; i < state->steps_len; i++)
		opblocks[i] = l_bb_append_v(eval_fn, "b.op.%d.start", i);

	/* jump from entry to first block */
	LLVMBuildBr(b, opblocks[0]);

	for (i = 0; i < state->steps_len; i++)
	{
		ExprEvalStep *op;
		ExprEvalOp opcode;
		LLVMValueRef v_resvaluep;
		LLVMValueRef v_resnullp;

		LLVMPositionBuilderAtEnd(b, opblocks[i]);

		op = &state->steps[i];
		opcode = ExecEvalStepOp(state, op);

		v_resvaluep = l_ptr_const(op->resvalue, l_ptr(TypeSizeT));
		v_resnullp = l_ptr_const(op->resnull, l_ptr(TypeStorageBool));

		switch (opcode)
		{
		case EEOP_DONE:
		{
			LLVMValueRef v_tmpisnull, v_tmpvalue, v_seed;
			
			v_tmpvalue = LLVMBuildLoad(b, v_tmpvaluep, "");
			v_tmpisnull = LLVMBuildLoad(b, v_tmpisnullp, "");
			v_tmpisnull =
				LLVMBuildTrunc(b, v_tmpisnull, TypeParamBool, "");

			LLVMBuildStore(b, v_tmpisnull, v_isnullp);

			if(state->lambdaContainsMatrix) {
				LLVMValueRef params[1];
				LLVMTypeRef types[1];
				if ((ExecEvalStepOp(state, &(state->steps[state->steps_len - 2])) == 17 ||
					 ExecEvalStepOp(state, &(state->steps[state->steps_len - 2])) == 18) &&
					state->steps[state->steps_len - 2].d.func.finfo->fn_oid == 7801) // This checks, if expressionsstep is func-call and calls softmax as last call
				{
					types[0] = LLVMDoubleType();
					params[0] = l_float8_const(1.0);

					v_seed = build_EvalCFunc(b, mod, "createScalar",
											 (LLVMValueRef *)&params,
											 (LLVMTypeRef *)&types,
											 TypeDatum,
											 1);
				}
				else
				{
					types[0] = LLVMDoubleType();
					params[0] = l_float8_const(1.0);

					v_seed = build_EvalCFunc(b, mod, "createScalar",
											 (LLVMValueRef *)&params,
											 (LLVMTypeRef *)&types,
											 TypeDatum,
											 1);
				}
			} else {
				v_seed = l_float8_const(1.0);
			}

			llvm_compile_expr_deriv_subtree(b, mod, state, state->steps_len - 2, v_seed, v_derivatives);

			LLVMBuildRet(b, v_tmpvalue);
			break;
		}
		case EEOP_CONST:
		{
			LLVMValueRef v_constvalue,
				v_constnull;

			v_constvalue = l_sizet_const(op->d.constval.value);
			v_constnull = l_sbool_const(op->d.constval.isnull);

			LLVMBuildStore(b, v_constvalue, v_resvaluep);
			LLVMBuildStore(b, v_constnull, v_resnullp);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}

		case EEOP_FUNCEXPR_STRICT:
		{
			FunctionCallInfo fcinfo = op->d.func.fcinfo_data;
			LLVMBasicBlockRef b_nonull;
			int argno;
			LLVMValueRef v_fcinfo;
			LLVMValueRef v_argnullp;
			LLVMBasicBlockRef *b_checkargnulls;

			/*
					 * Block for the actual function call, if args are
					 * non-NULL.
					 */
			b_nonull = l_bb_before_v(opblocks[i + 1],
									 "b.%d.no-null-args", i);

			/* should make sure they're optimized beforehand */
			if (op->d.func.nargs == 0)
				elog(ERROR, "argumentless strict functions are pointless");

			v_fcinfo =
				l_ptr_const(fcinfo, l_ptr(StructFunctionCallInfoData));

			v_argnullp =
				LLVMBuildStructGEP(b,
								   v_fcinfo,
								   FIELDNO_FUNCTIONCALLINFODATA_ARGNULL,
								   "v_argnullp");

			/*
					 * set resnull to true, if the function is actually
					 * called, it'll be reset
					 */
			LLVMBuildStore(b, l_sbool_const(1), v_resnullp);

			/* create blocks for checking args, one for each */
			b_checkargnulls =
				palloc(sizeof(LLVMBasicBlockRef *) * op->d.func.nargs);
			for (argno = 0; argno < op->d.func.nargs; argno++)
				b_checkargnulls[argno] =
					l_bb_before_v(b_nonull, "b.%d.isnull.%d", i, argno);

			/* jump to check of first argument */
			LLVMBuildBr(b, b_checkargnulls[0]);

			/* check each arg for NULLness */
			for (argno = 0; argno < op->d.func.nargs; argno++)
			{
				LLVMValueRef v_argisnull;
				LLVMBasicBlockRef b_argnotnull;

				LLVMPositionBuilderAtEnd(b, b_checkargnulls[argno]);

				/* compute block to jump to if argument is not null */
				if (argno + 1 == op->d.func.nargs)
					b_argnotnull = b_nonull;
				else
					b_argnotnull = b_checkargnulls[argno + 1];

				/* and finally load & check NULLness of arg */
				v_argisnull = l_load_struct_gep(b, v_argnullp,
												argno, "");
				LLVMBuildCondBr(b,
								LLVMBuildICmp(b, LLVMIntEQ,
											  v_argisnull,
											  l_sbool_const(1),
											  ""),
								opblocks[i + 1],
								b_argnotnull);
			}

			LLVMPositionBuilderAtEnd(b, b_nonull);

			//FunctionCallInfo fcinfo = op->d.func.fcinfo_data;
			LLVMValueRef v_fcinfo_isnull;
			LLVMValueRef v_retval;

			v_retval = BuildV1Call(context, b, mod, fcinfo,
								   &v_fcinfo_isnull);
			LLVMBuildStore(b, v_retval, v_resvaluep);
			LLVMBuildStore(b, v_fcinfo_isnull, v_resnullp);

			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}
		case EEOP_PARAM_EXTERN:
		{
			build_EvalXFunc(b, mod, "ExecEvalFastParamExtern",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}
		case EEOP_FIELDSELECT:
		{
			build_EvalXFunc(b, mod, "ExecEvalFastFieldSelect",
							v_state, v_econtext, op);
			LLVMBuildBr(b, opblocks[i + 1]);
			break;
		}
		default:
		{
			Assert(false);
			break;
		}
		}
	}

	LLVMDisposeBuilder(b);

	/*
	 * Don't immediately emit function, instead do so the first time the
	 * expression is actually evaluated. That allows to emit a lot of
	 * functions together, avoiding a lot of repeated llvm and memory
	 * remapping overhead.
	 */
	{
		CompiledExprState *cstate = palloc0(sizeof(CompiledExprState));

		cstate->context = context;
		cstate->funcname = funcname;

		state->derivefunc = ExecRunCompiledExprDeriv;
		state->derivefunc_private = cstate;
	}

	llvm_leave_fatal_on_oom();

	INSTR_TIME_SET_CURRENT(endtime);
	INSTR_TIME_ACCUM_DIFF(context->base.instr.generation_counter,
						  endtime, starttime);

	return true;
}

static int
llvm_compile_expr_deriv_subtree(LLVMBuilderRef b, 		/* Builder containing the pre-built eval-func */
                                LLVMModuleRef mod,		/* The module where the build function will be stored */
                                ExprState *state, 		/* State containing intermediate values for the derivations */
								int fetchIndex,   		/* The step in the op_code sequence from where to start */
								LLVMValueRef seed,		/* seed for current subtree, not a pointer */
								LLVMValueRef derivatives) /* Datum(therefore pointer) array, containing all derivatives */
{
	int resultFetchIndex = fetchIndex;
	switch (ExecEvalStepOp(state, &(state->steps[fetchIndex])))
	{
	case 59: /*EEOP_FIELDSELECT*/ 
	{
		LLVMValueRef v_derivative_p,		   	/* The pointer into the derivatives array */
			v_tmpfieldnum,		   				/* The fieldNum parameter of the Fieldselect */
			v_tmpderivative;	   				/* The value behind the datum-pointer */

		int fieldNum = (state->indexArray[state->steps[fetchIndex - 1].d.param.paramid - 1]) +
					   (state->steps[fetchIndex].d.fieldselect.fieldnum - 1);

		v_tmpfieldnum = l_int32_const(fieldNum);
		v_derivative_p = LLVMBuildGEP(b,
									  derivatives,
									  &v_tmpfieldnum,
									  1, "");
		if(state->lambdaContainsMatrix) {
			LLVMValueRef matrix_add_params[2];
			LLVMTypeRef matrix_add_types[2];

			matrix_add_params[0] = LLVMBuildLoad(b, v_derivative_p, "");
			matrix_add_params[1] = seed;

			matrix_add_types[0] = TypeDatum;
			matrix_add_types[1] = TypeDatum;

			v_tmpderivative = build_EvalCFunc(b, mod, "matrix_add_inplace", (LLVMValueRef *)&matrix_add_params,
											  (LLVMTypeRef *)&matrix_add_types, TypeDatum, 2);
		} else {
			v_tmpderivative = LLVMBuildBinOp(b, LLVMFAdd,
											 l_as_float8(b, seed),
											 l_as_float8(b, LLVMBuildLoad(b, v_derivative_p, "")),
											 "");
		}
		LLVMBuildStore(b, v_tmpderivative, v_derivative_p);
		resultFetchIndex = fetchIndex - 2;
		break;
	}
	case 16: /*EEOP_CONST*/
	{
		resultFetchIndex = fetchIndex - 1; //can be skipped, d_const/d_x=0
		break;
	}
	case 17: /* EEOP_FUNCEXPR */
	case 18: /*EEOP_FUNCEXPR_STRICT*/
	{
		switch (state->steps[fetchIndex].d.func.finfo->fn_oid)
		{
		case 1726:
		case 216: /*float8 binary multiplication*/
		{
			LLVMValueRef x, y, newSeedX, newSeedY;
			int startingPointY, stepAfterX;

			//TODO: Read value from fn_info
			x = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), ""));
			y = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[1], l_ptr(TypeDatum)), ""));

			newSeedX = LLVMBuildBinOp(b, LLVMFMul, y, l_as_float8(b, seed), "");
			newSeedY = LLVMBuildBinOp(b, LLVMFMul, x, l_as_float8(b, seed), "");

			startingPointY = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedY, derivatives);
			stepAfterX = llvm_compile_expr_deriv_subtree(b, mod, state, startingPointY, newSeedX, derivatives);
			resultFetchIndex = stepAfterX;
			break;
		}
		case 1727:
		case 217: /*float8 binary divison*/
		{
			LLVMValueRef x, y, newSeedX, newSeedY;
			int startingPointY, stepAfterX;

			x = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), ""));
			y = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[1], l_ptr(TypeDatum)), ""));

			newSeedX = LLVMBuildBinOp(b, LLVMFDiv, l_as_float8(b, seed), y, "");
			newSeedY = LLVMBuildBinOp(b,
									  LLVMFDiv,
									  (LLVMBuildBinOp(b,
													  LLVMFMul,
													  l_as_float8(b, seed),
													  (LLVMBuildBinOp(b, LLVMFMul, x, l_float8_const(-1.0), "")),
													  "")),
									  (LLVMBuildBinOp(b,
													  LLVMFMul,
													  y,
													  y,
													  "")),
									  "");

			startingPointY = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedY, derivatives);
			stepAfterX = llvm_compile_expr_deriv_subtree(b, mod, state, startingPointY, newSeedX, derivatives);
			resultFetchIndex = stepAfterX;
			break;
		}
		case 1724:
		case 218: /*float8 binary addition*/
		{
			int startingPointY, stepAfterX;

			startingPointY = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, seed, derivatives);
			stepAfterX = llvm_compile_expr_deriv_subtree(b, mod, state, startingPointY, seed, derivatives);
			resultFetchIndex = stepAfterX;
			break;
		}
		case 1725:
		case 219: /*float8 binary subtraction*/
		{
			int startingPointY, stepAfterX;

			startingPointY = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1,
															 LLVMBuildBinOp(
																 b,
																 LLVMFMul,
																 seed,
																 l_float8_const(-1.0),
																 ""),
															 derivatives);
			stepAfterX = llvm_compile_expr_deriv_subtree(b, mod, state, startingPointY, seed, derivatives);
			resultFetchIndex = stepAfterX;
			break;
		}
		case 232:
		case 1346: /*float8 binary pow x^y*/
		{
			LLVMValueRef x, y, newSeedX, newSeedY, params_pow_x[2], params_pow_y[2], params_log[1];
			LLVMTypeRef types[2];
			int startingPointY, stepAfterX;

			types[0] = LLVMDoubleType();
			types[1] = LLVMDoubleType();

			x = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), ""));
			y = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[1], l_ptr(TypeDatum)), ""));

			params_pow_x[0] = x;
			params_pow_x[1] = LLVMBuildBinOp(b, LLVMFSub, y, l_float8_const(1), "");

			params_pow_y[0] = x;
			params_pow_y[1] = y;

			params_log[0] = x;

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFMul,
									  LLVMBuildBinOp(b,
													 LLVMFMul,
													 y,
													 l_as_float8(b, seed),
													 ""),
									  build_EvalCFunc(b, 
									                  mod, 
													  "pow", 
													  (LLVMValueRef *)&params_pow_x, 
													  (LLVMTypeRef *)&types, 
													  types[0], 
													  2),
									  "");
			newSeedY = LLVMBuildBinOp(b,
									  LLVMFMul,
									  LLVMBuildBinOp(b,
													 LLVMFMul,
													 build_EvalCFunc(b,
																	 mod,
																	 "pow",
																	 (LLVMValueRef *)&params_pow_y,
																	 (LLVMTypeRef *)&types,
																	 types[0],
																	 2),
													 build_EvalCFunc(b,
																	 mod,
																	 "log",
																	 (LLVMValueRef *)&params_log,
																	 (LLVMTypeRef *)&types,
																	 types[0],
																	 1),
													 ""),
									  l_as_float8(b, seed),
									  "");

			startingPointY = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedY, derivatives);
			stepAfterX = llvm_compile_expr_deriv_subtree(b, mod, state, startingPointY, newSeedX, derivatives);
			resultFetchIndex = stepAfterX;
			break;
		}
		case 230:
		case 1344: /*float8 unary sqrt*/
		{
			LLVMValueRef x, newSeedX, params_sqrt[1];
			LLVMTypeRef types[1];
			int stepAfterX;

			types[0] = LLVMDoubleType();

			x = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), ""));

			params_sqrt[0] = x;

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFDiv,
									  l_as_float8(b, seed),
									  LLVMBuildBinOp(b,
													 LLVMFMul,
													 l_float8_const(2.0),
									  				 build_EvalCFunc(b,
																	 mod,
																	 "sqrt",
																	 (LLVMValueRef *)&params_sqrt,
																	 (LLVMTypeRef *)&types,
																	 types[0],
																	 1),
											         ""),
									 "");

			stepAfterX = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepAfterX;
			break;
		}
		case 221:
		case 1395: /*float8 unary abs*/
		{
			LLVMValueRef x, newSeedX, params_abs[1];
			LLVMTypeRef types[1];
			int stepAfterX;

			types[0] = LLVMDoubleType();

			x = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), ""));

			params_abs[0] = x;

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFMul,
									  l_as_float8(b, seed),
									  LLVMBuildBinOp(b,
													 LLVMFDiv,
													 x,
													 build_EvalCFunc(b,
																	 mod,
																	 "fabs",
																	 (LLVMValueRef *)&params_abs,
																	 (LLVMTypeRef *)&types,
																	 types[0],
																	 1),
													 ""),
									  "");

			stepAfterX = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepAfterX;
			break;
		}
		case 1604: /*float8 unary sin*/
		{
			LLVMValueRef x, newSeedX, params_cos[1];
			LLVMTypeRef types[1];
			int stepAfterX;

			types[0] = LLVMDoubleType();

			x = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), ""));

			params_cos[0] = x;

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFMul,
									  l_as_float8(b, seed),
									  build_EvalCFunc(b,
													  mod,
													  "cos",
													  (LLVMValueRef *)&params_cos,
													  (LLVMTypeRef *)&types,
													  types[0],
													  1),
									  "");

			stepAfterX = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepAfterX;
			break;
		}
		case 1605: /*float8 unary cos*/
		{
			LLVMValueRef x, newSeedX, params_sin[1];
			LLVMTypeRef types[1];
			int stepAfterX;

			types[0] = LLVMDoubleType();

			x = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), ""));

			params_sin[0] = x;

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFMul,
									  LLVMBuildBinOp(
										  b,
										  LLVMFMul,
										  seed,
										  l_float8_const(-1.0),
										  ""),
									  build_EvalCFunc(b,
													  mod,
													  "sin",
													  (LLVMValueRef *)&params_sin,
													  (LLVMTypeRef *)&types,
													  types[0],
													  1),
									  "");

			stepAfterX = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepAfterX;
			break;
		}
		case 1347: /*float8 unary exp*/
		{
			LLVMValueRef x, newSeedX, params_exp[1];
			LLVMTypeRef types[1];
			int stepAfterX;

			types[0] = LLVMDoubleType();

			x = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), ""));

			params_exp[0] = x;

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFMul,
									  l_as_float8(b, seed),
									  build_EvalCFunc(b,
													  mod,
													  "exp",
													  (LLVMValueRef *)&params_exp,
													  (LLVMTypeRef *)&types,
													  types[0],
													  1),
									  "");

			stepAfterX = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepAfterX;
			break;
		}
		case 220: /* float unary minus/negation */
		{
			LLVMValueRef newSeedX;
			int stepsAfterSubtree;

			newSeedX = LLVMBuildBinOp(
				b,
				LLVMFMul,
				seed,
				l_float8_const(-1.0),
				""
			);

			stepsAfterSubtree = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 1600: /* float arcus sine */
		{
			LLVMValueRef newSeedX, x, params_sqrt_1[1], params_sqrt_2[1];
			LLVMTypeRef types[1];
			int stepsAfterSubtree;

			types[0] = LLVMDoubleType();

			x = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), ""));

			params_sqrt_1[0] = LLVMBuildBinOp(b,
			 								  LLVMFSub,
											  l_float8_const(1.0),
											  x,
											  "");
			
			params_sqrt_2[0] = LLVMBuildBinOp(b,
											  LLVMFAdd,
											  l_float8_const(1.0),
											  x,
											  "");

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFDiv,
									  seed,
									  LLVMBuildBinOp(b,
													 LLVMFMul,
													 build_EvalCFunc(b,
																	 mod,
																	 "sqrt",
																	 (LLVMValueRef *)&params_sqrt_1,
																	 (LLVMTypeRef *)&types,
																	 types[0],
																	 1),
													 build_EvalCFunc(b,
																	 mod,
																	 "sqrt",
																	 (LLVMValueRef *)&params_sqrt_2,
																	 (LLVMTypeRef *)&types,
																	 types[0],
																	 1),
													 ""),
									  "");

			stepsAfterSubtree = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 1601: /* float arcus cosine */
		{
			LLVMValueRef newSeedX, x, params_sqrt_1[1], params_sqrt_2[1];
			LLVMTypeRef types[1];
			int stepsAfterSubtree;

			types[0] = LLVMDoubleType();

			x = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), ""));

			params_sqrt_1[0] = LLVMBuildBinOp(b,
											  LLVMFSub,
											  l_float8_const(1.0),
											  x,
											  "");

			params_sqrt_2[0] = LLVMBuildBinOp(b,
											  LLVMFAdd,
											  l_float8_const(1.0),
											  x,
											  "");

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFDiv,
									  LLVMBuildBinOp(
										  b,
										  LLVMFMul,
										  seed,
										  l_float8_const(-1.0),
										  ""),
									  LLVMBuildBinOp(b,
													 LLVMFMul,
													 build_EvalCFunc(b,
																	 mod,
																	 "sqrt",
																	 (LLVMValueRef *)&params_sqrt_1,
																	 (LLVMTypeRef *)&types,
																	 types[0],
																	 1),
													 build_EvalCFunc(b,
																	 mod,
																	 "sqrt",
																	 (LLVMValueRef *)&params_sqrt_2,
																	 (LLVMTypeRef *)&types,
																	 types[0],
																	 1),
													 ""),
									  "");

			stepsAfterSubtree = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 1602: /* float unary arcus tangens */
		{
			LLVMValueRef newSeedX, x;
			int stepsAfterSubtree;

			x = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), ""));

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFDiv,
									  seed,
									  LLVMBuildBinOp(b,
													 LLVMFAdd,
													 l_float8_const(1.0),
													 LLVMBuildBinOp(b,
																	LLVMFMul,
																	x,
																	x,
																	""),
													 ""),
									  "");

			stepsAfterSubtree = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 1603: /* float binary arcus tangens 2 (c: atan2)(normally defined as: atan2(y,x), but in this case, x is the first and y the second argument) */
		{
			LLVMValueRef x, y, newSeedX, newSeedY, tmp_newSeedY;
			int startingPointForY, stepIndexAfterX;

			x = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), ""));
			y = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[1], l_ptr(TypeDatum)), ""));

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFMul,
									  seed,
									  LLVMBuildBinOp(b,
													 LLVMFDiv,
													 y,
													 LLVMBuildBinOp(b,
																	LLVMFAdd,
																	LLVMBuildBinOp(b,
																				   LLVMFMul,
																				   x,
																				   x,
																				   ""),
																	LLVMBuildBinOp(b,
																				   LLVMFMul,
																				   y,
																				   y,
																				   ""),
																	""),
													 ""),
									  "");
			tmp_newSeedY = LLVMBuildBinOp(b,
										  LLVMFMul,
										  seed,
										  LLVMBuildBinOp(b,
														 LLVMFDiv,
														 x,
														 LLVMBuildBinOp(b,
																		LLVMFAdd,
																		LLVMBuildBinOp(b,
																					   LLVMFMul,
																					   x,
																					   x,
																					   ""),
																		LLVMBuildBinOp(b,
																					   LLVMFMul,
																					   y,
																					   y,
																					   ""),
																		""),
														 ""),
										  "");

			newSeedY = LLVMBuildBinOp(
				b,
				LLVMFMul,
				tmp_newSeedY,
				l_float8_const(-1.0),
				"");

			startingPointForY = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedY, derivatives);
			stepIndexAfterX = llvm_compile_expr_deriv_subtree(b, mod, state, startingPointForY, newSeedX, derivatives);
			resultFetchIndex = stepIndexAfterX;
			break;
		}
		case 1606: /* float tangens */
		{
			LLVMValueRef x, newSeedX, params_cos[1], tmp_cos;
			LLVMTypeRef types[1];
			int stepsAfterSubtree;

			x = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), ""));
			params_cos[0] = x;
			types[0] = LLVMDoubleType();

			tmp_cos = build_EvalCFunc(b,
									  mod,
									  "cos",
									  (LLVMValueRef *)&params_cos,
									  (LLVMTypeRef *)&types,
									  types[0],
									  1);

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFDiv,
									  seed,
									  LLVMBuildBinOp(b,
									  				 LLVMFMul,
													 tmp_cos,
													 tmp_cos,
													 ""),
									  "");

			stepsAfterSubtree = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 1607: /* float co-tangens */
		{
			LLVMValueRef x, newSeedX, params_sin[1], tmp_sin;
			LLVMTypeRef types[1];
			int stepsAfterSubtree;

			x = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), ""));
			params_sin[0] = x;
			types[0] = LLVMDoubleType();

			tmp_sin = build_EvalCFunc(b,
									  mod,
									  "sin",
									  (LLVMValueRef *)&params_sin,
									  (LLVMTypeRef *)&types,
									  types[0],
									  1);

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFDiv,
									  LLVMBuildBinOp(
										  b,
										  LLVMFMul,
										  seed,
										  l_float8_const(-1.0),
										  ""),
									  LLVMBuildBinOp(b,
													 LLVMFMul,
													 tmp_sin,
													 tmp_sin,
													 ""),
									  "");

			stepsAfterSubtree = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 1339: /* float log base 10 */
		{
			LLVMValueRef x, newSeedX, naturalLogOf10;
			int stepsAfterSubtree;

			x = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), ""));
			naturalLogOf10 = l_float8_const(2.3025850929940456840179914546843642076011014886287729760333279009);

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFDiv,
									  seed,
									  LLVMBuildBinOp(b,
									  				 LLVMFMul,
													 x,
													 naturalLogOf10,
													 ""),
									  "");

			stepsAfterSubtree = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 1341: /* float natural log */
		{
			LLVMValueRef x, newSeedX;
			int stepsAfterSubtree;

			x = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), ""));

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFDiv,
									  seed,
									  x,
									  "");

			stepsAfterSubtree = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 7801: /* softmax */ 
		{
			LLVMValueRef x, y, newSeedX, newSeedY, tmp, softmax_params[2], mat_mul_params[4], scalar_param[1];
			LLVMTypeRef softmax_types[2], mat_mul_types[4], scalar_type[1];
			int stepIndexAfterX, stepIndexAfterY;

			x = LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), "");
			y = LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[1], l_ptr(TypeDatum)), "");

			softmax_params[0] = x;
			softmax_params[1] = y;

			softmax_types[0] = TypeDatum;
			softmax_types[1] = TypeDatum;

			tmp = build_EvalCFunc(b, mod, "softmax_cce_derive", (LLVMValueRef *)&softmax_params, (LLVMTypeRef *)&softmax_types, TypeDatum, 2);

			mat_mul_params[0] = seed;
			mat_mul_params[1] = tmp;
			mat_mul_params[2] = l_pbool_const(false);
			mat_mul_params[3] = l_pbool_const(false);

			mat_mul_types[0] = TypeDatum;
			mat_mul_types[1] = TypeDatum;
			mat_mul_types[2] = TypeParamBool;
			mat_mul_types[3] = TypeParamBool;

			newSeedX = build_EvalCFunc(b, mod,
									   "matrix_mul_internal",
									   (LLVMValueRef *)&mat_mul_params,
									   (LLVMTypeRef *)&mat_mul_types,
									   TypeDatum, 4);

			scalar_param[0] = l_float8_const(0.0);
			scalar_type[0] = LLVMDoubleType();

			newSeedY = build_EvalCFunc(b, mod, "createScalar", (LLVMValueRef *)&scalar_param, (LLVMTypeRef *)&scalar_type, TypeDatum, 1);

			stepIndexAfterY = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedY, derivatives);
			stepIndexAfterX = llvm_compile_expr_deriv_subtree(b, mod, state, stepIndexAfterY, newSeedX, derivatives);
			resultFetchIndex = stepIndexAfterX;
			break;
		}
		case 7802: /* float sigmoidial rectified linear unit */
		{
			LLVMValueRef x, newSeedX, tmp, params[1], eToMX;
			LLVMTypeRef types[1];
			int stepsAfterSubtree;

			x = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), ""));

			types[0] = LLVMDoubleType();
			params[0] = LLVMBuildBinOp(b,
									   LLVMFMul,
									   l_float8_const(-1.0),
									   l_as_float8(b, x),
									   "");

			eToMX = build_EvalCFunc(b, mod, "exp", (LLVMValueRef *)&params, (LLVMTypeRef *)&types, types[0], 1);

			tmp = LLVMBuildBinOp(b,
								 LLVMFDiv,
								 LLVMBuildBinOp(b,
												LLVMFAdd,
												LLVMBuildBinOp(b,
															   LLVMFAdd,
															   l_float8_const(1.0),
															   eToMX,
															   ""),
												LLVMBuildBinOp(b,
															   LLVMFMul,
															   l_as_float8(b, x),
															   eToMX,
															   ""),
												""),
								 LLVMBuildBinOp(b,
												LLVMFMul,
												LLVMBuildBinOp(b,
															   LLVMFAdd,
															   l_float8_const(1.0),
															   eToMX,
															   ""),
												LLVMBuildBinOp(b,
															   LLVMFAdd,
															   l_float8_const(1.0),
															   eToMX,
															   ""),
												""),
								 "");

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFMul,
									  l_as_float8(b, seed),
									  l_as_float8(b, tmp),
									  "");
			// newSeedX = l_float8_const(2);

			stepsAfterSubtree = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 7803: /* float sigmoid */
		{
			LLVMValueRef x, newSeedX, tmp, params[1], sig_val;
			LLVMTypeRef types[1];
			int stepsAfterSubtree;

			x = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), ""));

			types[0] = LLVMDoubleType();
			params[0] = LLVMBuildBinOp(b,
									   LLVMFMul,
									   l_float8_const(-1.0),
									   l_as_float8(b, x),
									   "");

			sig_val = LLVMBuildBinOp(b,
									 LLVMFDiv,
									 l_float8_const(1.0),
									 LLVMBuildBinOp(b,
													LLVMFAdd,
													l_float8_const(1.0),
													build_EvalCFunc(b, mod, "exp", (LLVMValueRef *)&params, (LLVMTypeRef *)&types, types[0], 1),
													""),
									 "");

			tmp = LLVMBuildBinOp(b,
								 LLVMFMul,
								 sig_val,
								 LLVMBuildBinOp(b,
												LLVMFSub,
												l_float8_const(1.0),
												sig_val,
												""),
								 "");

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFMul,
									  seed,
									  tmp,
									  "");

			stepsAfterSubtree = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 7804: /* float tangens hyperbolicus */
		{
			LLVMValueRef x, newSeedX, tmp, params[1], tanh_val;
			LLVMTypeRef types[1];
			int stepsAfterSubtree;

			x = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), ""));

			types[0] = LLVMDoubleType();
			params[0] = l_as_float8(b, x);

			tanh_val = build_EvalCFunc(b, mod, "tanh", (LLVMValueRef *)&params, (LLVMTypeRef *)&types, types[0], 1);

			tmp = LLVMBuildBinOp(b,
								 LLVMFSub,
								 l_float8_const(1.0),
								 LLVMBuildBinOp(b,
												LLVMFMul,
												tanh_val,
												tanh_val,
												""),
								 "");

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFMul,
									  seed,
									  tmp,
									  "");

			stepsAfterSubtree = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 7805: /* float rectified linear unit(relu) */
		{
			LLVMValueRef newSeedX, x, factor;
			LLVMTypeRef typeDouble;
			int stepsAfterSubtree;

			typeDouble = LLVMDoubleType();

			x = l_as_float8(b, LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), ""));

			factor = LLVMBuildUIToFP(b, LLVMBuildFCmp(b, LLVMRealOGT, x, l_float8_const(0.0), ""), typeDouble, "");
			newSeedX = LLVMBuildBinOp(b,
									  LLVMFMul,
									  l_as_float8(b, seed),
									  l_as_float8(b, factor),
									  "");

			stepsAfterSubtree = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 9000: /* Matrix multiplication */
		{
			LLVMValueRef x, y, newSeedX, newSeedY, mat_mul_params_x[4], mat_mul_params_y[4];
			LLVMTypeRef mat_mul_types_x[4], mat_mul_types_y[4];
			int startingPointForY, stepIndexAfterX;

			x = LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), "");
			y = LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[1], l_ptr(TypeDatum)), "");

			mat_mul_params_x[0] = seed;
			mat_mul_params_x[1] = y;
			mat_mul_params_x[2] = l_pbool_const(false);
			mat_mul_params_x[3] = l_pbool_const(true);

			mat_mul_params_y[0] = x;
			mat_mul_params_y[1] = seed;
			mat_mul_params_y[2] = l_pbool_const(true);
			mat_mul_params_y[3] = l_pbool_const(false);

			mat_mul_types_x[0] = TypeDatum;
			mat_mul_types_x[1] = TypeDatum;
			mat_mul_types_x[2] = TypeParamBool;
			mat_mul_types_x[3] = TypeParamBool;

			mat_mul_types_y[0] = TypeDatum;
			mat_mul_types_y[1] = TypeDatum;
			mat_mul_types_y[2] = TypeParamBool;
			mat_mul_types_y[3] = TypeParamBool;

			newSeedX = build_EvalCFunc(b, mod, "matrix_mul_internal", (LLVMValueRef *)&mat_mul_params_x, (LLVMTypeRef *)&mat_mul_types_x, TypeDatum, 4);
			newSeedY = build_EvalCFunc(b, mod, "matrix_mul_internal", (LLVMValueRef *)&mat_mul_params_y, (LLVMTypeRef *)&mat_mul_types_y, TypeDatum, 4);

			startingPointForY = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedY, derivatives);
			stepIndexAfterX = llvm_compile_expr_deriv_subtree(b, mod, state, startingPointForY, newSeedX, derivatives);
			resultFetchIndex = stepIndexAfterX;
			break;
		}
		case 9001: /* matrix sigmoidial linear unit(silu) */
		{
			LLVMValueRef newSeedX, x, mat_elem_mul[2], mat_silu;
			LLVMTypeRef mat_elem_mul_types[2], mat_silu_type;
			int stepsAfterSubtree;

			x = LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), "");

			mat_elem_mul_types[0] = TypeDatum;
			mat_elem_mul_types[1] = TypeDatum;
			mat_silu_type = TypeDatum;

			mat_elem_mul[0] = seed;
			mat_silu = x;
			mat_elem_mul[1] = build_EvalCFunc(b, mod, "silu_m_derive", (LLVMValueRef *)&mat_silu, (LLVMTypeRef *)&mat_silu_type, TypeDatum, 1);

			newSeedX = build_EvalCFunc(b, mod, "matrix_elem_mult", (LLVMValueRef *)&mat_elem_mul, (LLVMTypeRef *)&mat_elem_mul_types, TypeDatum, 2);

			stepsAfterSubtree = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 9002: /* matrix sigmoid */
		{
			LLVMValueRef newSeedX, x, mat_elem_mul[2], mat_silu;
			LLVMTypeRef mat_elem_mul_types[2], mat_silu_type;
			int stepsAfterSubtree;

			x = LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), "");

			mat_elem_mul_types[0] = TypeDatum;
			mat_elem_mul_types[1] = TypeDatum;
			mat_silu_type = TypeDatum;

			mat_elem_mul[0] = seed;
			mat_silu = x;
			mat_elem_mul[1] = build_EvalCFunc(b, mod, "sigmoid_m_derive", (LLVMValueRef *)&mat_silu, (LLVMTypeRef *)&mat_silu_type, TypeDatum, 1);

			newSeedX = build_EvalCFunc(b, mod, "matrix_elem_mult", (LLVMValueRef *)&mat_elem_mul, (LLVMTypeRef *)&mat_elem_mul_types, TypeDatum, 2);

			stepsAfterSubtree = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 9003: /* matrix tanh */
		{
			LLVMValueRef newSeedX, x, mat_elem_mul[2], mat_silu;
			LLVMTypeRef mat_elem_mul_types[2], mat_silu_type;
			int stepsAfterSubtree;

			x = LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), "");

			mat_elem_mul_types[0] = TypeDatum;
			mat_elem_mul_types[1] = TypeDatum;
			mat_silu_type = TypeDatum;

			mat_elem_mul[0] = seed;
			mat_silu = x;
			mat_elem_mul[1] = build_EvalCFunc(b, mod, "tanh_m_derive", (LLVMValueRef *)&mat_silu, (LLVMTypeRef *)&mat_silu_type, TypeDatum, 1);

			newSeedX = build_EvalCFunc(b, mod, "matrix_elem_mult", (LLVMValueRef *)&mat_elem_mul, (LLVMTypeRef *)&mat_elem_mul_types, TypeDatum, 2);

			stepsAfterSubtree = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 9004: /* matrix rectified linear unit(relu) */
		{
			LLVMValueRef newSeedX, x, mat_elem_mul[2], mat_silu;
			LLVMTypeRef mat_elem_mul_types[2], mat_silu_type;
			int stepsAfterSubtree;

			x = LLVMBuildLoad(b, l_ptr_const((void *)&state->steps[fetchIndex].d.func.fcinfo_data->arg[0], l_ptr(TypeDatum)), "");

			mat_elem_mul_types[0] = TypeDatum;
			mat_elem_mul_types[1] = TypeDatum;
			mat_silu_type = TypeDatum;

			mat_elem_mul[0] = seed;
			mat_silu = x;
			mat_elem_mul[1] = build_EvalCFunc(b, mod, "relu_m_derive", (LLVMValueRef *)&mat_silu, (LLVMTypeRef *)&mat_silu_type, TypeDatum, 1);

			newSeedX = build_EvalCFunc(b, mod, "matrix_elem_mult", (LLVMValueRef *)&mat_elem_mul, (LLVMTypeRef *)&mat_elem_mul_types, TypeDatum, 2);

			stepsAfterSubtree = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 9005: /*matrix float8 binary addition*/
		{
			int startingPointY, stepAfterX;

			startingPointY = llvm_compile_expr_deriv_subtree(b, mod, state, fetchIndex - 1, seed, derivatives);
			stepAfterX = llvm_compile_expr_deriv_subtree(b, mod, state, startingPointY, seed, derivatives);
			resultFetchIndex = stepAfterX;
			break;
		}
		default:
		{
			ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED), errmsg("Derive(L2): current operator not supported, aborting...")));
			break;
		}
		}
		break;
	}
	default:
	{
		ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED), errmsg("Derive(L2): current step-opcode not recognized, aborting...")));
		break;
	}
	}
	return resultFetchIndex;
}

bool llvm_compile_simple_expr_derive(ExprState *state)
{
	PlanState *parent = state->parent;
	int i;
	char *funcname;

	LLVMJitContext *context = NULL;

	LLVMBuilderRef b;
	LLVMModuleRef mod;
	LLVMTypeRef eval_sig;
	LLVMValueRef eval_fn;
	
	LLVMValueRef datum_param;
	LLVMValueRef v_derivatives;

	LLVMBasicBlockRef entry;
	LLVMBasicBlockRef *opblocks;
	LLVMValueRef registers[50];
	LLVMValueRef intermediate_vals[2 * state->steps_len];
	int registerPointer = 0;
	int funcInputPointer = 0;

	instr_time starttime;
	instr_time endtime;

	llvm_enter_fatal_on_oom();

	/* get or create JIT context */
	if (parent && parent->state->es_jit)
	{
		context = (LLVMJitContext *)parent->state->es_jit;
	}
	else
	{
		context = llvm_create_context(parent->state->es_jit_flags);

		if (parent)
		{
			parent->state->es_jit = &context->base;
		}
	}

	INSTR_TIME_SET_CURRENT(starttime);

	mod = llvm_mutable_module(context);

	b = LLVMCreateBuilder();

	funcname = llvm_expand_funcname(context, "diffexpr", true);

	/* Create the signature and function */
	{
		LLVMTypeRef param_types[2];

		param_types[0] = l_ptr(l_ptr(TypeDatum));	/* function parameters, inputs */
		param_types[1] = l_ptr(TypeDatum); 			/* function parameters, derivatives */

		eval_sig = LLVMFunctionType(TypeDatum,
									param_types, lengthof(param_types),
									false);
	}

	eval_fn = LLVMAddFunction(mod, funcname, eval_sig);
	LLVMSetLinkage(eval_fn, LLVMExternalLinkage);
	LLVMSetVisibility(eval_fn, LLVMDefaultVisibility);

	entry = LLVMAppendBasicBlock(eval_fn, "entry");

	LLVMPositionBuilderAtEnd(b, entry);

	/* allocate blocks for each op upfront, so we can do jumps easily */
	opblocks = palloc(sizeof(LLVMBasicBlockRef));
	opblocks[0] = l_bb_append_v(eval_fn, "b.op.start");

	/* jump from entry to first block */
	LLVMBuildBr(b, opblocks[0]);

	datum_param = LLVMGetParam(eval_fn, 0);
	v_derivatives = LLVMGetParam(eval_fn, 1);

	LLVMPositionBuilderAtEnd(b, opblocks[0]);

	for (i = 0; i < state->steps_len; i++)
	{
		ExprEvalStep *op;
		ExprEvalOp opcode;
		LLVMValueRef v_opval;

		op = &state->steps[i];
		opcode = ExecEvalStepOp(state, op);

		switch (opcode)
		{
		case EEOP_DONE:
		{
			LLVMValueRef seed;
			if (state->lambdaContainsMatrix)
			{
				LLVMValueRef params[1];
				LLVMTypeRef types[1];
				if ((ExecEvalStepOp(state, &(state->steps[state->steps_len - 2])) == 17 ||
					 ExecEvalStepOp(state, &(state->steps[state->steps_len - 2])) == 18) &&
					state->steps[state->steps_len - 2].d.func.finfo->fn_oid == 7801) // This checks, if expressionsstep is func-call and calls softmax as last call
				{
					types[0] = LLVMDoubleType();
					params[0] = l_float8_const(1.0);

					seed = build_EvalCFunc(b, mod, "createScalar",
											 (LLVMValueRef *)&params,
											 (LLVMTypeRef *)&types,
											 TypeDatum,
											 1);
				}
				else
				{
					types[0] = LLVMDoubleType();
					params[0] = l_float8_const(1.0);

					seed = build_EvalCFunc(b, mod, "createScalar",
										   (LLVMValueRef *)&params,
										   (LLVMTypeRef *)&types,
										   TypeDatum,
										   1);
				}
			}
			else
			{
				seed = l_float8_const(1.0);
			}

			funcInputPointer--;
			llvm_compile_simple_deriv_subtree(b, mod, state, state->steps_len - 2, seed, v_derivatives, intermediate_vals, &funcInputPointer);

			LLVMBuildRet(b, LLVMBuildZExtOrBitCast(b, registers[registerPointer - 1], TypeDatum, ""));
			break;
		}

		case EEOP_CONST:
		{
			registers[registerPointer++] = l_sizet_const(op->d.constval.value);
			break;
		}

		case EEOP_FUNCEXPR_STRICT:
		case EEOP_FUNCEXPR:
		{
			char *funcname;
			int numparams;

			LLVMValueRef opres;
			FunctionCallInfo fcinfo = op->d.func.fcinfo_data;
			switch (fcinfo->flinfo->fn_oid)
			{
			case 1726:
			case 216:
			{
				numparams = 2;
				opres = LLVMBuildBinOp(b, LLVMFMul,
									   l_as_float8(b, registers[registerPointer - 2]),
									   l_as_float8(b, registers[registerPointer - 1]),
									   "fmul");
				break;
			}

			case 1727:
			case 217:
			{
				numparams = 2;
				opres = LLVMBuildBinOp(b, LLVMFDiv,
									   l_as_float8(b, registers[registerPointer - 2]),
									   l_as_float8(b, registers[registerPointer - 1]),
									   "fdiv");
				break;
			}

			case 1724:
			case 218:
			{
				numparams = 2;
				opres = LLVMBuildBinOp(b, LLVMFAdd,
									   l_as_float8(b, registers[registerPointer - 2]),
									   l_as_float8(b, registers[registerPointer - 1]),
									   "fadd");
				break;
			}

			case 1725:
			case 219:
			{
				numparams = 2;
				opres = LLVMBuildBinOp(b, LLVMFSub,
									   l_as_float8(b, registers[registerPointer - 2]),
									   l_as_float8(b, registers[registerPointer - 1]),
									   "fsub");
				break;
			}

			case 220:
			{
				numparams = 1;
				opres = LLVMBuildBinOp(
					b,
					LLVMFMul,
					l_as_float8(b, registers[registerPointer - 1]),
					l_float8_const(-1.0),
					"");
				break;
			}

			case 1395:
			case 221:
			{
				LLVMTypeRef types[1];
				LLVMValueRef params[1];
				numparams = 1;
				types[0] = LLVMDoubleType();
				params[0] = l_as_float8(b, registers[registerPointer - 1]);

				opres = build_EvalCFunc(b, mod, "fabs", (LLVMValueRef *)&params,
										(LLVMTypeRef *)&types, types[0], 1);
				break;
			}

			case 1344:
			case 230:
			{
				LLVMTypeRef types[1];
				LLVMValueRef params[1];
				numparams = 1;
				types[0] = LLVMDoubleType();
				params[0] = l_as_float8(b, registers[registerPointer - 1]);

				opres = build_EvalCFunc(b, mod, "sqrt", (LLVMValueRef *)&params,
										(LLVMTypeRef *)&types, types[0], 1);
				break;
			}

			case 1600:
			{
				LLVMTypeRef types[1];
				LLVMValueRef params[1];
				numparams = 1;
				types[0] = LLVMDoubleType();
				params[0] = l_as_float8(b, registers[registerPointer - 1]);

				opres = build_EvalCFunc(b, mod, "asin", (LLVMValueRef *)&params,
										(LLVMTypeRef *)&types, types[0], 1);
				break;
			}

			case 1601:
			{
				LLVMTypeRef types[1];
				LLVMValueRef params[1];
				numparams = 1;
				types[0] = LLVMDoubleType();
				params[0] = l_as_float8(b, registers[registerPointer - 1]);

				opres = build_EvalCFunc(b, mod, "acos", (LLVMValueRef *)&params,
										(LLVMTypeRef *)&types, types[0], 1);
				break;
			}

			case 1602:
			{
				LLVMTypeRef types[1];
				LLVMValueRef params[1];
				numparams = 1;
				types[0] = LLVMDoubleType();
				params[0] = l_as_float8(b, registers[registerPointer - 1]);

				opres = build_EvalCFunc(b, mod, "atan", (LLVMValueRef *)&params,
										(LLVMTypeRef *)&types, types[0], 1);
				break;
			}

			case 1604:
			{
				LLVMTypeRef types[1];
				LLVMValueRef params[1];
				numparams = 1;
				types[0] = LLVMDoubleType();
				params[0] = l_as_float8(b, registers[registerPointer - 1]);

				opres = build_EvalCFunc(b, mod, "sin", (LLVMValueRef *)&params,
										(LLVMTypeRef *)&types, types[0], 1);
				break;
			}

			case 1605:
			{
				LLVMTypeRef types[1];
				LLVMValueRef params[1];
				numparams = 1;
				types[0] = LLVMDoubleType();
				params[0] = l_as_float8(b, registers[registerPointer - 1]);

				opres = build_EvalCFunc(b, mod, "cos", (LLVMValueRef *)&params,
										(LLVMTypeRef *)&types, types[0], 1);
				break;
			}

			case 1606:
			{
				LLVMTypeRef types[1];
				LLVMValueRef params[1];
				numparams = 1;
				types[0] = LLVMDoubleType();
				params[0] = l_as_float8(b, registers[registerPointer - 1]);

				opres = build_EvalCFunc(b, mod, "tan", (LLVMValueRef *)&params,
										(LLVMTypeRef *)&types, types[0], 1);
				break;
			}

			case 1607:
			{
				LLVMTypeRef types[1];
				LLVMValueRef params[1];
				numparams = 1;
				types[0] = LLVMDoubleType();
				params[0] = l_as_float8(b, registers[registerPointer - 1]);

				opres = build_EvalCFunc(b, mod, "cot", (LLVMValueRef *)&params,
										(LLVMTypeRef *)&types, types[0], 1);
				break;
			}

			case 1339:
			{
				LLVMTypeRef types[1];
				LLVMValueRef params[1];
				numparams = 1;
				types[0] = LLVMDoubleType();
				params[0] = l_as_float8(b, registers[registerPointer - 1]);

				opres = build_EvalCFunc(b, mod, "log10", (LLVMValueRef *)&params,
										(LLVMTypeRef *)&types, types[0], 1);
				break;
			}

			case 1341:
			{
				LLVMTypeRef types[1];
				LLVMValueRef params[1];
				numparams = 1;
				types[0] = LLVMDoubleType();
				params[0] = l_as_float8(b, registers[registerPointer - 1]);

				opres = build_EvalCFunc(b, mod, "log", (LLVMValueRef *)&params,
										(LLVMTypeRef *)&types, types[0], 1);
				break;
			}

			case 232:
			case 1346:
			{
				LLVMTypeRef types[2];
				LLVMValueRef params[2];
				numparams = 2;
				types[0] = LLVMDoubleType();
				types[1] = LLVMDoubleType();
				params[0] = l_as_float8(b, registers[registerPointer - 2]);
				params[1] = l_as_float8(b, registers[registerPointer - 1]);

				opres = build_EvalCFunc(b, mod, "pow", (LLVMValueRef *)&params, (LLVMTypeRef *)&types, types[0], 2);
				break;
			}

			case 1347:
			{
				LLVMTypeRef types[1];
				LLVMValueRef params[1];
				numparams = 1;
				types[0] = LLVMDoubleType();
				params[0] = l_as_float8(b, registers[registerPointer - 1]);

				opres = build_EvalCFunc(b, mod, "exp", (LLVMValueRef *)&params,
										(LLVMTypeRef *)&types, types[0], 1);
				break;
			}

			case 1603:
			{
				LLVMTypeRef types[2];
				LLVMValueRef params[2];
				numparams = 2;
				types[0] = LLVMDoubleType();
				types[1] = LLVMDoubleType();
				params[0] = l_as_float8(b, registers[registerPointer - 2]);
				params[1] = l_as_float8(b, registers[registerPointer - 1]);

				opres = build_EvalCFunc(b, mod, "atan2", (LLVMValueRef *)&params, (LLVMTypeRef *)&types, types[0], 2);
				break;
			}

			case 7801:
			{
				LLVMTypeRef types[2];
				LLVMValueRef params[2];
				numparams = 2;
				types[0] = TypeDatum;
				types[1] = TypeDatum;
				params[0] = registers[registerPointer - 2];
				params[1] = registers[registerPointer - 1];

				opres = build_EvalCFunc(b, mod, "softmax_cce_internal", (LLVMValueRef *)&params, (LLVMTypeRef *)&types, TypeDatum, 2);
				break;
			}

			case 7802:
			{
				LLVMTypeRef types[1];
				LLVMValueRef params[1];
				numparams = 1;
				types[0] = LLVMDoubleType();
				params[0] = LLVMBuildBinOp(b,
										   LLVMFMul,
										   l_float8_const(-1.0),
										   l_as_float8(b, registers[registerPointer - 1]),
										   "");

				opres = LLVMBuildBinOp(b,
									   LLVMFDiv,
									   l_as_float8(b, registers[registerPointer - 1]),
									   LLVMBuildBinOp(b,
													  LLVMFAdd,
													  l_float8_const(1.0),
													  build_EvalCFunc(b, mod, "exp", (LLVMValueRef *)&params, (LLVMTypeRef *)&types, types[0], 1),
													  ""),
									   "");
				break;
			}

			case 7803:
			{
				LLVMTypeRef types[1];
				LLVMValueRef params[1];
				numparams = 1;
				types[0] = LLVMDoubleType();
				params[0] = LLVMBuildBinOp(b,
										   LLVMFMul,
										   l_float8_const(-1.0),
										   l_as_float8(b, registers[registerPointer - 1]),
										   "");

				opres = LLVMBuildBinOp(b,
									   LLVMFDiv,
									   l_float8_const(1.0),
									   LLVMBuildBinOp(b,
													  LLVMFAdd,
													  l_float8_const(1.0),
													  build_EvalCFunc(b, mod, "exp", (LLVMValueRef *)&params, (LLVMTypeRef *)&types, types[0], 1),
													  ""),
									   "");
				break;
			}

			case 7804:
			{
				LLVMTypeRef types[1];
				LLVMValueRef params[1], eToX, eToMX;
				numparams = 1;
				types[0] = LLVMDoubleType();
				params[0] = LLVMBuildBinOp(b,
										   LLVMFMul,
										   l_float8_const(-1.0),
										   l_as_float8(b, registers[registerPointer - 1]),
										   "");
				eToMX = build_EvalCFunc(b, mod, "exp", (LLVMValueRef *)&params, (LLVMTypeRef *)&types, types[0], 1);

				params[0] = l_as_float8(b, registers[registerPointer - 1]);
				eToX = build_EvalCFunc(b, mod, "exp", (LLVMValueRef *)&params, (LLVMTypeRef *)&types, types[0], 1);

				opres = LLVMBuildBinOp(b,
									   LLVMFDiv,
									   LLVMBuildBinOp(b, LLVMFSub, eToX, eToMX, ""),
									   LLVMBuildBinOp(b, LLVMFAdd, eToX, eToMX, ""),
									   "");
				break;
			}

			case 7805:
			{
				LLVMTypeRef types[2];
				LLVMValueRef params[2];
				numparams = 1;
				types[0] = LLVMDoubleType();
				types[1] = LLVMDoubleType();
				params[0] = l_as_float8(b, registers[registerPointer - 1]);
				params[1] = l_float8_const(0.0);

				opres = build_EvalCFunc(b, mod, "fmax", (LLVMValueRef *)&params, (LLVMTypeRef *)&types, types[0], 2);
				break;
			}

			case 9000:
			{
				LLVMTypeRef types[4];
				LLVMValueRef params[4];
				numparams = 2;
				types[0] = TypeDatum;
				types[1] = TypeDatum;
				types[2] = TypeParamBool;
				types[3] = TypeParamBool;
				params[0] = registers[registerPointer - 2];
				params[1] = registers[registerPointer - 1];
				params[2] = l_pbool_const(false);
				params[3] = l_pbool_const(false);

				opres = build_EvalCFunc(b, mod, "matrix_mul_internal", (LLVMValueRef *)&params, (LLVMTypeRef *)&types, TypeDatum, 4);
				break;
			}

			case 9001:
			{
				LLVMTypeRef type;
				LLVMValueRef param;
				numparams = 1;
				type = TypeDatum;
				param = registers[registerPointer - 1];

				opres = build_EvalCFunc(b, mod, "silu_m_internal", (LLVMValueRef *)&param, (LLVMTypeRef *)&type, type, 1);
				break;
			}

			case 9002:
			{
				LLVMTypeRef type;
				LLVMValueRef param;
				numparams = 1;
				type = TypeDatum;
				param = registers[registerPointer - 1];

				opres = build_EvalCFunc(b, mod, "sigmoid_m_internal", (LLVMValueRef *)&param, (LLVMTypeRef *)&type, type, 1);
				break;
			}

			case 9003:
			{
				LLVMTypeRef type;
				LLVMValueRef param;
				numparams = 1;
				type = TypeDatum;
				param = registers[registerPointer - 1];

				opres = build_EvalCFunc(b, mod, "tanh_m_internal", (LLVMValueRef *)&param, (LLVMTypeRef *)&type, type, 1);
				break;
			}

			case 9004:
			{
				LLVMTypeRef type;
				LLVMValueRef param;
				numparams = 1;
				type = TypeDatum;
				param = registers[registerPointer - 1];

				opres = build_EvalCFunc(b, mod, "relu_m_internal", (LLVMValueRef *)&param, (LLVMTypeRef *)&type, type, 1);
				break;
			}

			case 9005:
			{
				LLVMTypeRef types[2];
				LLVMValueRef params[2];
				numparams = 2;
				types[0] = TypeDatum;
				types[1] = TypeDatum;
				params[0] = registers[registerPointer - 2];
				params[1] = registers[registerPointer - 1];

				opres = build_EvalCFunc(b, mod, "matrix_add_internal", (LLVMValueRef *)&params, (LLVMTypeRef *)&types, TypeDatum, 2);
				break;
			}

			default:
				ereport(ERROR,
						(errcode(ERRCODE_INTERNAL_ERROR),
						 errmsg("Function with Oid %i has no fast JIT derivation or is not differentiable.", fcinfo->flinfo->fn_oid)));
				break;
			}

			
			if(state->lambdaContainsMatrix) {
				if (numparams == 2)
				{
					intermediate_vals[funcInputPointer++] = registers[registerPointer - 2];
					intermediate_vals[funcInputPointer++] = registers[registerPointer - 1];
				}
				else
				{
					intermediate_vals[funcInputPointer++] = registers[registerPointer - 1];
				}
			} else {
				if (numparams == 2)
				{
					intermediate_vals[funcInputPointer++] = l_as_float8(b, registers[registerPointer - 2]);
					intermediate_vals[funcInputPointer++] = l_as_float8(b, registers[registerPointer - 1]);
				}
				else
				{
					intermediate_vals[funcInputPointer++] = l_as_float8(b, registers[registerPointer - 1]);
				}
			}

			registerPointer -= numparams;
			registers[registerPointer++] = opres;

			break;
		}

		case EEOP_PARAM_EXTERN:
		{
			LLVMValueRef addr = l_int32_const(op->d.param.paramid - 1);

			registers[registerPointer++] = LLVMBuildLoad(b, LLVMBuildGEP(b, datum_param, &addr, 1, ""), "");
			break;
		}

		case EEOP_FIELDSELECT:
		{
			LLVMValueRef addr = l_int32_const(op->d.fieldselect.fieldnum - 1);

			registers[registerPointer - 1] = LLVMBuildLoad(b, LLVMBuildGEP(b, registers[registerPointer - 1], &addr, 1, ""), "");
			break;
		}
		
		default:
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("Opcode %i cannot be JIT-compiled with llvm_compile_simple_expr", opcode)));
			break;
		}
	}

	LLVMDisposeBuilder(b);

	/*
	 * Don't immediately emit function, instead do so the first time the
	 * expression is actually evaluated. That allows to emit a lot of
	 * functions together, avoiding a lot of repeated llvm and memory
	 * remapping overhead.
	 */
	{
		CompiledExprState *cstate = palloc0(sizeof(CompiledExprState));

		cstate->context = context;
		cstate->funcname = funcname;

		state->derivefunc_simple_private = cstate;
	}

	llvm_leave_fatal_on_oom();

	INSTR_TIME_SET_CURRENT(endtime);
	INSTR_TIME_ACCUM_DIFF(context->base.instr.generation_counter,
						  endtime, starttime);

	return true;
}

static int
llvm_compile_simple_deriv_subtree(LLVMBuilderRef b,			    /* Builder containing the pre-built eval-func */
								  LLVMModuleRef mod,		    /* The module where the build function will be stored */
								  ExprState *state,			    /* State containing intermediate values for the derivations */
								  int fetchIndex,			    /* The step in the op_code sequence from where to start */
								  LLVMValueRef seed,		    /* seed for current subtree, not a pointer */
								  LLVMValueRef derivatives,     /* Datum(therefore pointer) array, containing all derivatives */
								  LLVMValueRef *funcVals,	    /* All FuncInputs, as we have no access to them due to L3/L4 */
								  int *intermediates_pointer)    /* the stack pointer, as to which values to take from funcVals */
{
	int resultFetchIndex = fetchIndex;
	switch (ExecEvalStepOp(state, &(state->steps[fetchIndex])))
	{
	case 59: /*EEOP_FIELDSELECT*/
	{
		LLVMValueRef v_derivative_p, /* The pointer into the derivatives array */
			v_tmpfieldnum,			 /* The fieldNum parameter of the Fieldselect */
			v_tmpderivative;		 /* The value behind the datum-pointer */

		int fieldNum = (state->indexArray[state->steps[fetchIndex - 1].d.param.paramid - 1]) +
					   (state->steps[fetchIndex].d.fieldselect.fieldnum - 1);

		v_tmpfieldnum = l_int32_const(fieldNum);
		v_derivative_p = LLVMBuildGEP(b,
									  derivatives,
									  &v_tmpfieldnum,
									  1, "");
		if (state->lambdaContainsMatrix)
		{
			LLVMValueRef matrix_add_params[2];
			LLVMTypeRef matrix_add_types[2];

			matrix_add_params[0] = LLVMBuildLoad(b, v_derivative_p, "");
			matrix_add_params[1] = seed;

			matrix_add_types[0] = TypeDatum;
			matrix_add_types[1] = TypeDatum;

			v_tmpderivative = build_EvalCFunc(b, mod, "matrix_add_inplace", (LLVMValueRef *)&matrix_add_params,
											  (LLVMTypeRef *)&matrix_add_types, TypeDatum, 2);
		}
		else
		{
			v_tmpderivative = LLVMBuildBinOp(b, LLVMFAdd,
											 l_as_float8(b, seed),
											 l_as_float8(b, LLVMBuildLoad(b, v_derivative_p, "")),
											 "");
		}
		LLVMBuildStore(b, v_tmpderivative, v_derivative_p);
		resultFetchIndex = fetchIndex - 2;
		break;
	}
	case 16: /*EEOP_CONST*/
	{
		resultFetchIndex = fetchIndex - 1; //can be skipped, d_const/d_x=0
		break;
	}
	case 17: /* EEOP_FUNCEXPR */
	case 18: /*EEOP_FUNCEXPR_STRICT*/
	{
		switch (state->steps[fetchIndex].d.func.finfo->fn_oid)
		{
		case 1726:
		case 216: /*float8 binary multiplication*/
		{
			LLVMValueRef x, y, newSeedX, newSeedY;
			int startingPointY, stepAfterX;

			y = l_as_float8(b, funcVals[(*intermediates_pointer)--]);
			x = l_as_float8(b, funcVals[(*intermediates_pointer)--]);

			newSeedX = LLVMBuildBinOp(b, LLVMFMul, y, l_as_float8(b, seed), "");
			newSeedY = LLVMBuildBinOp(b, LLVMFMul, x, l_as_float8(b, seed), "");

			startingPointY = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedY, derivatives, funcVals, intermediates_pointer);
			stepAfterX = llvm_compile_simple_deriv_subtree(b, mod, state, startingPointY, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepAfterX;
			break;
		}
		case 1727:
		case 217: /*float8 binary divison*/
		{
			LLVMValueRef x, y, newSeedX, newSeedY;
			int startingPointY, stepAfterX;

			y = l_as_float8(b, funcVals[(*intermediates_pointer)--]);
			x = l_as_float8(b, funcVals[(*intermediates_pointer)--]);

			newSeedX = LLVMBuildBinOp(b, LLVMFDiv, l_as_float8(b, seed), y, "");
			newSeedY = LLVMBuildBinOp(b,
									  LLVMFDiv,
									  (LLVMBuildBinOp(b,
													  LLVMFMul,
													  l_as_float8(b, seed),
													  (LLVMBuildBinOp(b, LLVMFMul, x, l_float8_const(-1.0), "")),
													  "")),
									  (LLVMBuildBinOp(b,
													  LLVMFMul,
													  y,
													  y,
													  "")),
									  "");

			startingPointY = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedY, derivatives, funcVals, intermediates_pointer);
			stepAfterX = llvm_compile_simple_deriv_subtree(b, mod, state, startingPointY, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepAfterX;
			break;
		}
		case 1724:
		case 218: /*float8 binary addition*/
		{
			int startingPointY, stepAfterX;
			(*intermediates_pointer)--; 				//add, subtract, um, etc.  put values onto stack, but dont retrieve them, we need to decrease the stack 
			(*intermediates_pointer)--;

			startingPointY = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, seed, derivatives, funcVals, intermediates_pointer);
			stepAfterX = llvm_compile_simple_deriv_subtree(b, mod, state, startingPointY, seed, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepAfterX;

			break;
		}
		case 1725:
		case 219: /*float8 binary subtraction*/
		{
			int startingPointY, stepAfterX;
			(*intermediates_pointer)--;
			(*intermediates_pointer)--;

			startingPointY = llvm_compile_simple_deriv_subtree(b,
															   mod,
															   state,
															   fetchIndex - 1,
															   LLVMBuildBinOp(
																   b,
																   LLVMFMul,
																   seed,
																   l_float8_const(-1.0),
																   ""),
															   derivatives,
															   funcVals,
															   intermediates_pointer);
			stepAfterX = llvm_compile_simple_deriv_subtree(b, mod, state, startingPointY, seed, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepAfterX;

			break;
		}
		case 232:
		case 1346: /*float8 binary pow x^y*/
		{
			LLVMValueRef x, y, newSeedX, newSeedY, params_pow_x[2], params_pow_y[2], params_log[1];
			LLVMTypeRef types[2];
			int startingPointY, stepAfterX;

			types[0] = LLVMDoubleType();
			types[1] = LLVMDoubleType();

			y = l_as_float8(b, funcVals[(*intermediates_pointer)--]);
			x = l_as_float8(b, funcVals[(*intermediates_pointer)--]);

			params_pow_x[0] = x;
			params_pow_x[1] = LLVMBuildBinOp(b, LLVMFSub, y, l_float8_const(1), "");

			params_pow_y[0] = x;
			params_pow_y[1] = y;

			params_log[0] = x;

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFMul,
									  LLVMBuildBinOp(b,
													 LLVMFMul,
													 y,
													 l_as_float8(b, seed),
													 ""),
									  build_EvalCFunc(b,
													  mod,
													  "pow",
													  (LLVMValueRef *)&params_pow_x,
													  (LLVMTypeRef *)&types,
													  types[0],
													  2),
									  "");
			newSeedY = LLVMBuildBinOp(b,
									  LLVMFMul,
									  LLVMBuildBinOp(b,
													 LLVMFMul,
													 build_EvalCFunc(b,
																	 mod,
																	 "pow",
																	 (LLVMValueRef *)&params_pow_y,
																	 (LLVMTypeRef *)&types,
																	 types[0],
																	 2),
													 build_EvalCFunc(b,
																	 mod,
																	 "log",
																	 (LLVMValueRef *)&params_log,
																	 (LLVMTypeRef *)&types,
																	 types[0],
																	 1),
													 ""),
									  l_as_float8(b, seed),
									  "");

			startingPointY = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedY, derivatives, funcVals, intermediates_pointer);
			stepAfterX = llvm_compile_simple_deriv_subtree(b, mod, state, startingPointY, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepAfterX;
			break;
		}
		case 230:
		case 1344: /*float8 unary sqrt*/
		{
			LLVMValueRef x, newSeedX, params_sqrt[1];
			LLVMTypeRef types[1];
			int stepAfterX;

			types[0] = LLVMDoubleType();

			x = l_as_float8(b, funcVals[(*intermediates_pointer)--]);

			params_sqrt[0] = x;

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFDiv,
									  l_as_float8(b, seed),
									  LLVMBuildBinOp(b,
													 LLVMFMul,
													 l_float8_const(2.0),
													 build_EvalCFunc(b,
																	 mod,
																	 "sqrt",
																	 (LLVMValueRef *)&params_sqrt,
																	 (LLVMTypeRef *)&types,
																	 types[0],
																	 1),
													 ""),
									  "");

			stepAfterX = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepAfterX;
			break;
		}
		case 221:
		case 1395: /*float8 unary abs*/
		{
			LLVMValueRef x, newSeedX, params_abs[1];
			LLVMTypeRef types[1];
			int stepAfterX;

			types[0] = LLVMDoubleType();

			x = l_as_float8(b, funcVals[(*intermediates_pointer)--]);

			params_abs[0] = x;

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFMul,
									  l_as_float8(b, seed),
									  LLVMBuildBinOp(b,
													 LLVMFDiv,
													 x,
													 build_EvalCFunc(b,
																	 mod,
																	 "fabs",
																	 (LLVMValueRef *)&params_abs,
																	 (LLVMTypeRef *)&types,
																	 types[0],
																	 1),
													 ""),
									  "");

			stepAfterX = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepAfterX;
			break;
		}
		case 1604: /*float8 unary sin*/
		{
			LLVMValueRef x, newSeedX, params_cos[1];
			LLVMTypeRef types[1];
			int stepAfterX;

			types[0] = LLVMDoubleType();

			x = l_as_float8(b, funcVals[(*intermediates_pointer)--]);

			params_cos[0] = x;

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFMul,
									  l_as_float8(b, seed),
									  build_EvalCFunc(b,
													  mod,
													  "cos",
													  (LLVMValueRef *)&params_cos,
													  (LLVMTypeRef *)&types,
													  types[0],
													  1),
									  "");

			stepAfterX = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepAfterX;
			break;
		}
		case 1605: /*float8 unary cos*/
		{
			LLVMValueRef x, newSeedX, params_sin[1];
			LLVMTypeRef types[1];
			int stepAfterX;

			types[0] = LLVMDoubleType();

			x = l_as_float8(b, funcVals[(*intermediates_pointer)--]);

			params_sin[0] = x;

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFMul,
									  LLVMBuildBinOp(
										  b,
										  LLVMFMul,
										  seed,
										  l_float8_const(-1.0),
										  ""),
									  build_EvalCFunc(b,
													  mod,
													  "sin",
													  (LLVMValueRef *)&params_sin,
													  (LLVMTypeRef *)&types,
													  types[0],
													  1),
									  "");

			stepAfterX = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepAfterX;
			break;
		}
		case 1347: /*float8 unary exp*/
		{
			LLVMValueRef x, newSeedX, params_exp[1];
			LLVMTypeRef types[1];
			int stepAfterX;

			types[0] = LLVMDoubleType();

			x = l_as_float8(b, funcVals[(*intermediates_pointer)--]);

			params_exp[0] = x;

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFMul,
									  l_as_float8(b, seed),
									  build_EvalCFunc(b,
													  mod,
													  "exp",
													  (LLVMValueRef *)&params_exp,
													  (LLVMTypeRef *)&types,
													  types[0],
													  1),
									  "");

			stepAfterX = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepAfterX;
			break;
		}
		case 220: /* float unary minus/negation */
		{
			LLVMValueRef newSeedX;
			int stepsAfterSubtree;
			(*intermediates_pointer)--;

			newSeedX = LLVMBuildBinOp(
				b,
				LLVMFMul,
				seed,
				l_float8_const(-1.0),
				"");

			stepsAfterSubtree = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 1600: /* float arcus sine */
		{
			LLVMValueRef newSeedX, x, params_sqrt_1[1], params_sqrt_2[1];
			LLVMTypeRef types[1];
			int stepsAfterSubtree;

			types[0] = LLVMDoubleType();

			x = l_as_float8(b, funcVals[(*intermediates_pointer)--]);

			params_sqrt_1[0] = LLVMBuildBinOp(b,
											  LLVMFSub,
											  l_float8_const(1.0),
											  x,
											  "");

			params_sqrt_2[0] = LLVMBuildBinOp(b,
											  LLVMFAdd,
											  l_float8_const(1.0),
											  x,
											  "");

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFDiv,
									  seed,
									  LLVMBuildBinOp(b,
													 LLVMFMul,
													 build_EvalCFunc(b,
																	 mod,
																	 "sqrt",
																	 (LLVMValueRef *)&params_sqrt_1,
																	 (LLVMTypeRef *)&types,
																	 types[0],
																	 1),
													 build_EvalCFunc(b,
																	 mod,
																	 "sqrt",
																	 (LLVMValueRef *)&params_sqrt_2,
																	 (LLVMTypeRef *)&types,
																	 types[0],
																	 1),
													 ""),
									  "");

			stepsAfterSubtree = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 1601: /* float arcus cosine */
		{
			LLVMValueRef newSeedX, x, params_sqrt_1[1], params_sqrt_2[1];
			LLVMTypeRef types[1];
			int stepsAfterSubtree;

			types[0] = LLVMDoubleType();

			x = l_as_float8(b, funcVals[(*intermediates_pointer)--]);

			params_sqrt_1[0] = LLVMBuildBinOp(b,
											  LLVMFSub,
											  l_float8_const(1.0),
											  x,
											  "");

			params_sqrt_2[0] = LLVMBuildBinOp(b,
											  LLVMFAdd,
											  l_float8_const(1.0),
											  x,
											  "");

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFDiv,
									  LLVMBuildBinOp(
										  b,
										  LLVMFMul,
										  seed,
										  l_float8_const(-1.0),
										  ""),
									  LLVMBuildBinOp(b,
													 LLVMFMul,
													 build_EvalCFunc(b,
																	 mod,
																	 "sqrt",
																	 (LLVMValueRef *)&params_sqrt_1,
																	 (LLVMTypeRef *)&types,
																	 types[0],
																	 1),
													 build_EvalCFunc(b,
																	 mod,
																	 "sqrt",
																	 (LLVMValueRef *)&params_sqrt_2,
																	 (LLVMTypeRef *)&types,
																	 types[0],
																	 1),
													 ""),
									  "");

			stepsAfterSubtree = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 1602: /* float unary arcus tangens */
		{
			LLVMValueRef newSeedX, x;
			int stepsAfterSubtree;

			x = l_as_float8(b, funcVals[(*intermediates_pointer)--]);

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFDiv,
									  seed,
									  LLVMBuildBinOp(b,
													 LLVMFAdd,
													 l_float8_const(1.0),
													 LLVMBuildBinOp(b,
																	LLVMFMul,
																	x,
																	x,
																	""),
													 ""),
									  "");

			stepsAfterSubtree = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 1603: /* float binary arcus tangens 2 (c: atan2)(normally defined as: atan2(y,x), but in this case, x is the first and y the second argument) */
		{
			LLVMValueRef x, y, newSeedX, newSeedY, tmp_newSeedY;
			int startingPointForY, stepIndexAfterX;

			y = l_as_float8(b, funcVals[(*intermediates_pointer)--]);
			x = l_as_float8(b, funcVals[(*intermediates_pointer)--]);

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFMul,
									  seed,
									  LLVMBuildBinOp(b,
													 LLVMFDiv,
													 y,
													 LLVMBuildBinOp(b,
																	LLVMFAdd,
																	LLVMBuildBinOp(b,
																				   LLVMFMul,
																				   x,
																				   x,
																				   ""),
																	LLVMBuildBinOp(b,
																				   LLVMFMul,
																				   y,
																				   y,
																				   ""),
																	""),
													 ""),
									  "");
			tmp_newSeedY = LLVMBuildBinOp(b,
										  LLVMFMul,
										  seed,
										  LLVMBuildBinOp(b,
														 LLVMFDiv,
														 x,
														 LLVMBuildBinOp(b,
																		LLVMFAdd,
																		LLVMBuildBinOp(b,
																					   LLVMFMul,
																					   x,
																					   x,
																					   ""),
																		LLVMBuildBinOp(b,
																					   LLVMFMul,
																					   y,
																					   y,
																					   ""),
																		""),
														 ""),
										  "");

			newSeedY = LLVMBuildBinOp(
				b,
				LLVMFMul,
				tmp_newSeedY,
				l_float8_const(-1.0),
				"");

			startingPointForY = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedY, derivatives, funcVals, intermediates_pointer);
			stepIndexAfterX = llvm_compile_simple_deriv_subtree(b, mod, state, startingPointForY, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepIndexAfterX;
			break;
		}
		case 1606: /* float tangens */
		{
			LLVMValueRef x, newSeedX, params_cos[1], tmp_cos;
			LLVMTypeRef types[1];
			int stepsAfterSubtree;

			x = l_as_float8(b, funcVals[(*intermediates_pointer)--]);
			params_cos[0] = x;
			types[0] = LLVMDoubleType();

			tmp_cos = build_EvalCFunc(b,
									  mod,
									  "cos",
									  (LLVMValueRef *)&params_cos,
									  (LLVMTypeRef *)&types,
									  types[0],
									  1);

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFDiv,
									  seed,
									  LLVMBuildBinOp(b,
													 LLVMFMul,
													 tmp_cos,
													 tmp_cos,
													 ""),
									  "");

			stepsAfterSubtree = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 1607: /* float co-tangens */
		{
			LLVMValueRef x, newSeedX, params_sin[1], tmp_sin;
			LLVMTypeRef types[1];
			int stepsAfterSubtree;

			x = l_as_float8(b, funcVals[(*intermediates_pointer)--]);
			params_sin[0] = x;
			types[0] = LLVMDoubleType();

			tmp_sin = build_EvalCFunc(b,
									  mod,
									  "sin",
									  (LLVMValueRef *)&params_sin,
									  (LLVMTypeRef *)&types,
									  types[0],
									  1);

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFDiv,
									  LLVMBuildBinOp(
										  b,
										  LLVMFMul,
										  seed,
										  l_float8_const(-1.0),
										  ""),
									  LLVMBuildBinOp(b,
													 LLVMFMul,
													 tmp_sin,
													 tmp_sin,
													 ""),
									  "");

			stepsAfterSubtree = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 1339: /* float log base 10 */
		{
			LLVMValueRef x, newSeedX, naturalLogOf10;
			int stepsAfterSubtree;

			x = l_as_float8(b, funcVals[(*intermediates_pointer)--]);
			naturalLogOf10 = l_float8_const(log(10));

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFDiv,
									  seed,
									  LLVMBuildBinOp(b,
													 LLVMFMul,
													 x,
													 naturalLogOf10,
													 ""),
									  "");

			stepsAfterSubtree = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 1341: /* float natural log */
		{
			LLVMValueRef x, newSeedX;
			int stepsAfterSubtree;

			x = l_as_float8(b, funcVals[(*intermediates_pointer)--]);

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFDiv,
									  seed,
									  x,
									  "");

			stepsAfterSubtree = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 7801: /* Softmax_CCE */
		{
			LLVMValueRef x, y, newSeedX, newSeedY, tmp, params_softmax[2], params_mul[4], param_scalar[1];
			LLVMTypeRef types_softmax[2], types_mul[4], type_scalar[1];
			int stepAfterX, stepAfterY;
			y = funcVals[(*intermediates_pointer)--];
			x = funcVals[(*intermediates_pointer)--];

			params_softmax[0] = x;       			
			params_softmax[1] = y;

			types_softmax[0] = TypeDatum;
			types_softmax[1] = TypeDatum;

			tmp = build_EvalCFunc(b, mod, "softmax_cce_derive", (LLVMValueRef *)&params_softmax, (LLVMTypeRef *)&types_softmax, TypeDatum, 2);

			types_mul[0] = TypeDatum;
			types_mul[1] = TypeDatum;
			types_mul[2] = TypeParamBool;
			types_mul[3] = TypeParamBool;

			params_mul[0] = seed;
			params_mul[1] = tmp;
			params_mul[2] = l_pbool_const(false);
			params_mul[3] = l_pbool_const(false);

			newSeedX = build_EvalCFunc(b, mod, "matrix_mul_internal",
									   (LLVMValueRef *)&params_mul,
									   (LLVMTypeRef *)&types_mul,
									   TypeDatum,
									   4);

			param_scalar[0] = l_float8_const(0.0);
			type_scalar[0] = LLVMDoubleType();

			newSeedY = build_EvalCFunc(b, mod, "createScalar", (LLVMValueRef *)&param_scalar, (LLVMTypeRef *)&type_scalar, TypeDatum, 1);

			stepAfterY = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedY, derivatives, funcVals, intermediates_pointer);
			stepAfterX = llvm_compile_simple_deriv_subtree(b, mod, state, stepAfterY, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepAfterX;
			break;
		}
		case 7802: /* float sigmoidial rectified linear unit */
		{
			LLVMValueRef x, newSeedX, tmp, params[1], eToMX;
			LLVMTypeRef types[1];
			int stepsAfterSubtree;

			x = l_as_float8(b, funcVals[(*intermediates_pointer)--]);

			types[0] = LLVMDoubleType();
			params[0] = LLVMBuildBinOp(b,
									   LLVMFMul,
									   l_float8_const(-1.0),
									   l_as_float8(b, x),
									   "");

			eToMX = build_EvalCFunc(b, mod, "exp", (LLVMValueRef *)&params, (LLVMTypeRef *)&types, types[0], 1);

			tmp = LLVMBuildBinOp(b,
								 LLVMFDiv,
								 LLVMBuildBinOp(b,
												LLVMFAdd,
												LLVMBuildBinOp(b,
															   LLVMFAdd,
															   l_float8_const(1.0),
															   eToMX,
															   ""),
												LLVMBuildBinOp(b,
															   LLVMFMul,
															   l_as_float8(b, x),
															   eToMX,
															   ""),
												""),
								 LLVMBuildBinOp(b,
												LLVMFMul,
												LLVMBuildBinOp(b,
															   LLVMFAdd,
															   l_float8_const(1.0),
															   eToMX,
															   ""),
												LLVMBuildBinOp(b,
															   LLVMFAdd,
															   l_float8_const(1.0),
															   eToMX,
															   ""),
												""),
								 "");

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFMul,
									  l_as_float8(b, seed),
									  l_as_float8(b, tmp),
									  "");

			stepsAfterSubtree = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 7803: /* float sigmoid */
		{
			LLVMValueRef x, newSeedX, tmp, params[1], sig_val;
			LLVMTypeRef types[1];
			int stepsAfterSubtree;

			x = l_as_float8(b, funcVals[(*intermediates_pointer)--]);

			types[0] = LLVMDoubleType();
			params[0] = LLVMBuildBinOp(b,
									   LLVMFMul,
									   l_float8_const(-1.0),
									   l_as_float8(b, x),
									   "");

			sig_val = LLVMBuildBinOp(b,
										 LLVMFDiv,
										 l_float8_const(1.0),
										 LLVMBuildBinOp(b,
														LLVMFAdd,
														l_float8_const(1.0),
														build_EvalCFunc(b, mod, "exp", (LLVMValueRef *)&params, (LLVMTypeRef *)&types, types[0], 1),
														""),
										 "");

			tmp = LLVMBuildBinOp(b,
								 LLVMFMul,
								 sig_val,
								 LLVMBuildBinOp(b,
												LLVMFSub,
												l_float8_const(1.0),
												sig_val,
												""),
								 "");

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFMul,
									  seed,
									  tmp,
									  "");

			stepsAfterSubtree = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 7804: /* float tangens hyperbolicus */
		{
			LLVMValueRef x, newSeedX, tmp, params[1], tanh_val;
			LLVMTypeRef types[1];
			int stepsAfterSubtree;

			x = l_as_float8(b, funcVals[(*intermediates_pointer)--]);

			types[0] = LLVMDoubleType();
			params[0] = l_as_float8(b, x);

			tanh_val = build_EvalCFunc(b, mod, "tanh", (LLVMValueRef *)&params, (LLVMTypeRef *)&types, types[0], 1);

			tmp = LLVMBuildBinOp(b,
								 LLVMFSub,
								 l_float8_const(1.0),
								 LLVMBuildBinOp(b,
												LLVMFMul,
												tanh_val,
												tanh_val,
												""),
								 "");

			newSeedX = LLVMBuildBinOp(b,
									  LLVMFMul,
									  seed,
									  tmp,
									  "");

			stepsAfterSubtree = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 7805: /* float rectified linear unit(relu) */
		{
			LLVMValueRef newSeedX, x, factor;
			LLVMTypeRef typeDouble;
			int stepsAfterSubtree;

			typeDouble = LLVMDoubleType();

			x = l_as_float8(b, funcVals[(*intermediates_pointer)--]);

			factor = LLVMBuildUIToFP(b, LLVMBuildFCmp(b, LLVMRealOGT, x, l_float8_const(0.0), ""), typeDouble, "");
			newSeedX = LLVMBuildBinOp(b,
									  LLVMFMul,
									  l_as_float8(b, seed),
									  l_as_float8(b, factor),
									  "");

			stepsAfterSubtree = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 9000: /* Matrix multiplication */
		{
			LLVMValueRef x, y, newSeedX, newSeedY, mat_mul_params_x[4], mat_mul_params_y[4];
			LLVMTypeRef mat_mul_types_x[4], mat_mul_types_y[4];
			int startingPointY, stepAfterX;

			y = funcVals[(*intermediates_pointer)--];
			x = funcVals[(*intermediates_pointer)--];

			mat_mul_params_x[0] = seed;
			mat_mul_params_x[1] = y;
			mat_mul_params_x[2] = l_pbool_const(false);
			mat_mul_params_x[3] = l_pbool_const(true);

			mat_mul_params_y[0] = x;
			mat_mul_params_y[1] = seed;
			mat_mul_params_y[2] = l_pbool_const(true);
			mat_mul_params_y[3] = l_pbool_const(false);

			mat_mul_types_x[0] = TypeDatum;
			mat_mul_types_x[1] = TypeDatum;
			mat_mul_types_x[2] = TypeParamBool;
			mat_mul_types_x[3] = TypeParamBool;

			mat_mul_types_y[0] = TypeDatum;
			mat_mul_types_y[1] = TypeDatum;
			mat_mul_types_y[2] = TypeParamBool;
			mat_mul_types_y[3] = TypeParamBool;

			newSeedX = build_EvalCFunc(b, mod, "matrix_mul_internal", (LLVMValueRef *)&mat_mul_params_x, (LLVMTypeRef *)&mat_mul_types_x, TypeDatum, 4);
			newSeedY = build_EvalCFunc(b, mod, "matrix_mul_internal", (LLVMValueRef *)&mat_mul_params_y, (LLVMTypeRef *)&mat_mul_types_y, TypeDatum, 4);

			startingPointY = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedY, derivatives, funcVals, intermediates_pointer);
			stepAfterX = llvm_compile_simple_deriv_subtree(b, mod, state, startingPointY, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepAfterX;
			break;
		}
		case 9001: /* matrix sigmoidial linear unit(silu) */
		{
			LLVMValueRef newSeedX, x, mat_elem_mul[2], mat_silu;
			LLVMTypeRef mat_elem_mul_types[2], mat_silu_type;
			int stepsAfterSubtree;

			x = funcVals[(*intermediates_pointer)--];

			mat_elem_mul_types[0] = TypeDatum;
			mat_elem_mul_types[1] = TypeDatum;
			mat_silu_type = TypeDatum;

			mat_elem_mul[0] = seed;
			mat_silu = x;
			mat_elem_mul[1] = build_EvalCFunc(b, mod, "silu_m_derive", (LLVMValueRef *)&mat_silu, (LLVMTypeRef *)&mat_silu_type, TypeDatum, 1);

			newSeedX = build_EvalCFunc(b, mod, "matrix_elem_mult", (LLVMValueRef *)&mat_elem_mul, (LLVMTypeRef *)&mat_elem_mul_types, TypeDatum, 2);

			stepsAfterSubtree = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 9002: /* matrix sigmoid */
		{
			LLVMValueRef newSeedX, x, mat_elem_mul[2], mat_silu;
			LLVMTypeRef mat_elem_mul_types[2], mat_silu_type;
			int stepsAfterSubtree;

			x = funcVals[(*intermediates_pointer)--];

			mat_elem_mul_types[0] = TypeDatum;
			mat_elem_mul_types[1] = TypeDatum;
			mat_silu_type = TypeDatum;

			mat_elem_mul[0] = seed;
			mat_silu = x;
			mat_elem_mul[1] = build_EvalCFunc(b, mod, "sigmoid_m_derive", (LLVMValueRef *)&mat_silu, (LLVMTypeRef *)&mat_silu_type, TypeDatum, 1);

			newSeedX = build_EvalCFunc(b, mod, "matrix_elem_mult", (LLVMValueRef *)&mat_elem_mul, (LLVMTypeRef *)&mat_elem_mul_types, TypeDatum, 2);

			stepsAfterSubtree = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 9003: /* matrix tanh */
		{
			LLVMValueRef newSeedX, x, mat_elem_mul[2], mat_silu;
			LLVMTypeRef mat_elem_mul_types[2], mat_silu_type;
			int stepsAfterSubtree;

			x = funcVals[(*intermediates_pointer)--];

			mat_elem_mul_types[0] = TypeDatum;
			mat_elem_mul_types[1] = TypeDatum;
			mat_silu_type = TypeDatum;

			mat_elem_mul[0] = seed;
			mat_silu = x;
			mat_elem_mul[1] = build_EvalCFunc(b, mod, "tanh_m_derive", (LLVMValueRef *)&mat_silu, (LLVMTypeRef *)&mat_silu_type, TypeDatum, 1);

			newSeedX = build_EvalCFunc(b, mod, "matrix_elem_mult", (LLVMValueRef *)&mat_elem_mul, (LLVMTypeRef *)&mat_elem_mul_types, TypeDatum, 2);

			stepsAfterSubtree = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 9004: /* matrix rectified linear unit(relu) */
		{
			LLVMValueRef newSeedX, x, mat_elem_mul[2], mat_silu;
			LLVMTypeRef mat_elem_mul_types[2], mat_silu_type;
			int stepsAfterSubtree;

			x = funcVals[(*intermediates_pointer)--];

			mat_elem_mul_types[0] = TypeDatum;
			mat_elem_mul_types[1] = TypeDatum;
			mat_silu_type = TypeDatum;

			mat_elem_mul[0] = seed;
			mat_silu = x;
			mat_elem_mul[1] = build_EvalCFunc(b, mod, "relu_m_derive", (LLVMValueRef *)&mat_silu, (LLVMTypeRef *)&mat_silu_type, TypeDatum, 1);

			newSeedX = build_EvalCFunc(b, mod, "matrix_elem_mult", (LLVMValueRef *)&mat_elem_mul, (LLVMTypeRef *)&mat_elem_mul_types, TypeDatum, 2);

			stepsAfterSubtree = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, newSeedX, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepsAfterSubtree;
			break;
		}
		case 9005: /* Matrix addition */
		{
			int startingPointY, stepAfterX;

			(*intermediates_pointer)--;
			(*intermediates_pointer)--;

			startingPointY = llvm_compile_simple_deriv_subtree(b, mod, state, fetchIndex - 1, seed, derivatives, funcVals, intermediates_pointer);
			stepAfterX = llvm_compile_simple_deriv_subtree(b, mod, state, startingPointY, seed, derivatives, funcVals, intermediates_pointer);
			resultFetchIndex = stepAfterX;
			break;
		}
		default:
		{
			ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED), errmsg("Derive(L3/L4): current operator not supported, aborting...")));
			break;
		}
		}
		break;
	}
	default:
	{
		ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED), errmsg("Derive(L3/L4): current step-opcode not recognized, aborting...")));
		break;
	}
	}
	return resultFetchIndex;
}