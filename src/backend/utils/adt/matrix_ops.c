/*-------------------------------------------------------------------------
 *
 * matrix_ops.c
 *	  This file contains some support routines 
 *    required for matrix arithmetic functions.
 *
 * Portions Copyright (c) 1996-2018, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 *
 * IDENTIFICATION
 *	  src/backend/utils/adt/matrix_ops.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"

#include "fmgr.h"

#include "catalog/pg_type.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/memutils.h"

#include <cblas.h>

/*
 * Calculate the matrix product of Matrix(ArrayType) A and B
 */
Datum matrix_mul(PG_FUNCTION_ARGS)
{
    // The formal PostgreSQL array objects:
    ArrayType *a1, *a2, *ret;

    // The array element types:
    Oid type1, type2;

    // The array element type widths :
    int16 typeWidth1, typeWidth2;

    // The array element type "is passed by value" flags (not used, should always be true):
    bool typeByValue1, typeByValue2;

    // The array element type alignment codes (not used):
    char typeAlignmentCode1, typeAlignmentCode2;

    // The array contents, as PostgreSQL "datum" objects:
    Datum *retContent;

    // The size of each array:
    int length1, length2;

    if (PG_ARGISNULL(0))
    {
        ereport(ERROR, (errmsg("Null arrays not accepted")));
    }

    // Extract the PostgreSQL arrays from the parameters passed to this function call.
    a1 = PG_GETARG_ARRAYTYPE_P(0);
    a2 = PG_GETARG_ARRAYTYPE_P(1);

    if (ARR_NDIM(a1) == 0 || ARR_NDIM(a2) == 0)
    {
        PG_RETURN_NULL();
    }

    if (array_contains_nulls(a1))
    {
        ereport(ERROR, (errmsg("Array left contains null elements")));
    }

    if (array_contains_nulls(a2))
    {
        ereport(ERROR, (errmsg("Array right contains null elements")));
    }

    if (ARR_NDIM(a2) != 1 && ARR_NDIM(a2) != 2)
    {
        ereport(ERROR, (errmsg("Array right dimensions wrong!")));
    }

    if (ARR_NDIM(a1) != 1 && ARR_NDIM(a1) != 2)
    {
        ereport(ERROR, (errmsg("Array left dimensions wrong!")));
    }

    int *dim1 = ARR_DIMS(a1);
    int *dim2 = ARR_DIMS(a2);
    int lbs[2];

    if (ARR_NDIM(a1) == 2 && ARR_NDIM(a2) == 2 && dim1[1] != dim2[0])
    {
        ereport(ERROR, (errmsg("Matrix dimensions does not match!")));
    }
    if (ARR_NDIM(a1) == 1 && ARR_NDIM(a2) == 2 && dim1[0] != dim2[0])
    {
        ereport(ERROR, (errmsg("Matrix dimensions does not match!")));
    }

    for (int i = 0; i < 2; i++)
    {
        lbs[i] = 1;
    }

    // Determine the array element types.
    type1 = ARR_ELEMTYPE(a1);
    get_typlenbyvalalign(type1, &typeWidth1, &typeByValue1, &typeAlignmentCode1);
    type2 = ARR_ELEMTYPE(a2);
    get_typlenbyvalalign(type2, &typeWidth2, &typeByValue2, &typeAlignmentCode2);

    if (type1 != FLOAT8OID)
    {
        ereport(ERROR, (errmsg("Arrays must be SMALLINT, INTEGER, BIGINT, REAL, or DOUBLE PRECISION values")));
    }
    if (type2 != FLOAT8OID)
    {
        ereport(ERROR, (errmsg("Arrays must be SMALLINT, INTEGER, BIGINT, REAL, or DOUBLE PRECISION values")));
    }

    int dima = 1;
    if (ARR_NDIM(a1) == 2)
    {
        dima = dim1[0];
    }

    int dimb = dim2[0];

    int dimc = 1;
    if (ARR_NDIM(a2) == 2)
    {
        dimc = dim2[1];
    }

    int dims[2];
    dims[0] = dima;
    dims[1] = dimc;

    retContent = (Datum *)palloc0(sizeof(Datum) * dima * dimc);
    double *arrayContent1f = (double *)palloc(sizeof(double) * dima * dimb);
    double *arrayContent2Tf = (double *)palloc(sizeof(double) * dimb * dimc);
    
    length1 = ArrayGetNItems(ARR_NDIM(a1), ARR_DIMS(a1));
    Datum *ps1 = (Datum *)ARR_DATA_PTR(a1);

    length2 = ArrayGetNItems(ARR_NDIM(a2), ARR_DIMS(a2));
    Datum *ps2 = (Datum *)ARR_DATA_PTR(a2);

    Datum *ps;
#pragma omp parallel
    {
#pragma omp single nowait
        {
            //Already prepare result array!
            ret = initResult(2, dims, lbs);
            ps = (Datum *)ARR_DATA_PTR(ret);
        }

#pragma omp for nowait
        for (int pos = 0; pos < length1; pos++)
        {
            arrayContent1f[pos] = (double)DatumGetFloat8(ps1[pos]);
        }
#pragma omp for nowait
        for (int pos = 0; pos < length2; pos++)
        {
            arrayContent2Tf[pos] = (double)DatumGetFloat8(ps2[pos]);
        }

#pragma omp barrier

#pragma omp single
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    dima, dimc, dimb, 1, arrayContent1f, dimb, arrayContent2Tf, dimc, 1, (double *)retContent, dimc);

#pragma omp for
        for (int i = 0; i < dima * dimc; i++)
        {
            ps[i] = (retContent[i]);
        }
    }

    PG_RETURN_ARRAYTYPE_P(ret);
}

/*
 * Create a new array, based on the given dimensions, including 
 * the data_array afterwards
 */
ArrayType *initResult(int ndims, int *dims, int *lbs)
{
    int nelems = ArrayGetNItems(ndims, dims);
    int32 nbytes = nelems * ALIGNOF_DOUBLE;
    nbytes += ARR_OVERHEAD_NONULLS(ndims);
    ArrayType *ret = (ArrayType *)palloc0(nbytes);
    SET_VARSIZE(ret, nbytes);
    ret->ndim = ndims;
    ret->dataoffset = 0;
    ret->elemtype = FLOAT8OID;
    memcpy(ARR_DIMS(ret), dims, ndims * sizeof(int));
    memcpy(ARR_LBOUND(ret), lbs, ndims * sizeof(int));
    return ret;
}