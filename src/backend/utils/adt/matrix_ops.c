/*-------------------------------------------------------------------------
 *
 * matrix_ops.c
 *	  This file contains some support routines 
 *    required for matrix arithmetic functions.
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
#include "utils/lsyscache.h"

#define MAT_2D(X,Y,ROW_SIZE) (X*ROW_SIZE+Y)

/*
 * Calculate the matrix product of Matrix(ArrayType) A and B
 */
Datum matrix_mul(PG_FUNCTION_ARGS)
{
    printf("matrix mul external beginning\n");
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
    float8 *retContent;

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

    retContent = palloc0(sizeof(float8) * dima * dimc);
    double *arrayContent1f = (double *)palloc(sizeof(double) * dima * dimb);
    double *arrayContent2Tf = (double *)palloc(sizeof(double) * dimb * dimc);
    
    length1 = ArrayGetNItems(ARR_NDIM(a1), ARR_DIMS(a1));
    Datum *ps1 = (Datum *)ARR_DATA_PTR(a1);

    length2 = ArrayGetNItems(ARR_NDIM(a2), ARR_DIMS(a2));
    Datum *ps2 = (Datum *)ARR_DATA_PTR(a2);

    Datum *ps;
#pragma omp parallel shared(arrayContent1f, arrayContent2Tf, retContent)
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
            //printf("Mat A at pos(%d): %lf\n", pos, arrayContent1f[pos]);
        }
#pragma omp for nowait
        for (int pos = 0; pos < length2; pos++)
        {
            arrayContent2Tf[pos] = (double)DatumGetFloat8(ps2[pos]);
            //printf("Mat B at pos(%d): %lf\n", pos, arrayContent2Tf[pos]);
        }
#pragma omp for nowait
        for (int pos = 0; pos < dima*dimc; pos++)
        {
            retContent[pos] = 0.0;
            //printf("Mat B at pos(%d): %lf\n", pos, arrayContent2Tf[pos]);
        }

#pragma omp barrier
        int i, j, k;
#pragma omp for private(i,j,k) 
        for(i = 0; i < dima; i++) {
            for(j = 0; j < dimb; j++) {
                for(k = 0; k < dimc; k++) {
                    retContent[MAT_2D(i,k,dimc)] += arrayContent1f[MAT_2D(i,j,dimb)] * arrayContent2Tf[MAT_2D(j,k,dimc)];
                }
            }
        }

#pragma omp barrier
#pragma omp for
        for (int i = 0; i < dima * dimc; i++)
        {
            ps[i] = Float8GetDatumFast(retContent[i]);
            //printf("Mat C at pos(%d): %lf\n", i, retContent[i]);
        }
    }

    PG_RETURN_ARRAYTYPE_P(ret);
    // int32 transposeA = PG_GETARG_INT32(2) % 2 == 0; // true for 0 and 2
    // int32 transposeB = PG_GETARG_INT32(2) / 2 == 0; // true for 0 and 1 => (0, true, true), (1, false, true), (2, true, false), (3, false, false)
    // return matrix_mul_internal(PG_GETARG_DATUM(0), PG_GETARG_DATUM(1), transposeA, transposeB);
}

/*
 * Fast implementation, forgoing all dimension, size and type checks, beware of undefined behaviour
 * NOTICE: Expects Base ArrayType passed as Pointer(cast to Datum), with elementtype being double/float8
 *         Always returns a 2D-Matrix, even if a dimension is 1(result being a vector)
 */
Datum matrix_mul_internal(Datum MatA, Datum MatB, const bool transposeA, const bool transposeB) {
    printf("matrix mul internal beginning\n");
    // The formal PostgreSQL array objects:
    ArrayType *a1, *a2, *ret;

    // The array contents of result matrix:
    float8 *retContent;

    // The size of each array:
    int length1, length2;

    // Extract the PostgreSQL arrays from the parameters passed to this function call.
    a1 = DatumGetArrayTypeP(MatA);
    a2 = DatumGetArrayTypeP(MatB);

    int *dim1 = ARR_DIMS(a1);
    int *dim2 = ARR_DIMS(a2);
    int lbs[2] = {1, 1};

    int dima, dimb, dimc;
    dima = 1;
    dimb = 1;
    dimc = 1;

    //printf("The dimensions(before transpose) are: \nA(%dx%d)\nB(%dx%d)\n", dim1[0], dim1[1], dim2[0], dim2[1]);

    if(ARR_NDIM(a1) == 2) {
        if(transposeA) 
        {
            dima = dim1[1];
            dimb = dim1[0];
        } else 
        {
            dima = dim1[0];
            dimb = dim1[1];
        }
    } else {
        if (transposeA)
        {
            dimb = dim1[0];
        }
        else
        {
            dima = dim1[0];
        }
    }

    if (ARR_NDIM(a2) == 2)
    {
        if (transposeB)
        {
            dimc = dim2[0];
        }
        else
        {
            dimc = dim2[1];
        }
    }
    else
    {
        if (transposeB)
        {
            dimc = dim2[0];
        }
    }

    //printf("The dimensions(with transpose) are: \nA(%dx%d)\nB(%dx%d)\nC(%dx%d)\n", dima, dimb, dimb, dimc, dima, dimc);

    int dims[2];
    dims[0] = dima;
    dims[1] = dimc;

    retContent = palloc0(sizeof(float8) * dima * dimc);
    length1 = ArrayGetNItems(ARR_NDIM(a1), ARR_DIMS(a1));
    length2 = ArrayGetNItems(ARR_NDIM(a2), ARR_DIMS(a2));

    double *arrayContent1f = (double *)palloc(sizeof(double) * length1);
    double *arrayContent2Tf = (double *)palloc(sizeof(double) * length2);

    
    Datum *ps1 = (Datum *)ARR_DATA_PTR(a1);
    Datum *ps2 = (Datum *)ARR_DATA_PTR(a2);

    Datum *ps;
#pragma omp parallel shared(arrayContent1f, arrayContent2Tf, retContent)
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
            arrayContent1f[pos] = DatumGetFloat8(ps1[pos]);
            //printf("Mat A at pos(%d): %lf\n", pos, arrayContent1f[pos]);
        }
#pragma omp for nowait
        for (int pos = 0; pos < length2; pos++)
        {
            arrayContent2Tf[pos] = DatumGetFloat8(ps2[pos]);
            //printf("Mat B at pos(%d): %lf\n", pos, arrayContent2Tf[pos]);
        }
#pragma omp for nowait
        for (int pos = 0; pos < dima * dimc; pos++)
        {
            retContent[pos] = 0.0;
        }

#pragma omp barrier
        int i, j, k;
#pragma omp for private(i, j, k)
        for (i = 0; i < dima; i++)
        {
            for (j = 0; j < dimb; j++)
            {
                for (k = 0; k < dimc; k++)
                {
                    retContent[MAT_2D(i, k, dimc)] 
                                            += arrayContent1f[((transposeA) ? (MAT_2D(j, i, dima)) : (MAT_2D(i, j, dimb)))] 
                                            *  arrayContent2Tf[((transposeB) ? (MAT_2D(k, j, dimb)) : (MAT_2D(j, k, dimc)))];
                }
            }
        }

#pragma omp barrier
#pragma omp for
        for (int i = 0; i < dima * dimc; i++)
        {
            ps[i] = Float8GetDatumFast(retContent[i]);
            //printf("Mat C at pos(%d): %lf\n", i, retContent[i]);
        }
    }
    printf("matrix mul internal end\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/*
 * Adds Two matricies elementwise together inlace (a += b)
 *  
 * Fast implementation, forgoing all dimension, size and type checks, beware of undefined behaviour
 * NOTICE: Expects Base ArrayType passed as Pointer(cast to Datum), with elementtype being double/float8
 *         Always returns a 2D-Matrix, even if a dimension is 1(result being a vector)
 */
Datum matrix_add_internal(Datum MatA, Datum MatB)
{
    printf("matrix add beginning\n");
    // The formal PostgreSQL array objects:
    ArrayType *a1, *a2;

    // The size of each array:
    int length1;

    // Extract the PostgreSQL arrays from the parameters passed to this function call.
    a1 = DatumGetArrayTypeP(MatA);
    a2 = DatumGetArrayTypeP(MatB);

    length1 = ArrayGetNItems(ARR_NDIM(a1), ARR_DIMS(a1));

    Datum *ps1 = (Datum *)ARR_DATA_PTR(a1);
    Datum *ps2 = (Datum *)ARR_DATA_PTR(a2);
#pragma omp parallel for
    for (int pos = 0; pos < length1; pos++)
    {
        float8 ret = DatumGetFloat8(ps1[pos]);
        ret += DatumGetFloat8(ps2[pos]);
        ps1[pos] = Float8GetDatum(ret);
    }

    PG_RETURN_ARRAYTYPE_P(a1);
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