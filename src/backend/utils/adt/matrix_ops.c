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

#include <math.h>

#define MAT_2D(X,Y,ROW_SIZE) (X*ROW_SIZE+Y)

/*
 * Calculate the matrix product of Matrix(ArrayType) A and B
 */
Datum matrix_mul(PG_FUNCTION_ARGS)
{
    printf("begin of matrix_mul_external\n");
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
#pragma omp for nowait
        for (int pos = 0; pos < dima*dimc; pos++)
        {
            retContent[pos] = 0.0;
        }
#pragma omp barrier
        int i, j, k;
#pragma omp for
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
        }
    }
    printf("end of matrix_mul_external\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/*
 * Internal implementation, uses element-wise multiplication, if one of the inputs is of length 1
 * NOTICE: Expects Base ArrayType passed as Pointer(cast to Datum), with elementtype being double/float8
 *         Always returns a 2D-Matrix, even if a dimension is 1(result being a vector)
 */
Datum matrix_mul_internal(Datum MatA, Datum MatB, const bool transposeA, const bool transposeB) 
{
    printf("begin of matrix_mul_internal\n");
    // The formal PostgreSQL array objects:
    ArrayType *a1, *a2, *ret;

    // The array contents of result matrix:
    float8 *retContent;

    // The size of each array:
    int length1, length2;

    if(DatumGetPointer(MatA) == NULL || DatumGetPointer(MatB) == NULL) {
        ereport(ERROR, (errmsg("Matrix Multiplication Internal: Null pointer passed as Matrix!")));
    }

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

    length1 = ArrayGetNItems(ARR_NDIM(a1), ARR_DIMS(a1));
    length2 = ArrayGetNItems(ARR_NDIM(a2), ARR_DIMS(a2));

    // if one or both are scalars, use element-wise multiplication
    if (length1 == 1)
    {
        if (transposeB)
        {
            ret = matrix_transpose_internal(MatB);
        }
        else
        {
            ret = copyArray(MatB);
        }
        float8 *data = (float8 *)ARR_DATA_PTR(ret);
        for (int i = 0; i < length2; i++)
        {
            data[i] *= DatumGetFloat8(((Datum *)ARR_DATA_PTR(a1))[0]);
        }
        PG_RETURN_ARRAYTYPE_P(ret);
    }
    if (length2 == 1)
    {
        if (transposeA)
        {
            ret = matrix_transpose_internal(MatA);
        }
        else
        {
            ret = copyArray(MatA);
        }
        float8 *data = (float8 *)ARR_DATA_PTR(ret);
        for (int i = 0; i < length1; i++)
        {
            data[i] *= DatumGetFloat8(((Datum *)ARR_DATA_PTR(a2))[0]);
        }
        PG_RETURN_ARRAYTYPE_P(ret);
    }

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
            if(dimb != dim2[1]) {
                ereport(ERROR, (errmsg("Matrix Multiplication Internal: Mismatched array dimensions!")));
            }
            dimc = dim2[0];
        }
        else
        {
            if (dimb != dim2[0])
            {
                ereport(ERROR, (errmsg("Matrix Multiplication Internal: Mismatched array dimensions!")));
            }
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

    int dims[2];
    dims[0] = dima;
    dims[1] = dimc;

    retContent = palloc0(sizeof(float8) * dima * dimc);

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
        }
#pragma omp for nowait
        for (int pos = 0; pos < length2; pos++)
        {
            arrayContent2Tf[pos] = DatumGetFloat8(ps2[pos]);
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
        }
    }
    printf("end of matrix_mul_internal\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/*
 * Adds Two matricies elementwise together inlace (a += b)
 *  
 * Fast implementation, forgoing all dimension, size and type checks, beware of undefined behaviour
 * NOTICE: Expects Base ArrayType passed as Pointer(cast to Datum), with elementtype being double/float8
 *         If MatA is a NullPointer, a copy of MatB will be returned 
 */
Datum matrix_add_inplace(Datum MatA, Datum MatB)
{
    printf("begin of matrix_add_inplace\n");
    // The formal PostgreSQL array objects:
    ArrayType *a1, *a2;

    // The size of each array:
    int length1, length2;
    a1 = DatumGetArrayTypeP(MatA);
    a2 = DatumGetArrayTypeP(MatB);

    length1 = ArrayGetNItems(ARR_NDIM(a1), ARR_DIMS(a1));
    length2 = ArrayGetNItems(ARR_NDIM(a2), ARR_DIMS(a2));

    if (length1 == 1)
    {
        a1 = copyArray(MatB);

        Datum *data = (Datum *)ARR_DATA_PTR(a2);
        Datum elem = ((Datum *)ARR_DATA_PTR(a1))[0];
        for(int i = 0; i < length2; i++) {
            data[i] = Float8GetDatum(DatumGetFloat8(data[i]) + DatumGetFloat8(elem));
        }

        printf("end of matrix_add_inplace\n");
        PG_RETURN_ARRAYTYPE_P(a1);
    }
    if (length2 == 1)
    {
        a1 = copyArray(MatA);

        Datum *data = (Datum *)ARR_DATA_PTR(a1);
        Datum elem = ((Datum *)ARR_DATA_PTR(a2))[0];
        for (int i = 0; i < length2; i++)
        {
            data[i] = Float8GetDatum(DatumGetFloat8(data[i]) + DatumGetFloat8(elem));
        }

        printf("end of matrix_add_inplace\n");
        PG_RETURN_ARRAYTYPE_P(a1);
    }

    Datum *ps1 = (Datum *)ARR_DATA_PTR(a1);
    Datum *ps2 = (Datum *)ARR_DATA_PTR(a2);
#pragma omp parallel for
    for (int pos = 0; pos < length1; pos++)
    {
        float8 ret = DatumGetFloat8(ps1[pos]);
        ret += DatumGetFloat8(ps2[pos]);
        ps1[pos] = Float8GetDatum(ret);
    }
    printf("end of matrix_add_inplace\n");
    PG_RETURN_ARRAYTYPE_P(a1);
}

/*
 * Multiply two matrices element-wise and return product as new stand-alone matrix(to avert pointer issues)
 */
Datum matrix_elem_mult(Datum matA, Datum matB) {
    printf("begin of matrix_elem_mult\n");
    ArrayType *a, *b, *ret;
    int ndims, length1, length2;
    int *dims;

    if (DatumGetPointer(matA) == NULL || DatumGetPointer(matB) == NULL) {
        ereport(ERROR, (errmsg("Matrix element-wise Multiplication: MatrixPointers are null")));
    }

    a = DatumGetArrayTypeP(matA);
    b = DatumGetArrayTypeP(matB);

    length1 = ArrayGetNItems(ARR_NDIM(a), ARR_DIMS(a));
    length2 = ArrayGetNItems(ARR_NDIM(b), ARR_DIMS(b));

    if(length1 == 1) {
        ret = copyArray(matB);

        Datum *data = (Datum *)ARR_DATA_PTR(ret);
        const Datum elem = ((Datum *)ARR_DATA_PTR(a))[0];

        for(int i = 0; i < length2; i++) {
            data[i] = Float8GetDatum(DatumGetFloat8(elem) * DatumGetFloat8(data[i]));
        }
        printf("end of matrix_elem_mult\n");
        PG_RETURN_ARRAYTYPE_P(ret);
    } else if(length2 == 1) {
        ret = copyArray(matA);

        Datum *data = (Datum *)ARR_DATA_PTR(ret);
        const Datum elem = ((Datum *)ARR_DATA_PTR(b))[0];

        for (int i = 0; i < length1; i++)
        {
            data[i] = Float8GetDatum(DatumGetFloat8(elem) * DatumGetFloat8(data[i]));
        }
        printf("end of matrix_elem_mult\n");
        PG_RETURN_ARRAYTYPE_P(ret);
    }

    ndims = ARR_NDIM(a);
    dims = ARR_DIMS(a);
    if (ndims != ARR_NDIM(b))
    {
        ereport(ERROR, (errmsg("Matrix element-wise Multiplication: Matrices are mismatched!")));
    }
    for(int i = 0; i < ndims; i++) {
        if(dims[i] != ARR_DIMS(b)[i]) {
            ereport(ERROR, (errmsg("Matrix element-wise Multiplication: Matrices are mismatched!")));
        }
    }

    ret = initResult(ndims, dims, ARR_LBOUND(a));

    Datum *ret_data = (Datum *)ARR_DATA_PTR(ret);
    Datum *a_data = (Datum *)ARR_DATA_PTR(a);
    Datum *b_data = (Datum *)ARR_DATA_PTR(b);

#pragma omp parrallel for
    for (int i = 0; i < ArrayGetNItems(ARR_NDIM(a), ARR_DIMS(a)); i++) {
        ret_data[i] = Float8GetDatum(DatumGetFloat8(a_data[i]) * DatumGetFloat8(b_data[i]));
    }
    printf("end of matrix_elem_mult\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/*
 * Transpose a matrix and return transposed copy
 */
Datum matrix_transpose_internal(Datum MatA)
{
    printf("begin of matrix_transpose_internal\n");
    // The formal PostgreSQL array objects:
    ArrayType *ret, *org;
    org = DatumGetArrayTypeP(MatA);

    int dims[2], lbs[2];

    if (ARR_NDIM(org) == 1)
    {
        dims[0] = 1;
        dims[1] = ARR_DIMS(org)[0];
        lbs[0] = 1;
        lbs[1] = ARR_LBOUND(org)[0];
        ret = initResult(2, dims, lbs);
        memcpy(ARR_DATA_PTR(ret), ARR_DATA_PTR(org), ArrayGetNItems(2, dims) * ALIGNOF_DOUBLE);
        PG_RETURN_ARRAYTYPE_P(ret);
    }
    dims[0] = ARR_DIMS(org)[1];
    dims[1] = ARR_DIMS(org)[0];
    lbs[0] = ARR_LBOUND(org)[1];
    lbs[1] = ARR_LBOUND(org)[0];

    ret = initResult(2, dims, lbs);
    Datum *A = (Datum *)ARR_DATA_PTR(org);
    Datum *B = (Datum *)ARR_DATA_PTR(ret);

    //transpose data
#pragma omp parallel for
    for(int i = 0; i < dims[1]; i++){
        for(int j = 0; j < dims[0]; j++) {
            B[MAT_2D(j, i, dims[1])] = A[MAT_2D(i, j, dims[0])];
        }
    }
    printf("end of matrix_transpose_internal\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/*
 *		softmax			- returns the softmax_cce of arg1 as inputs and arg2 as labels(one_hot)
 */
inline Datum softmax_cce(PG_FUNCTION_ARGS)
{
    return softmax_cce_internal(PG_GETARG_DATUM(0), PG_GETARG_DATUM(1));
}

/*
 *		softmax, returns the softmax_cce loss of arg1 as inputs and arg2 as labels(one_hot)
 */
Datum softmax_cce_internal(Datum inputs_in, Datum labels_in)
{
    printf("begin of softmax_cce_internal\n");
    ArrayType *input, *labels;
    input = DatumGetArrayTypeP(inputs_in);
    labels = DatumGetArrayTypeP(labels_in);
    float8 max, sum = 0.0;
    float8 *data = (float8 *)ARR_DATA_PTR(input);
    float8 *label_data = (float8 *)ARR_DATA_PTR(labels);

    int length = ArrayGetNItems(ARR_NDIM(input), ARR_DIMS(input));
    if (length != ArrayGetNItems(ARR_NDIM(labels), ARR_DIMS(labels)))
    {
        ereport(ERROR, (errmsg("Softmax derivation: Arrays *labels* and *input* do not match!")));
    }
    if (ARR_NDIM(input) > 2)
    {
        ereport(ERROR, (errmsg("Softmax derivation: Array *input* is not a vector!")));
    }
    else if (ARR_NDIM(input) == 2 && (ARR_DIMS(input)[1] != 1 || ARR_DIMS(input)[0] != 1))
    {
        ereport(ERROR, (errmsg("Softmax derivation: Array *input* is not a vector!")));
    }
    if (ARR_NDIM(labels) > 2)
    {
        ereport(ERROR, (errmsg("Softmax derivation: Array *labels* is not a vector!")));
    }
    else if (ARR_NDIM(labels) == 2 && (ARR_DIMS(labels)[1] != 1 || ARR_DIMS(labels)[0] != 1))
    {
        ereport(ERROR, (errmsg("Softmax derivation: Array *label* is not a vector!")));
    }
    float8 *result = (float8 *)palloc0(length * sizeof(float8));

    max = data[0];
    for (int i = 0; i < length; i++)
    {
        if ((max < data[i]))
        {
            max = data[i];
        }
    }
#pragma omp parallel for
    for (int i = 0; i < length; i++)
    {
        result[i] = data[i] - max;
    }
#pragma omp barrier
    for (int i = 0; i < length; i++)
    {
        sum += exp(result[i]);
    }
#pragma omp parallel for
    for (int i = 0; i < length; i++)
    {
        result[i] -= log(sum);
        result[i] *= label_data[i];
    }
#pragma omp barrier
    sum = 0.0;
    for(int i = 0; i < length; i++)
    {
        sum += result[i];
    }
    printf("end of softmax_cce_internal\n");
    PG_RETURN_FLOAT8(sum);
}

/*
 *		softmax_der:			
 *      returns the derivative of the softmax function w.r.t. x(input 1)
 */
Datum softmax_cce_derive(Datum inputs_in, Datum labels_in)
{
    printf("begin of softmax_cce_derive\n");
    ArrayType *input_arr, *labels_arr, *result_arr;
    input_arr = DatumGetArrayTypeP(inputs_in);
    labels_arr = DatumGetArrayTypeP(labels_in);
    float8 max, sum = 0.0;
    float8 *input = (float8 *)ARR_DATA_PTR(input_arr);
    float8 *labels = (float8 *)ARR_DATA_PTR(labels_arr);

    int length = ArrayGetNItems(ARR_NDIM(input_arr), ARR_DIMS(input_arr));
    if (length != ArrayGetNItems(ARR_NDIM(labels_arr), ARR_DIMS(labels_arr)))
    {
        ereport(ERROR, (errmsg("Softmax derivation: Arrays *labels* and *input* do not match!")));
    }
    if (ARR_NDIM(input_arr) > 2) 
    {
        ereport(ERROR, (errmsg("Softmax derivation: Array *input* is not a vector!")));
    }
    else if (ARR_NDIM(input_arr) == 2 && (ARR_DIMS(input_arr)[1] != 1 || ARR_DIMS(input_arr)[0] != 1))
    {
        ereport(ERROR, (errmsg("Softmax derivation: Array *input* is not a vector!")));
    }
    if (ARR_NDIM(labels_arr) > 2)
    {
        ereport(ERROR, (errmsg("Softmax derivation: Array *labels* is not a vector!")));
    }
    else if (ARR_NDIM(labels_arr) == 2 && (ARR_DIMS(labels_arr)[1] != 1 || ARR_DIMS(labels_arr)[0] != 1))
    {
        ereport(ERROR, (errmsg("Softmax derivation: Array *label* is not a vector!")));
    }
    result_arr = initResult(ARR_NDIM(input_arr), ARR_DIMS(input_arr), ARR_LBOUND(input_arr));
    float8 *result = (float8 *)ARR_DATA_PTR(result_arr);

    max = input[0];
    for (int i = 1; i < length; i++)
    {
        if (max < input[i])
        {
            max = input[i];
        }
    }
#pragma omp parallel for
    for (int i = 0; i < length; i++)
    {
        result[i] = input[i] - max;
    }
#pragma omp barrier
    for (int i = 0; i < length; i++)
    {
        sum += exp(result[i]);
    }
#pragma omp parallel for
    for (int i = 0; i < length; i++)
    {
        result[i] -= log(sum);
        result[i] = exp(result[i]);
        result[i] -= labels[i];
    }
    printf("end of softmax_cce_derive\n");
    PG_RETURN_ARRAYTYPE_P(result_arr);
}

/* silu_m   -   apply silu to an entire n-dimensional array*/
Datum silu_m(PG_FUNCTION_ARGS)
{
    return silu_m_internal(PG_GETARG_DATUM(0));
}

/* silu_m   -   apply silu to an entire n-dimensional array*/
Datum silu_m_internal(Datum input)
{
    printf("begin of silu_m_internal\n");
    ArrayType *ret = copyArray(input);
    Datum *data = ARR_DATA_PTR(ret);
#pragma omp parallel for
    for(int i = 0; i < ArrayGetNItems(ARR_NDIM(ret), ARR_DIMS(ret)); i++) {
        float8 val = (DatumGetFloat8(data[i]))/(1 + exp((-1) * DatumGetFloat8(data[i])));
        data[i] = Float8GetDatum(val);
    }
    printf("end of silu_m_internal\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/* silu_m_derive   -   calculate the derivative of element-wise silu*/
Datum silu_m_derive(Datum input)
{
    printf("begin of silu_m_derive\n");
    ArrayType *ret = copyArray(input);
    Datum *data = ARR_DATA_PTR(ret);
#pragma omp parallel for
    for (int i = 0; i < ArrayGetNItems(ARR_NDIM(ret), ARR_DIMS(ret)); i++)
    {
        float8 x = DatumGetFloat8(data[i]);
        float8 val = (1 + exp((-1)*x) + x * exp((-1) * x))/(pow((1 + exp((-1) * x)), 2));
        data[i] = Float8GetDatum(val);
    }
    printf("end of silu_m_derive\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/* sigmoid_m   -   apply sigmoid to an entire n-dimensional array*/
Datum sigmoid_m(PG_FUNCTION_ARGS)
{
    return sigmoid_m_internal(PG_GETARG_DATUM(0));
}

/* sigmoid_m   -   apply sigmoid to an entire n-dimensional array*/
Datum sigmoid_m_internal(Datum input)
{
    printf("begin of sigmoid_m_internal\n");
    ArrayType *ret = copyArray(input);
    Datum *data = ARR_DATA_PTR(ret);
#pragma omp parallel for
    for (int i = 0; i < ArrayGetNItems(ARR_NDIM(ret), ARR_DIMS(ret)); i++)
    {
        float8 x = DatumGetFloat8(data[i]);
        float8 val = (1) / (1 + exp((-1) * x));
        data[i] = Float8GetDatum(val);
    }
    printf("end of sigmoid_m_internal\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/* sigmoid_m_derive   -   calculate the derivative of element-wise sigmoid*/
Datum sigmoid_m_derive(Datum input)
{
    printf("begin of sigmoid_m_derive\n");
    ArrayType *ret = copyArray(input);
    Datum *data = ARR_DATA_PTR(ret);
#pragma omp parallel for
    for (int i = 0; i < ArrayGetNItems(ARR_NDIM(ret), ARR_DIMS(ret)); i++)
    {
        float8 x = DatumGetFloat8(data[i]);
        float8 sig_of_x = (1) / (1 + exp((-1) * x));
        float8 val = sig_of_x * (1 - sig_of_x);
        data[i] = Float8GetDatum(val);
    }
    printf("end of sigmoid_m_derive\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/* tanh_m   -   apply tanh to an entire n-dimensional array*/
Datum tanh_m(PG_FUNCTION_ARGS)
{
    return tanh_m_internal(PG_GETARG_DATUM(0));
}

/* tanh_m   -   apply tanh to an entire n-dimensional array*/
Datum tanh_m_internal(Datum input)
{
    printf("begin of tanh_m_internal\n");
    ArrayType *ret = copyArray(input);
    Datum *data = ARR_DATA_PTR(ret);
#pragma omp parallel for
    for (int i = 0; i < ArrayGetNItems(ARR_NDIM(ret), ARR_DIMS(ret)); i++)
    {
        float8 x = DatumGetFloat8(data[i]);
        float8 val = tanh(x);
        data[i] = Float8GetDatum(val);
    }
    printf("end of tanh_m_internal\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/* tanh_m_derive   -   calculate the derivative of element-wise tanh*/
Datum tanh_m_derive(Datum input)
{
    printf("begin of tanh_m_derive\n");
    ArrayType *ret = copyArray(input);
    Datum *data = ARR_DATA_PTR(ret);
#pragma omp parallel for
    for (int i = 0; i < ArrayGetNItems(ARR_NDIM(ret), ARR_DIMS(ret)); i++)
    {
        float8 x = DatumGetFloat8(data[i]);
        float8 tanh_of_x = tanh(x);
        float8 val = 1 - tanh_of_x * tanh_of_x;
        data[i] = Float8GetDatum(val);
    }
    printf("end of tanh_m_derive\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/* relu_m   -   apply relu to an entire n-dimensional array*/
Datum relu_m(PG_FUNCTION_ARGS)
{
    return relu_m_internal(PG_GETARG_DATUM(0));
}

/* relu_m   -   apply relu to an entire n-dimensional array*/
Datum relu_m_internal(Datum input)
{
    printf("begin of relu_m_internal\n");
    ArrayType *ret = copyArray(input);
    Datum *data = (Datum *)ARR_DATA_PTR(ret);
#pragma omp parallel for
    for (int i = 0; i < ArrayGetNItems(ARR_NDIM(ret), ARR_DIMS(ret)); i++)
    {
        float8 x = DatumGetFloat8(data[i]);
        float8 val = Max(x, 0);
        data[i] = Float8GetDatum(val);
    }
    printf("end of relu_m_internal\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/* relu_m_derive   -   calculate the derivative of element-wise relu*/
Datum relu_m_derive(Datum input)
{
    printf("begin of relu_m_derive\n");
    ArrayType *ret = copyArray(input);
    Datum *data = (Datum *)ARR_DATA_PTR(ret);
#pragma omp parallel for
    for (int i = 0; i < ArrayGetNItems(ARR_NDIM(ret), ARR_DIMS(ret)); i++)
    {
        float8 x = DatumGetFloat8(data[i]);
        float8 val = (x > 0) ? (1) : (0);
        data[i] = Float8GetDatum(val);
    }
    printf("end of relu_m_derive\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/*
 * Create a new array, based on the given dimensions, including 
 * the data_array afterwards
 */
ArrayType *initResult(int ndims, int *dims, int *lbs)
{
    printf("begin of initResult\n");
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
    printf("end of initResult\n");
    return ret;
}

/*
 * Create a new array, copy of the given Array
 */
ArrayType *copyArray(Datum orgArray)
{
    printf("begin of copyArray\n");
    //get size and dimensions from original
    ArrayType *original = DatumGetArrayTypeP(orgArray);
    int ndims = ARR_NDIM(original);
    int *dims = ARR_DIMS(original);
    int *lbs = ARR_LBOUND(original);
    //create new array
    int nelems = ArrayGetNItems(ndims, dims);
    int32 nbytes = nelems * ALIGNOF_DOUBLE;
    nbytes += ARR_OVERHEAD_NONULLS(ndims);
    ArrayType *ret = (ArrayType *)palloc0(nbytes);
    SET_VARSIZE(ret, nbytes);
    ret->ndim = ndims;
    ret->dataoffset = 0;
    ret->elemtype = FLOAT8OID;
    //copy all relevant data
    memcpy(ARR_DIMS(ret), dims, ndims * sizeof(int));
    memcpy(ARR_LBOUND(ret), lbs, ndims * sizeof(int));
    memcpy(ARR_DATA_PTR(ret), ARR_DATA_PTR(original), nbytes);
    printf("end of copyArray\n");
    return ret;
}

/*
 * Create a new ArrayType, either filled with a given value, or the identity matrix
 * NOTICE: The identitymatrix will always be quadratic, but every other matrix does not have to be an can even be 1x3(vector)
 */
Datum createArray(int *dims, const float8 value, const bool identityMatrix)
{
    printf("begin of createArray\n");
    ArrayType *ret;
    int lbs[2] = {1,1};
    ret = initResult(2, dims, lbs);
    Datum* data = (Datum *)ARR_DATA_PTR(ret);
    if(identityMatrix) {
#pragma omp parallel for
        for (int i = 0; i < dims[0]; i++)
        {
            for (int j = 0; j < dims[0]; j++)
            {
                if (i == j)
                {
                    data[MAT_2D(i, j, dims[0])] = Float8GetDatum(1.0);
                }
                else
                {
                    data[MAT_2D(i, j, dims[0])] = Float8GetDatum(0.0);
                }
            }
        }
    } else {
#pragma omp parallel for
        for (int i = 0; i < dims[0]; i++)
        {
            for (int j = 0; j < dims[1]; j++)
            {
                data[MAT_2D(i, j, dims[1])] = Float8GetDatum(value);
            }
        }
    }
    printf("end of createArray\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}