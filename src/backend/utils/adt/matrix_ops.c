/*-------------------------------------------------------------------------
 *
 * matrix_ops.c
 *	  This file contains some support routines 
 *    required for matrix arithmetic functions.
 *    Important: Because Postgres only passes Datum as an all-encompassing type, 
 *               that does not save its type, Arrays, which have -1 as 
 *               the first lbs, will be considered scalar values and only have a single item, 
 *               which is that scalar itself.
 *               This is such that the automatic differentiation can output scalars as operations results.
 * 
 *               Vectors are always 1D Nx1 Arrays, if one needs a 1xN Array, use 2D Arrays with dim[0] == 1
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
    //  The formal PostgreSQL array objects:
    ArrayType *a1, *a2, *ret;

    // The size of each array:
    int length1, length2;

    if (PG_ARGISNULL(0))
    {
        ereport(ERROR, (errmsg("Matrix Multiplication external: Null pointer passed as Matrix A!")));
    }
    if (PG_ARGISNULL(1))
    {
        ereport(ERROR, (errmsg("Matrix Multiplication external: Null pointer passed as Matrix B!")));
    }

    // Extract the PostgreSQL arrays from the parameters passed to this function call.
    a1 = PG_GETARG_ARRAYTYPE_P(0);
    a2 = PG_GETARG_ARRAYTYPE_P(1);

    int *dim1 = ARR_DIMS(a1);
    int *dim2 = ARR_DIMS(a2);
    int lbs[2] = {1, 1};
    int dims[2] = {1, 1};
    int ndim_res = 2;

    int dima, dimb, dimc;
    dima = 1, dimb = 1, dimc = 1;

    length1 = ArrayGetNItems(ARR_NDIM(a1), ARR_DIMS(a1));
    length2 = ArrayGetNItems(ARR_NDIM(a2), ARR_DIMS(a2));

    // if one or both are scalars, use element-wise multiplication
    if (isScalar(a1))
    {
        ret = PG_GETARG_ARRAYTYPE_P_COPY(1);
        float8 *data = (float8 *)ARR_DATA_PTR(ret);
#pragma omp parallel for
        for (int i = 0; i < length2; i++)
        {
            data[i] *= DatumGetFloat8(((Datum *)ARR_DATA_PTR(a1))[0]);
        }
        // printf("MatA is scalar, MatB has dimensions: [%d, %d]\n", ARR_DIMS(ret)[0], ARR_DIMS(ret)[1]);
        PG_RETURN_ARRAYTYPE_P(ret);
    }
    if (isScalar(a2))
    {
        ret = PG_GETARG_ARRAYTYPE_P_COPY(0);
        float8 *data = (float8 *)ARR_DATA_PTR(ret);
#pragma omp parallel for
        for (int i = 0; i < length1; i++)
        {
            data[i] *= DatumGetFloat8(((Datum *)ARR_DATA_PTR(a2))[0]);
        }
        // printf("MatB is scalar, MatA has dimensions: [%d, %d]\n", ARR_DIMS(ret)[0], ARR_DIMS(ret)[1]);
        PG_RETURN_ARRAYTYPE_P(ret);
    }

    if (ARR_NDIM(a1) == 1 && ARR_NDIM(a2) == 1)
    {
        ereport(ERROR, (errmsg("Matrix Multiplication Internal: Two column vectors can not be multiplied!")));
    }

    // no matter the NDIM of MatA, dima will always be the first dimension of the result
    dima = dim1[0];
    dims[0] = dim1[0];

    // if NDIM of MatB is 2, it is either a transposed vector(row-vector), or full matrix
    // in either case, we create a result matrix, as both would result in matrices
    //(except for 1xN * Nx1 case)
    if (ARR_NDIM(a2) == 2)
    {
        dimb = dim2[0];
        dimc = dim2[1];
        dims[1] = dim2[1];
        ndim_res = 2;
    }
    else
    {
        dimb = dim2[0];
        dimc = 1;
        dims[1] = 1;
        ndim_res = 1;
    }

    // 1xN * Nx1 case (create scalar)
    if (dima == 1 && dimc == 1)
    {
        lbs[0] = -1;
        ndim_res = 1;
    }

    ret = initResult(ndim_res, dims, lbs);

    float8 *ps1 = (float8 *)ARR_DATA_PTR(a1);
    float8 *ps2 = (float8 *)ARR_DATA_PTR(a2);
    float8 *ps;

    ps = (float8 *)ARR_DATA_PTR(ret);
#pragma omp parallel for
    for (int i = 0; i < dima; i++)
    {
        for (int j = 0; j < dimb; j++)
        {
            for (int k = 0; k < dimc; k++)
            {
                ps[MAT_2D(i, k, dimc)] += ps1[MAT_2D(i, j, dimb)] * ps2[MAT_2D(j, k, dimc)];
                // ps[MAT_2D(i, k, dimc)] += ps1[((transposeA) ? (MAT_2D(j, i, dima)) : (MAT_2D(i, j, dimb)))] * ps2[((transposeB) ? (MAT_2D(k, j, dimb)) : (MAT_2D(j, k, dimc)))];
            }
        }
    }
    // printf("end of matrix_mul_external\n");
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

    // The size of each array:
    int length1, length2;

    if(DatumGetPointer(MatA) == NULL) 
    {
        ereport(ERROR, (errmsg("Matrix Multiplication Internal: Null pointer passed as Matrix A!")));
    }
    if (DatumGetPointer(MatB) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix Multiplication Internal: Null pointer passed as Matrix B!")));
    }

    // Extract the PostgreSQL arrays from the parameters passed to this function call.
    a1 = DatumGetArrayTypeP((transposeA) ? (matrix_transpose_internal(MatA)) : (MatA));
    a2 = DatumGetArrayTypeP((transposeB) ? (matrix_transpose_internal(MatB)) : (MatB));

    int *dim1 = ARR_DIMS(a1);
    int *dim2 = ARR_DIMS(a2);
    int lbs[2] = {1, 1};
    int dims[2] = {1, 1};
    int ndim_res;

    int dima, dimb, dimc;
    dima = 1, dimb = 1, dimc = 1;

    length1 = ArrayGetNItems(ARR_NDIM(a1), ARR_DIMS(a1));
    length2 = ArrayGetNItems(ARR_NDIM(a2), ARR_DIMS(a2));

    // if one or both are scalars, use element-wise multiplication
    if (isScalar(a1))
    {
        ret = copyArray(MatB);
        float8 *data = (float8 *)ARR_DATA_PTR(ret);
#pragma omp parallel for
        for (int i = 0; i < length2; i++)
        {
            data[i] *= DatumGetFloat8(((Datum *)ARR_DATA_PTR(a1))[0]);
        }
        printf("MatA is scalar, MatB has dimensions: [%d, %d]\n", ARR_DIMS(ret)[0], ARR_DIMS(ret)[1]);
        PG_RETURN_ARRAYTYPE_P(ret);
    }
    if (isScalar(a2))
    {
        ret = copyArray(MatA);
        float8 *data = (float8 *)ARR_DATA_PTR(ret);
#pragma omp parallel for
        for (int i = 0; i < length1; i++)
        {
            data[i] *= DatumGetFloat8(((Datum *)ARR_DATA_PTR(a2))[0]);
        }
        printf("MatB is scalar, MatA has dimensions: [%d, %d]\n", ARR_DIMS(ret)[0], ARR_DIMS(ret)[1]);
        PG_RETURN_ARRAYTYPE_P(ret);
    }

    if(ARR_NDIM(a1) == 1 && ARR_NDIM(a2) == 1) {
        ereport(ERROR, (errmsg("Matrix Multiplication Internal: Two column vectors can not be multiplied!")));
    }

    //no matter the NDIM of MatA, dima will always be the first dimension of the result
    dima = dim1[0];
    dims[0] = dim1[0];

    //if NDIM of MatB is 2, it is either a transposed vector(row-vector), or full matrix
    //in either case, we create a result matrix, as both would result in matrices
    //(except for 1xN * Nx1 case)
    if (ARR_NDIM(a2) == 2)
    {
        dimb = dim2[0];
        dimc = dim2[1];
        dims[1] = dim2[1];
        ndim_res = 2;
    }
    else
    {
        dimb = dim2[0];
        dimc = 1;
        dims[1] = 1;
        ndim_res = 1;
    }

    //1xN * Nx1 case (create scalar)
    if(dima == 1 && dimc == 1) {
        lbs[0] = -1;
        ndim_res = 1;
    }

    ret = initResult(ndim_res, dims, lbs);

    float8 *ps1 = (float8 *)ARR_DATA_PTR(a1);
    float8 *ps2 = (float8 *)ARR_DATA_PTR(a2);
    float8 *ps;

    
    ps = (float8 *)ARR_DATA_PTR(ret);
#pragma omp parallel for
    for (int i = 0; i < dima; i++)
    {
        for (int j = 0; j < dimb; j++)
        {
            for (int k = 0; k < dimc; k++)
            {
                ps[MAT_2D(i, k, dimc)] += ps1[MAT_2D(i, j, dimb)] * ps2[MAT_2D(j, k, dimc)];
                // ps[MAT_2D(i, k, dimc)] += ps1[((transposeA) ? (MAT_2D(j, i, dima)) : (MAT_2D(i, j, dimb)))] * ps2[((transposeB) ? (MAT_2D(k, j, dimb)) : (MAT_2D(j, k, dimc)))];
            }
        }
    }
    //printf("end of matrix_mul_internal\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/*
 * Adds Two matricies elementwise together inplace (a += b)
 */
Datum matrix_add_inplace(Datum MatA, Datum MatB)
{
    printf("begin of matrix_add_inplace\n");
    // The formal PostgreSQL array objects:
    ArrayType *a1, *a2;

    // The size of each array:
    int length1, length2;

    if (DatumGetPointer(MatA) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix add Internal: Null pointer passed as Matrix A!")));
    }
    if (DatumGetPointer(MatB) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix add Internal: Null pointer passed as Matrix B!")));
    }

    a1 = DatumGetArrayTypeP(MatA);
    a2 = DatumGetArrayTypeP(MatB);

    length1 = ArrayGetNItems(ARR_NDIM(a1), ARR_DIMS(a1));
    length2 = ArrayGetNItems(ARR_NDIM(a2), ARR_DIMS(a2));

    if (isScalar(a1))
    {
        Datum elem = ((Datum *)ARR_DATA_PTR(a1))[0];
        a1 = copyArray(MatB);
        Datum *data = (Datum *)ARR_DATA_PTR(a1);
#pragma omp parallel for
        for(int i = 0; i < length2; i++) {
            data[i] = Float8GetDatum(DatumGetFloat8(data[i]) + DatumGetFloat8(elem));
        }
        //printf("end of matrix_add_inplace\n");
        PG_RETURN_ARRAYTYPE_P(a1);
    }
    if (isScalar(a2))
    {
        Datum *data = (Datum *)ARR_DATA_PTR(a1);
        Datum elem = ((Datum *)ARR_DATA_PTR(a2))[0];
#pragma omp parallel for
        for (int i = 0; i < length2; i++)
        {
            data[i] = Float8GetDatum(DatumGetFloat8(data[i]) + DatumGetFloat8(elem));
        }

        //printf("end of matrix_add_inplace\n");
        PG_RETURN_ARRAYTYPE_P(a1);
    }

    if(ARR_NDIM(a1) != ARR_NDIM(a2)) {
        ereport(ERROR, (errmsg("Matrix element-wise addition(inplace): Number of dimensions mismatched!")));
    }
    for(int i = 0; i < ARR_NDIM(a1); i++) {
        if (ARR_DIMS(a1)[i] != ARR_DIMS(a2)[i])
        {
            ereport(ERROR, (errmsg("Matrix element-wise addition(inplace): dimension %d mismatched!", i + 1)));
        }
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
    //printf("end of matrix_add_inplace\n");
    PG_RETURN_ARRAYTYPE_P(a1);
}

/*
 * Multiply two matrices element-wise and return product as new matrix
 */
Datum matrix_elem_mult_external(PG_FUNCTION_ARGS)
{
    return matrix_elem_mult(PG_GETARG_DATUM(0), PG_GETARG_DATUM(1));
}

/*
 * Multiply two matrices element-wise and return product as new matrix
 */
Datum matrix_elem_mult(Datum matA, Datum matB)
{
    printf("begin of matrix_elem_mult\n");
    ArrayType *a, *b, *ret;
    int ndims, length1, length2;
    int *dims;

    if (DatumGetPointer(matA) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix Elem-Multiplication Internal: Null pointer passed as Matrix A!")));
    }
    if (DatumGetPointer(matB) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix Elem-Multiplication Internal: Null pointer passed as Matrix B!")));
    }

    a = DatumGetArrayTypeP(matA);
    b = DatumGetArrayTypeP(matB);

    length1 = ArrayGetNItems(ARR_NDIM(a), ARR_DIMS(a));
    length2 = ArrayGetNItems(ARR_NDIM(b), ARR_DIMS(b));

    if (isScalar(a))
    {
        ret = copyArray(matB);

        Datum *data = (Datum *)ARR_DATA_PTR(ret);
        const Datum elem = ((Datum *)ARR_DATA_PTR(a))[0];
#pragma omp parallel for
        for (int i = 0; i < length2; i++)
        {
            data[i] = Float8GetDatum(DatumGetFloat8(elem) * DatumGetFloat8(data[i]));
        }
        // printf("end of matrix_elem_mult\n");
        PG_RETURN_ARRAYTYPE_P(ret);
    }
    else if (isScalar(b))
    {
        ret = copyArray(matA);

        Datum *data = (Datum *)ARR_DATA_PTR(ret);
        const Datum elem = ((Datum *)ARR_DATA_PTR(b))[0];
#pragma omp parallel for
        for (int i = 0; i < length1; i++)
        {
            data[i] = Float8GetDatum(DatumGetFloat8(elem) * DatumGetFloat8(data[i]));
        }
        // printf("end of matrix_elem_mult\n");
        PG_RETURN_ARRAYTYPE_P(ret);
    }

    ndims = ARR_NDIM(a);
    dims = ARR_DIMS(a);
    if (ndims != ARR_NDIM(b))
    {
        ereport(ERROR, (errmsg("Matrix element-wise Multiplication: Number of dimensions mismatched!")));
    }
    for (int i = 0; i < ndims; i++)
    {
        if (dims[i] != ARR_DIMS(b)[i])
        {
            ereport(ERROR, (errmsg("Matrix element-wise Multiplication: Matrices are mismatched!")));
        }
    }

    ret = initResult(ndims, dims, ARR_LBOUND(a));

    Datum *ret_data = (Datum *)ARR_DATA_PTR(ret);
    Datum *a_data = (Datum *)ARR_DATA_PTR(a);
    Datum *b_data = (Datum *)ARR_DATA_PTR(b);

#pragma omp parrallel for
    for (int i = 0; i < ArrayGetNItems(ARR_NDIM(a), ARR_DIMS(a)); i++)
    {
        ret_data[i] = Float8GetDatum(DatumGetFloat8(a_data[i]) * DatumGetFloat8(b_data[i]));
    }
    // printf("end of matrix_elem_mult\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/*
 * Subtract matrix from matrix and return result as new matrix
 */
Datum mat_sub_mm(PG_FUNCTION_ARGS)
{
    printf("begin of mat_sub_mm!\n");
    ArrayType *ret, *a1, *a2;
    a1 = PG_GETARG_ARRAYTYPE_P(0);
    a2 = PG_GETARG_ARRAYTYPE_P(1);

    if(ARR_NDIM(a1) != ARR_NDIM(a2)) {
        printf("NDIM_a1: %d\nNDIM_a2: %d\n", ARR_NDIM(a1), ARR_NDIM(a2));
        ereport(ERROR, (errmsg("Matrix element-wise Subtraction: Number of dimensions mismatched!")));
    }
    for(int i = 0; i < ARR_NDIM(a1); i++) {
        if(ARR_DIMS(a1)[i] != ARR_DIMS(a2)[i]) {
            ereport(ERROR, (errmsg("Matrix element-wise Subtraction: Dimensions mismatched!")));
        }
    }

    ret = initResult(ARR_NDIM(a1), ARR_DIMS(a1), ARR_LBOUND(a1));
    float8 *retData = (float8 *)ARR_DATA_PTR(ret);
    float8 *a1Data = (float8 *)ARR_DATA_PTR(a1);
    float8 *a2Data = (float8 *)ARR_DATA_PTR(a2);

#pragma omp parallel for
    for (int i = 0; i < ArrayGetNItems(ARR_NDIM(ret), ARR_DIMS(ret)); i++) {
        retData[i] = a1Data[i] - a2Data[i];
    }

    PG_RETURN_ARRAYTYPE_P(ret);
}

/*
 * Subtract scalar from matrix and return result as new matrix
 */
Datum mat_sub_ms(PG_FUNCTION_ARGS)
{
    printf("begin of mat_sub_ms!\n");
    ArrayType *ret, *a1;
    float8 scalar;
    a1 = PG_GETARG_ARRAYTYPE_P(0);
    scalar = PG_GETARG_FLOAT8(1);

    ret = initResult(ARR_NDIM(a1), ARR_DIMS(a1), ARR_LBOUND(a1));
    float8 *retData = (float8 *)ARR_DATA_PTR(ret);

#pragma omp parallel for
    for (int i = 0; i < ArrayGetNItems(ARR_NDIM(ret), ARR_DIMS(ret)); i++)
    {
        retData[i] = retData[i] - scalar;
    }

    PG_RETURN_ARRAYTYPE_P(ret);
}

/*
 * Subtract matrix from scalar and return result as new matrix
 */
Datum mat_sub_sm(PG_FUNCTION_ARGS)
{
    printf("begin of mat_sub_sm!\n");
    ArrayType *ret, *a1;
    float8 scalar;
    a1 = PG_GETARG_ARRAYTYPE_P(1);
    scalar = PG_GETARG_FLOAT8(0);

    ret = initResult(ARR_NDIM(a1), ARR_DIMS(a1), ARR_LBOUND(a1));
    float8 *retData = (float8 *)ARR_DATA_PTR(ret);

#pragma omp parallel for
    for (int i = 0; i < ArrayGetNItems(ARR_NDIM(ret), ARR_DIMS(ret)); i++)
    {
        retData[i] = scalar - retData[i];
    }

    PG_RETURN_ARRAYTYPE_P(ret);
}

/*
 * Multiply scalar with matrix elementwise and return result as new matrix
 */
Datum mat_mul_sm(PG_FUNCTION_ARGS)
{
    printf("begin of mat_mul_sm!\n");
    ArrayType *ret, *a1;
    float8 scalar;
    a1 = PG_GETARG_ARRAYTYPE_P(1);
    scalar = PG_GETARG_FLOAT8(0);

    ret = initResult(ARR_NDIM(a1), ARR_DIMS(a1), ARR_LBOUND(a1));
    float8 *retData = (float8 *)ARR_DATA_PTR(ret);

#pragma omp parallel for
    for (int i = 0; i < ArrayGetNItems(ARR_NDIM(ret), ARR_DIMS(ret)); i++)
    {
        retData[i] = scalar * retData[i];
    }

    PG_RETURN_ARRAYTYPE_P(ret);
}

/*
 * Return transposed copy of matrix/vector
 */
Datum mat_transpose_external(PG_FUNCTION_ARGS) {
    return matrix_transpose_internal(PG_GETARG_ARRAYTYPE_P(0));
}

/*
 * Transpose a matrix and return transposed copy
 */
Datum matrix_transpose_internal(Datum MatA)
{
    printf("begin of matrix_transpose_internal\n");
    // The formal PostgreSQL array objects:
    ArrayType *ret, *org;
    if (DatumGetPointer(MatA) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix Multiplication Internal: Null pointer passed as Matrix A!")));
    }
    org = DatumGetArrayTypeP(MatA);

    int dims[2], lbs[2];

    if(isScalar(org)) {
        return copyArray(MatA);
    }
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
    if ((ARR_NDIM(org) == 2 && ARR_DIMS(org)[0] == 1))
    {
        dims[0] = ARR_DIMS(org)[1];
        lbs[0] = ARR_LBOUND(org)[1];
        ret = initResult(1, dims, lbs);
        memcpy(ARR_DATA_PTR(ret), ARR_DATA_PTR(org), dims[0] * ALIGNOF_DOUBLE);
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
    //printf("end of matrix_transpose_internal\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/*
 * Aggregate average array over multiple matrices/vectors of same dimension
 */
Datum mat_avg(PG_FUNCTION_ARGS)
{
    //printf("mat_avg begin!\n");
    if (AggCheckCallContext(fcinfo, NULL))
    {
        ArrayType *state_array, *new_val;

        state_array = PG_GETARG_ARRAYTYPE_P(0);
        //the data contained in state_array
        Datum *state_data = (Datum *)ARR_DATA_PTR(state_array);
        new_val = PG_GETARG_ARRAYTYPE_P(1);
        // printf("mat_avg begin:\n %p\taddress of state_data[1]\n %p\t address of new_val\n", (void *) state_data[1], (void *) new_val);
        // printf("mat_avg begin:\n %p\taddress of state_data[1]\n", (void *) state_data[1]);

        if (DatumGetFloat8(state_data[0]) == 0.0) //if avg has not been invoked before, copy the current first array
        {
            state_data[1] = PointerGetDatum(copyArray(PointerGetDatum(new_val)));
        }
        else //otherwise add new_val on top of the current accumulate
        {
            state_data[1] = matrix_add_inplace(state_data[1], PointerGetDatum(new_val));
            state_data[1] = PointerGetDatum(copyArray(state_data[1]));
        }
        //increment the counter
        state_data[0] = Float8GetDatum(DatumGetFloat8(state_data[0]) + 1.0);
        // printf("mat_avg end:\n %p\taddress of state_data[1]\n\n", (void *)state_data[1]);
        PG_RETURN_ARRAYTYPE_P(state_array);
    }
    ereport(ERROR, (errmsg("Aggregate function AVG(float8[]) called in non-aggregate context!")));
    PG_RETURN_NULL();
}

/*
 * Aggregate function avg, final function (divide each element by the count of elements)
 */
Datum mat_avg_final(PG_FUNCTION_ARGS)
{
    if (AggCheckCallContext(fcinfo, NULL))
    {
        ArrayType *state_array, *avg_array;
        state_array = PG_GETARG_ARRAYTYPE_P(0);

        Datum *state_data = (Datum *)ARR_DATA_PTR(state_array);
        if(DatumGetFloat8(state_data[0]) == 0.0) {
            PG_RETURN_NULL();
        } else {
            avg_array = DatumGetArrayTypeP(state_data[1]);
            float8 *avg_data = (float8 *) ARR_DATA_PTR(avg_array);
#pragma omp parallel for
            for(int i = 0; i < ArrayGetNItems(ARR_NDIM(avg_array), ARR_DIMS(avg_array)); i++) {
                avg_data[i] /= DatumGetFloat8(state_data[0]); 
            }
            PG_RETURN_ARRAYTYPE_P(avg_array);
        }
    }
    ereport(ERROR, (errmsg("Aggregate function AVG-final(float8[]) called in non-aggregate context!")));
    PG_RETURN_NULL();
}

/*
 * Apply a single gradient update to a matrix(updates inplace)
 * params: weights -> matrix to be updated
 *         derivatives -> gradient of weights
 *         lr -> learn_rate modifier to gradient update
 *         batch_size -> amount of derivatives, that went int derivatives
 */
Datum mat_apply_gradient(Datum weights, Datum derivatives, float8 learning_rate, int batch_size){
    printf("begin of mat_apply_gradient!\n");
    ArrayType *weights_a, *derivatives_a;
    if (DatumGetPointer(weights) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix apply Gradient: Null pointer passed as Matrix weights!")));
    }
    if (DatumGetPointer(derivatives) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix apply gradient: Null pointer passed as Matrix derivatives!")));
    }
    weights_a = DatumGetArrayTypeP(weights);
    derivatives_a = DatumGetArrayTypeP(derivatives);

    if(isScalar(weights_a)) {
        weights_a = copyArray(derivatives);
        Datum *data = ARR_DATA_PTR(weights_a);
        for(int i = 0; i < ArrayGetNItems(ARR_NDIM(weights_a), ARR_DIMS(weights_a)); i++) {
            data[i] = Float8GetDatum((DatumGetFloat8(data[i]) * learning_rate) / batch_size);
        }
        PG_RETURN_ARRAYTYPE_P(weights_a);
    }
    if(isScalar(derivatives_a)) {
        ereport(ERROR, (errmsg("mat_apply_gradient: The to-be-applied gradient cannot be a scalar!")));
    }

    int ndims = ARR_NDIM(weights_a);
    int *dims = ARR_DIMS(weights_a);
    if(ndims != ARR_NDIM(derivatives_a)) {
        ereport(ERROR, (errmsg("mat_apply_gradient: derivatives and weight matrix do not match!")));
    }
    for(int i = 0; i < ndims; i++) {
        if(dims[i] != ARR_DIMS(derivatives_a)) {
            ereport(ERROR, (errmsg("mat_apply_gradient: Dimensions do not match in derivatives and weights!")));
        }
    }

    Datum *data = ARR_DATA_PTR(weights_a);
    Datum *derivatives_data = ARR_DATA_PTR(derivatives_a);

    for(int i = 0; i < ArrayGetNItems(ndims, dims); i++) {
        data[i] = Float8GetDatum(DatumGetFloat8(data[i]) - ((learning_rate * DatumGetFloat8(derivatives_data[i]))/batch_size));
    }
    PG_RETURN_ARRAYTYPE_P(weights_a);
}

/*
 * Return index of largest value
 */
Datum index_max(PG_FUNCTION_ARGS) {
    if (PG_ARGISNULL(0))
    {
        ereport(ERROR, (errmsg("Index_max recieved null input!")));
    }
    ArrayType *input = PG_GETARG_ARRAYTYPE_P(0);
    float8 *data = (float8 *)ARR_DATA_PTR(input);
    float8 max_value = data[0];
    int max_index = 0;
    for(int i = 1; i < ArrayGetNItems(ARR_NDIM(input), ARR_DIMS(input)); i++) {
        if(data[i] > max_value) {
            max_value = data[i];
            max_index = i;
        }
    }
    PG_RETURN_INT32(max_index);
}

/*
 * Softmax implementation with numerical stability
 */
Datum softmax(PG_FUNCTION_ARGS)
{
    printf("begin of mat_softmax (without cross-entropy loss)!\n");
    ArrayType *array;

    if(PG_ARGISNULL(0)) {
        ereport(ERROR, (errmsg("Softmax recieved null input!")));
    }

    array = copyArray(PG_GETARG_DATUM(0));
    float8 *data = (float8 *)ARR_DATA_PTR(array);
    float8 max = 0.0, sum = 0.0;

    for(int i = 0; i < ArrayGetNItems(ARR_NDIM(array), ARR_DIMS(array)); i++) 
    {
        if(max < data[i])
            max = data[i];
    }
    for (int i = 0; i < ArrayGetNItems(ARR_NDIM(array), ARR_DIMS(array)); i++)
    {
        sum += exp(data[i] - max);
    }
#pragma omp parallel for
    for (int i = 0; i < ArrayGetNItems(ARR_NDIM(array), ARR_DIMS(array)); i++)
    {
        data[i] = exp(data[i] - max) / sum;
    }
    PG_RETURN_ARRAYTYPE_P(array);
}

/*
 *		softmax, returns the softmax_cce loss of arg1 as inputs and arg2 as labels(one_hot)
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
    if (DatumGetPointer(inputs_in) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix Multiplication Internal: Null pointer passed as Matrix Inputs!")));
    }
    if (DatumGetPointer(labels_in) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix Multiplication Internal: Null pointer passed as Matrix Labels!")));
    }
    input = DatumGetArrayTypeP(inputs_in);
    labels = DatumGetArrayTypeP(labels_in);
    float8 max, sum = 0.0;
    float8 *data = (float8 *)ARR_DATA_PTR(input);
    float8 *label_data = (float8 *)ARR_DATA_PTR(labels);

    int length = ArrayGetNItems(ARR_NDIM(input), ARR_DIMS(input));
    if (length != ArrayGetNItems(ARR_NDIM(labels), ARR_DIMS(labels)))
    {
        ereport(ERROR, (errmsg("Softmax: Arrays *labels* and *input* do not match!")));
    }
    if ((ARR_NDIM(input) > 2 && ARR_NDIM(input) <= 0) || (ARR_NDIM(input) == 2 && ARR_DIMS(input)[0] != 1))
    {
        ereport(ERROR, (errmsg("Softmax: Array *input* is not a vector!")));
    }
    if ((ARR_NDIM(labels) > 2 && ARR_NDIM(labels) <= 0) || (ARR_NDIM(labels) == 2 && ARR_DIMS(labels)[0] != 1))
    {
        ereport(ERROR, (errmsg("Softmax: Array *labels* is not a vector!")));
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
    //printf("end of softmax_cce_internal\n");
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
    if (DatumGetPointer(inputs_in) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix Multiplication Internal: Null pointer passed as Matrix Inputs!")));
    }
    if (DatumGetPointer(labels_in) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix Multiplication Internal: Null pointer passed as Matrix Labels!")));
    }
    input_arr = DatumGetArrayTypeP(inputs_in);
    labels_arr = DatumGetArrayTypeP(labels_in);
    float8 max, sum = 0.0;
    float8 *input = (float8 *)ARR_DATA_PTR(input_arr);
    float8 *labels = (float8 *)ARR_DATA_PTR(labels_arr);

    int length = ArrayGetNItems(ARR_NDIM(input_arr), ARR_DIMS(input_arr));
    if (length != ArrayGetNItems(ARR_NDIM(labels_arr), ARR_DIMS(labels_arr)))
    {
        ereport(ERROR, (errmsg("Softmax Derivation: Arrays *labels* and *input* do not match!")));
    }
    if ((ARR_NDIM(input_arr) > 2 && ARR_NDIM(input_arr) <= 0) || (ARR_NDIM(input_arr) == 2 && ARR_DIMS(input_arr)[0] != 1))
    {
        ereport(ERROR, (errmsg("Softmax Derivation: Array *input* is not a vector!")));
    }
    if ((ARR_NDIM(labels_arr) > 2 && ARR_NDIM(labels_arr) <= 0) || (ARR_NDIM(labels_arr) == 2 && ARR_DIMS(labels_arr)[0] != 1))
    {
        ereport(ERROR, (errmsg("Softmax Derivation: Array *labels* is not a vector!")));
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
    //printf("end of softmax_cce_derive\n");
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
    //printf("begin of silu_m_internal\n");
    if (DatumGetPointer(input) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix Silu Internal: Null pointer passed as Matrix Inputs!")));
    }
    ArrayType *ret = copyArray(input);
    Datum *data = ARR_DATA_PTR(ret);
#pragma omp parallel for
    for(int i = 0; i < ArrayGetNItems(ARR_NDIM(ret), ARR_DIMS(ret)); i++) {
        float8 val = (DatumGetFloat8(data[i]))/(1 + exp((-1) * DatumGetFloat8(data[i])));
        data[i] = Float8GetDatum(val);
    }
    //printf("end of silu_m_internal\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/* silu_m_derive   -   calculate the derivative of element-wise silu*/
Datum silu_m_derive(Datum input)
{
    //printf("begin of silu_m_derive\n");
    if (DatumGetPointer(input) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix Silu Derive: Null pointer passed as Matrix Inputs!")));
    }
    ArrayType *ret = copyArray(input);
    Datum *data = ARR_DATA_PTR(ret);
#pragma omp parallel for
    for (int i = 0; i < ArrayGetNItems(ARR_NDIM(ret), ARR_DIMS(ret)); i++)
    {
        float8 x = DatumGetFloat8(data[i]);
        float8 val = (1 + exp((-1)*x) + x * exp((-1) * x))/(pow((1 + exp((-1) * x)), 2));
        data[i] = Float8GetDatum(val);
    }
    //printf("end of silu_m_derive\n");
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
    //printf("begin of sigmoid_m_internal\n");
    if (DatumGetPointer(input) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix Sigmoid Internal: Null pointer passed as Matrix Inputs!")));
    }
    ArrayType *ret = copyArray(input);
    Datum *data = ARR_DATA_PTR(ret);
#pragma omp parallel for
    for (int i = 0; i < ArrayGetNItems(ARR_NDIM(ret), ARR_DIMS(ret)); i++)
    {
        float8 x = DatumGetFloat8(data[i]);
        float8 val = (1) / (1 + exp((-1) * x));
        data[i] = Float8GetDatum(val);
    }
    //printf("end of sigmoid_m_internal\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/* sigmoid_m_derive   -   calculate the derivative of element-wise sigmoid*/
Datum sigmoid_m_derive(Datum input)
{
    //printf("begin of sigmoid_m_derive\n");
    if (DatumGetPointer(input) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix Sigmoid Derive: Null pointer passed as Matrix Inputs!")));
    }
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
    //printf("end of sigmoid_m_derive\n");
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
    if (DatumGetPointer(input) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix tanh Internal: Null pointer passed as Matrix Inputs!")));
    }
    ArrayType *ret = copyArray(input);
    Datum *data = ARR_DATA_PTR(ret);
#pragma omp parallel for
    for (int i = 0; i < ArrayGetNItems(ARR_NDIM(ret), ARR_DIMS(ret)); i++)
    {
        float8 x = DatumGetFloat8(data[i]);
        float8 val = tanh(x);
        data[i] = Float8GetDatum(val);
    }
    //printf("end of tanh_m_internal\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/* tanh_m_derive   -   calculate the derivative of element-wise tanh*/
Datum tanh_m_derive(Datum input)
{
    printf("begin of tanh_m_derive\n");
    if (DatumGetPointer(input) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix tanh derive: Null pointer passed as Matrix Inputs!")));
    }
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
    //printf("end of tanh_m_derive\n");
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
    //printf("begin of relu_m_internal\n");
    if (DatumGetPointer(input) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix relu Internal: Null pointer passed as Matrix Inputs!")));
    }
    ArrayType *ret = copyArray(input);
    Datum *data = (Datum *)ARR_DATA_PTR(ret);
#pragma omp parallel for
    for (int i = 0; i < ArrayGetNItems(ARR_NDIM(ret), ARR_DIMS(ret)); i++)
    {
        float8 x = DatumGetFloat8(data[i]);
        float8 val = Max(x, 0);
        data[i] = Float8GetDatum(val);
    }
    //printf("end of relu_m_internal\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/* relu_m_derive   -   calculate the derivative of element-wise relu*/
Datum relu_m_derive(Datum input)
{
    //printf("begin of relu_m_derive\n");
    if (DatumGetPointer(input) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix relu derive: Null pointer passed as Matrix Inputs!")));
    }
    ArrayType *ret = copyArray(input);
    Datum *data = (Datum *)ARR_DATA_PTR(ret);
#pragma omp parallel for
    for (int i = 0; i < ArrayGetNItems(ARR_NDIM(ret), ARR_DIMS(ret)); i++)
    {
        float8 x = DatumGetFloat8(data[i]);
        float8 val = (x > 0) ? (1) : (0);
        data[i] = Float8GetDatum(val);
    }
    //printf("end of relu_m_derive\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/*
 * Create a new array, based on the given dimensions, including 
 * the data_array afterwards
 */
ArrayType *initResult(int ndims, int *dims, int *lbs)
{
    //printf("begin of initResult\n");
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
    //printf("end of initResult\n");
    return ret;
}

/*
 * Create a new array, copy of the given Array
 */
ArrayType *copyArray(Datum orgArray)
{
    //printf("begin of copyArray\n");
    //get size and dimensions from original
    if (DatumGetPointer(orgArray) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix Copy Array: Null pointer passed as Matrix Inputs!")));
    }
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
    //printf("end of copyArray\n");
    return ret;
}

/*
 * Create a new ArrayType, either filled with a given value, or the identity matrix
 * NOTICE: The identitymatrix will always be quadratic, but every other matrix does not have to be an can even be 1x3(vector)
 */
Datum createArray(int *dims, const float8 value, const bool identityMatrix)
{
    //printf("begin of createArray\n");
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
    //printf("end of createArray\n");
    PG_RETURN_ARRAYTYPE_P(ret);
}

/*
 * Create the seed for auto_diff algorithm
 */
Datum createSeedArray(Datum result) {
    if (DatumGetPointer(result) == NULL)
    {
        ereport(ERROR, (errmsg("Matrix createSeedArray: Null pointer passed as Matrix Inputs!")));
    }
    ArrayType *ret = copyArray(result);
#pragma omp parallel for
    for(int i = 0; i < ArrayGetNItems(ARR_NDIM(ret), ARR_DIMS(ret)); i++) {
        ARR_DATA_PTR(ret)[i] = Float8GetDatum(1.0);
    }
    PG_RETURN_ARRAYTYPE_P(ret);
}

/*
 * This creates a valid scalar in arrays representation(lbs[0] == -1)
 */
Datum createScalar(float8 value)
{
    int lbs[1], dims[1];
    lbs[0] = -1;
    dims[0] = 1;
    ArrayType *ret = initResult(1, dims, lbs);
    float8 *data = (float8 *)ARR_DATA_PTR(ret);
    data[0] = value;
    PG_RETURN_ARRAYTYPE_P(ret);
}

/*
 * Check if array is scalar(lbs[0] == -1)
 */
bool isScalar(ArrayType *in) {
    if (in == NULL)
    {
        ereport(ERROR, (errmsg("Matrix isScalar(): Null pointer passed to Check!")));
    }
    //Because C is C, we need an if, otherwise the postgres Bool gets converted
    if(ARR_LBOUND(in)[0] == -1) {
        return true;
    } else {
        return false;
    }
}

// ------------------------------Beginning of old version of mat mul ---------------------------------------

// // printf("begin of matrix_mul_external\n");
// //  The formal PostgreSQL array objects:
// ArrayType *a1, *a2, *ret;

// // The array element types:
// Oid type1, type2;

// // The array element type widths :
// int16 typeWidth1, typeWidth2;

// // The array element type "is passed by value" flags (not used, should always be true):
// bool typeByValue1, typeByValue2;

// // The array element type alignment codes (not used):
// char typeAlignmentCode1, typeAlignmentCode2;

// // The array contents, as PostgreSQL "datum" objects:
// float8 *retContent;

// // The size of each array:
// int length1, length2;

// if (PG_ARGISNULL(0))
// {
//     ereport(ERROR, (errmsg("Null arrays not accepted")));
// }

// // Extract the PostgreSQL arrays from the parameters passed to this function call.
// a1 = PG_GETARG_ARRAYTYPE_P(0);
// a2 = PG_GETARG_ARRAYTYPE_P(1);

// if (ARR_NDIM(a1) == 0 || ARR_NDIM(a2) == 0)
// {
//     PG_RETURN_NULL();
// }

// if (array_contains_nulls(a1))
// {
//     ereport(ERROR, (errmsg("Array left contains null elements")));
// }

// if (array_contains_nulls(a2))
// {
//     ereport(ERROR, (errmsg("Array right contains null elements")));
// }

// if (ARR_NDIM(a2) != 1 && ARR_NDIM(a2) != 2)
// {
//     ereport(ERROR, (errmsg("Array right dimensions wrong!")));
// }

// if (ARR_NDIM(a1) != 1 && ARR_NDIM(a1) != 2)
// {
//     ereport(ERROR, (errmsg("Array left dimensions wrong!")));
// }

// int *dim1 = ARR_DIMS(a1);
// int *dim2 = ARR_DIMS(a2);

// if (ARR_NDIM(a1) == 2 && dim1[1] != dim2[0])
// {
//     ereport(ERROR, (errmsg("Matrix dimensions do not match!")));
// }
// if (ARR_NDIM(a1) == 1 && dim1[0] != dim2[0])
// {
//     ereport(ERROR, (errmsg("Matrix dimensions do not match!")));
// }

// int lbs[2];
// for (int i = 0; i < 2; i++)
// {
//     lbs[i] = 1;
// }

// // // Determine the array element types.
// // type1 = ARR_ELEMTYPE(a1);
// // get_typlenbyvalalign(type1, &typeWidth1, &typeByValue1, &typeAlignmentCode1);
// // type2 = ARR_ELEMTYPE(a2);
// // get_typlenbyvalalign(type2, &typeWidth2, &typeByValue2, &typeAlignmentCode2);

// // if (type1 != FLOAT8OID)
// // {
// //     ereport(ERROR, (errmsg("Arrays must be SMALLINT, INTEGER, BIGINT, REAL, or DOUBLE PRECISION values")));
// // }
// // if (type2 != FLOAT8OID)
// // {
// //     ereport(ERROR, (errmsg("Arrays must be SMALLINT, INTEGER, BIGINT, REAL, or DOUBLE PRECISION values")));
// // }

// int dima = 1;
// if (ARR_NDIM(a1) == 2)
// {
//     dima = dim1[0];
// }

// int dimb = dim2[0];

// int dimc = 1;
// if (ARR_NDIM(a2) == 2)
// {
//     dimc = dim2[1];
// }

// int dims[2];
// dims[0] = dima;
// dims[1] = dimc;

// retContent = palloc0(sizeof(float8) * dima * dimc);
// double *arrayContent1f = (double *)palloc(sizeof(double) * dima * dimb);
// double *arrayContent2Tf = (double *)palloc(sizeof(double) * dimb * dimc);

// length1 = ArrayGetNItems(ARR_NDIM(a1), ARR_DIMS(a1));
// Datum *ps1 = (Datum *)ARR_DATA_PTR(a1);

// length2 = ArrayGetNItems(ARR_NDIM(a2), ARR_DIMS(a2));
// Datum *ps2 = (Datum *)ARR_DATA_PTR(a2);

// Datum *ps;
// #pragma omp parallel
// {
// #pragma omp single nowait
//     {
//         // Already prepare result array!
//         ret = initResult(2, dims, lbs);
//         ps = (Datum *)ARR_DATA_PTR(ret);
//     }

// #pragma omp for nowait
//     for (int pos = 0; pos < length1; pos++)
//     {
//         arrayContent1f[pos] = (double)DatumGetFloat8(ps1[pos]);
//     }
// #pragma omp for nowait
//     for (int pos = 0; pos < length2; pos++)
//     {
//         arrayContent2Tf[pos] = (double)DatumGetFloat8(ps2[pos]);
//     }
// #pragma omp for nowait
//     for (int pos = 0; pos < dima * dimc; pos++)
//     {
//         retContent[pos] = 0.0;
//     }
// #pragma omp barrier
//     int i, j, k;
// #pragma omp for
//     for (i = 0; i < dima; i++)
//     {
//         for (j = 0; j < dimb; j++)
//         {
//             for (k = 0; k < dimc; k++)
//             {
//                 retContent[MAT_2D(i, k, dimc)] += arrayContent1f[MAT_2D(i, j, dimb)] * arrayContent2Tf[MAT_2D(j, k, dimc)];
//             }
//         }
//     }

// #pragma omp barrier
// #pragma omp for
//     for (int i = 0; i < dima * dimc; i++)
//     {
//         ps[i] = Float8GetDatumFast(retContent[i]);
//     }
// }
// // printf("end of matrix_mul_external\n");
// PG_RETURN_ARRAYTYPE_P(ret);