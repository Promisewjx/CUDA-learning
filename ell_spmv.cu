#include<iostream>
#include<vector>
#include <cuda_runtime.h>
#include <cassert>

__global__ void spmv_ell_kernel(
    const float *values,const int *col_idx,
    const float *x,float *y,
    int num_rows,int max_nnz_per_row
){
    int row = blockIdx.x * x blockIdx
}