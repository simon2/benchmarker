#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <omp.h>
#include "config.h"

int main() {    
    t_float *A, *B, *C;
    double dtime;
    size_t i;
    // TODO: move this to shared module
    printf("doing clbas\n");
    A = (t_float*) malloc(sizeof(t_float)*n*n);
    B = (t_float*) malloc(sizeof(t_float)*n*n);
    C = (t_float*) malloc(sizeof(t_float)*n*n);
    for(i=0; i<n*n; i++) { A[i] = rand()/RAND_MAX; B[i] = rand()/RAND_MAX;}
    printf("done random init\n");

    dtime = omp_get_wtime();
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,n,n,n, 1, A, n, B, n, 1, C ,n);
    dtime = omp_get_wtime() - dtime;
    double gflop = (2.0 * n * n * n) / (1024 * 1024 * 1024);
    double gflops = gflop / dtime;
    printf("time: \t%f\n", dtime);
    printf("gflops/s: \t%f\n", gflops);
}
