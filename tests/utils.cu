#include "helpers.h"

#include "utils.cuh"

void testutils_create_cublas_tenv(ExecCtx** e_ctx, cublasHandle_t* cublas_handle)
{
	exec_ctx_create(e_ctx);
	CHECK_CUBLAS(cublasCreate(cublas_handle));

	return;
}

void testutils_destroy_cublas_tenv(ExecCtx** e_ctx, cublasHandle_t* cublas_handle)
{
	exec_ctx_destroy(e_ctx);
	CHECK_CUBLAS(cublasDestroy(*cublas_handle));

	return;
}
