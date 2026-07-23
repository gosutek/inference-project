#pragma once

#include "cublas_v2.h"

#include "allocator.h"

void testutils_create_cublas_tenv(ExecCtx** e_ctx, cublasHandle_t* cublas_handle);
void testutils_destroy_cublas_tenv(ExecCtx** e_ctx, cublasHandle_t* cublas_handle);
