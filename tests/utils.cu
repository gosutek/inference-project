#include <iostream>

#include "helpers.h"

#include "utils.cuh"

constexpr const f64 ATOL = 1e-7;
constexpr const f64 RTOL = 1e-3;

void testutils_create_cublas_tenv(ExecCtx** e_ctx, cublasHandle_t* cublas_handle)
{
	CHECK_ERROR(exec_ctx_create(e_ctx));
	CHECK_CUBLAS(cublasCreate(cublas_handle));

	return;
}

void testutils_destroy_cublas_tenv(ExecCtx** e_ctx, cublasHandle_t* cublas_handle)
{
	CHECK_ERROR(exec_ctx_destroy(e_ctx));
	CHECK_CUBLAS(cublasDestroy(*cublas_handle));

	return;
}

bool comparef(const f32 a, const f32 b)
{
	if (std::isnan(a) || std::isnan(b)) {
		std::cout << "isnan: " << a << " or " << b << std::endl;
		return false;
	}
	if (a == b) {
		return true;
	}

	const f64 abs_diff = std::fabs(a - b);
	const f64 tol = ATOL + RTOL * std::fabs(static_cast<f64>(b));
	if (std::isfinite(abs_diff) && abs_diff <= tol) {
		return true;
	}

	std::cout << "Not close: " << a << " | " << b << std::endl;
	return false;
}
