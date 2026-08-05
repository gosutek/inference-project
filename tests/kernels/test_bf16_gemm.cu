#include <cublas_v2.h>

#include "../utils.cuh"
#include "cuda_helpers.cuh"
#include "cuda_mem_wrapper.cuh"
#include "helpers.h"
#include "kernels/gemm.cuh"

constexpr const u32 ROWS = 512;
constexpr const u32 COLS = 512;

static void gen_dn_rm_bf16(bf16* a, const u32 seed)
{
	srand(seed);
	for (u32 i = 0; i < ROWS * COLS; ++i) {
		const f32 v = ((f32)rand() / (f32)RAND_MAX) * 2.0f - 1.0f;  // NOTE: Don't really get how but this gives random numbers in the inclusive range[-1,1]
		a[i] = __float2bfloat16(v);
	}
}

static void run_cublas_ref(cublasHandle_t* cublas_handle, const bf16* const a, const bf16* const b, bf16* const c)
{
	const f32 alpha = 1.0f;
	const f32 beta = 0.0f;

	CHECK_CUBLAS(cublasGemmEx(*cublas_handle,
		CUBLAS_OP_T, CUBLAS_OP_N,
		ROWS, ROWS, ROWS,
		&alpha,
		b, CUDA_R_16BF, ROWS,
		a, CUDA_R_16BF, ROWS,
		&beta,
		c, CUDA_R_16BF, ROWS,
		CUBLAS_COMPUTE_32F,
		CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

int main(void)
{
	ExecCtx*       e_ctx = nullptr;
	cublasHandle_t cublas_handle;
	testutils_create_cublas_tenv(&e_ctx, &cublas_handle);

	bf16*     a = nullptr;
	bf16*     b = nullptr;
	bf16*     c = nullptr;
	bf16*     c_ref = nullptr;
	const u64 a_bsize = ROWS * COLS * sizeof *a;

	CHECK_ERROR(arena_host_push((HostArena*)(e_ctx), a_bsize, (void**)&a));
	CHECK_ERROR(arena_host_push((HostArena*)(e_ctx), a_bsize, (void**)&b));
	CHECK_ERROR(arena_host_push((HostArena*)(e_ctx), a_bsize, (void**)&c));
	CHECK_ERROR(arena_host_push((HostArena*)(e_ctx), a_bsize, (void**)&c_ref));

	gen_dn_rm_bf16(a, 69);
	gen_dn_rm_bf16(b, 42);

	CHECK_ERROR(arena_dev_create(&e_ctx->dev_arena, GIB(1)));

	bf16* d_a = nullptr;

	CHECK_ERROR(arena_dev_push(&e_ctx->dev_arena, a_bsize * 3, (void**)&d_a));
	bf16* d_b = (bf16*)((u8*)d_a + a_bsize);
	bf16* d_c = (bf16*)((u8*)d_b + a_bsize);

	CHECK_ERROR(cu_memcpy_htd(d_a, a, a_bsize));
	CHECK_ERROR(cu_memcpy_htd(d_b, b, a_bsize));

	k_gemmt(d_a, d_b, d_c, ROWS, NULL, ROWS, ROWS);
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_ERROR(cu_memcpy_dth(c, d_c, a_bsize));

	run_cublas_ref(&cublas_handle, a, b, c_ref);
	CHECK_CUDA(cudaDeviceSynchronize());
	CHECK_ERROR(cu_memcpy_dth(c_ref, d_c, a_bsize));

	for (u32 i = 0; i < ROWS * COLS; ++i) {
		if (!comparef(c[i], c_ref[i])) {
			testutils_destroy_cublas_tenv(&e_ctx, &cublas_handle);
			return -1;
		}
	}

#ifndef NDEBUG
	if (e_ctx) {
		printf("[WARNING] e_ctx is not null\n");
	}
#endif
	return 0;
}
