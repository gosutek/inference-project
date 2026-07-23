#include <cublas_v2.h>

#include "../utils.cuh"
#include "cuda_helpers.cuh"
#include "helpers.h"

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

static void run_k_gemmt()
{
}

static void run_cublas_ref()
{}

int main(void)
{
	ExecCtx*       e_ctx = nullptr;
	cublasHandle_t cublas_handle;
	testutils_create_cublas_tenv(&e_ctx, &cublas_handle);

	bf16*     a = nullptr;
	bf16*     b = nullptr;
	const u64 a_bsize = ROWS * COLS * sizeof *a;

	arena_host_push((HostArena*)e_ctx, a_bsize, (void**)&a);
	arena_host_push((HostArena*)e_ctx, a_bsize, (void**)&b);

	gen_dn_rm_bf16(a, 69);
	gen_dn_rm_bf16(b, 69);

	testutils_destroy_cublas_tenv(&e_ctx, &cublas_handle);

#ifndef NDEBUG
	if (e_ctx) {
		printf("[WARNING] e_ctx is not null\n");
	}
#endif
}
