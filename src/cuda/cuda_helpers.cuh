#pragma once

#include <cuda_bf16.h>

#include "helpers.h"

typedef __nv_bfloat16_raw bf16;

/*
  * +------------------------------------------------------------------------------+
  * |                             GLOBAL CONSTANTS                                 |
  * +------------------------------------------------------------------------------+
*/

constexpr uint8_t _CONSTANTS_WARP_SIZE = 32;

/*
  * +------------------------------------------------------------------------------+
  * |                             HELPER FUNCTIONS                                 |
  * +------------------------------------------------------------------------------+
*/

__device__ inline bool is_aligned(const void* addr, const size_t alignment_bytes)
{
	return (reinterpret_cast<uintptr_t>(addr) & (alignment_bytes - 1)) == 0;
}

__device__ inline uint8_t align(const void* base, const void* addr, const size_t alignment_bytes)
{
	const uintptr_t offset = reinterpret_cast<uintptr_t>(addr) - reinterpret_cast<uintptr_t>(base);
	const uintptr_t aligned_offset = (reinterpret_cast<uintptr_t>(offset) + (alignment_bytes - 1)) & ~size_t(alignment_bytes - 1);
	return reinterpret_cast<uintptr_t>(base) + aligned_offset;
}

__device__ inline f32 _d_dn_rm_get(const f32* const a, u32 n_cols, u32 row, u32 col)
{
	return a[row * n_cols + col];
}

__device__ inline f32 _d_dn_cm_get(const f32* const a, u32 n_rows, u32 row, u32 col)
{
	return a[col * n_rows + row];
}

__device__ inline void _d_dn_rm_set(f32* const a, u32 n_cols, u32 row, u32 col, f32 val)
{
	a[row * n_cols + col] = val;
}

__device__ inline void _d_dn_cm_set(f32* const a, u32 n_rows, u32 row, u32 col, f32 val)
{
	a[col * n_rows + row] = val;
}

__host__ inline bool _is_dev_ptr(const void* ptr)
{
	cudaPointerAttributes a;
	cudaPointerGetAttributes(&a, ptr);
	return a.type == cudaMemoryTypeDevice;
}
