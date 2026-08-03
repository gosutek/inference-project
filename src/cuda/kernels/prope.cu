#include "prope.cuh"

#include "cuda_helpers.cuh"
#include "helpers.h"

constexpr const f32 THETA = 1e4f;
constexpr const f32 PERC = 0.25f;
constexpr const u32 TILE_PREF_SIZE = 32;

/*
 * +------------------------------------------------------------------------------+
 * |                                  KERNELS                                     |
 * +------------------------------------------------------------------------------+
 */
__global__ void __prope_kernel(bf16* const x, const u32 model_dim, const u32 input_sequence_len, const u32 num_of_active_pairs)
{
	const u32 stride = 2;
	const u32 i = threadIdx.x * stride + blockIdx.x * blockDim.x;
	if (i > num_of_active_pairs) {
		return;
	}

	const u32 p = threadIdx.y + blockIdx.y * blockDim.y;
	if (p > input_sequence_len - 1) {  // -1 cause input_sequence_len is 1-index while p is 0-indexed.
		return;
	}

	const f32 exp = (2.0f * (f32)i) / (f32)model_dim;
	const f32 omega = 1.0f / (std::pow(THETA, exp));
	const f32 trig = omega * p;

	const bf16 rot_mat[4] = {
		(bf16)std::cos(trig),
		(bf16)std::sin(trig),
		(bf16)-std::sin(trig),
		(bf16)std::cos(trig)
	};

	const bf16 x1 = _d_dn_rm_get(x, model_dim, input_sequence_len, i);
	const bf16 x2 = _d_dn_rm_get(x, model_dim, input_sequence_len, i + 1);

	const bf16 res1 = x1 * rot_mat[0] + x2 * rot_mat[1];
	const bf16 res2 = x1 * rot_mat[2] + x2 * rot_mat[3];

	_d_dn_rm_set(x, model_dim, input_sequence_len, i, res1);
	_d_dn_rm_set(x, model_dim, input_sequence_len, i + 1, res2);
}

// TODO:
// Next kernel premise:
// 1 thread for 4 muls and 2 additions is too little
// **Make the 32x32 tile virtual across the y axis**.

/*
 * +------------------------------------------------------------------------------+
 * |                                  WRAPPER                                     |
 * +------------------------------------------------------------------------------+
 */
Error_t k_prope(bf16* const x, const u32 input_sequence_len, const u32 model_dim, cudaStream_t stream)
{
	if (MOD_POW2(model_dim, 2) != 0) {
		return ErrorUnexpectedValue;
	}

	const u32 num_of_pairs = model_dim / 2;
	const u32 num_of_active_pairs = num_of_pairs * PERC;

	const u32 grid_size_x = CEIL_DIVI(num_of_active_pairs, TILE_PREF_SIZE);
	const u32 grid_size_y = CEIL_DIVI(input_sequence_len, TILE_PREF_SIZE);

	const dim3 grid_size = { grid_size_x, grid_size_y };
	const dim3 block_size = { TILE_PREF_SIZE, TILE_PREF_SIZE };

	__prope_kernel<<<grid_size, block_size, 0, stream>>>(x, model_dim, input_sequence_len, num_of_active_pairs);

	return Success;
}
