#pragma once

#include "helpers.h"
#include <cuda_bf16.h>

typedef __nv_bfloat16 bf16;

Error_t k_gemm();
Error_t k_gemmt(const bf16* const a, const bf16* const b, bf16* const c, const u32 hidden_size, const cudaStream_t stream, const u32 input_seq_len, const u32 n);
