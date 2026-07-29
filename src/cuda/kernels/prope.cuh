#pragma once

#include <cuda_bf16.h>

#include "helpers.h"

typedef __nv_bfloat16 bf16;

/**
 * This kernel gets its own separate file
 * such that I can later improve on it
 * and keep an archive of older versions
**/

__global__ void k_prope(bf16* const q, bf16* const k, const u32 p);
