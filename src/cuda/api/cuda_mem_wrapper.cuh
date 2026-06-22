#pragma once

#include "helpers.h"

#if defined(__cplusplus)
extern "C"
{
#endif

	b32     cu_malloc(void* dev_ptr, const u64 bsize);
	b32     cu_free(void* dev_ptr);
	Error_t cu_memcpy_htd(void* dst, const void* src, const u64 bsize);
	b32     cu_memcpy_htd_async(void* dst, const void* src, const u64 bsize, cudaStream_t stream);
	b32     cu_memcpy_dth(void* dst, const void* src, const u64 bsize);
	b32     cu_host_register_read_only(void* ptr, const u64 bsize);
	b32     cu_memset(void* s, i32 c, u64 n);

#if defined(__cplusplus)
}
#endif
