#include "cuda_helpers.cuh"
#include "helpers.h"

#if defined(__cplusplus)
extern "C"
{
#endif

	b32 cu_malloc(void* dev_ptr, const u64 bsize)
	{
		if (!_is_dev_ptr(dev_ptr)) {
			return 0;
		}
		CHECK_CUDA(cudaMalloc(&dev_ptr, bsize));
		return 1;
	}

	b32 cu_free(void* dev_ptr)
	{
		if (!_is_dev_ptr(dev_ptr)) {
			return 0;
		}
		CHECK_CUDA(cudaFree(dev_ptr));
		return 1;
	}

	Error_t cu_memcpy_htd(void* dst, const void* src, const u64 bsize)
	{
		if (!_is_dev_ptr(dst)) {
			return ErrorInvalidDevPtr;
		}
		CHECK_CUDA(cudaMemcpy(dst, src, bsize, cudaMemcpyHostToDevice));
		return Success;
	}

	b32 cu_memcpy_htd_async(void* dst, const void* src, const u64 bsize, cudaStream_t stream)
	{
		if (!_is_dev_ptr(dst)) {
			return 0;
		}
		CHECK_CUDA(cudaMemcpyAsync(dst, src, bsize, cudaMemcpyHostToDevice, stream));
		return 1;
	}

	b32 cu_memcpy_dth(void* dst, const void* src, const u64 bsize)
	{
		if (!_is_dev_ptr(src)) {
			return 0;
		}
		CHECK_CUDA(cudaMemcpy(dst, src, bsize, cudaMemcpyDeviceToHost));
		return 1;
	}

	b32 cu_host_register_read_only(void* ptr, const u64 bsize)
	{
		if (_is_dev_ptr(ptr)) {
			return 0;
		}
		CHECK_CUDA(cudaHostRegister(ptr, bsize, cudaHostRegisterReadOnly));
		return 1;
	}

	b32 cu_memset(void* s, i32 c, u64 n)
	{
		if (!_is_dev_ptr(s)) {
			return 0;
		}
		CHECK_CUDA(cudaMemset(s, c, n));
		return 1;
	}

#if defined(__cplusplus)
}
#endif
