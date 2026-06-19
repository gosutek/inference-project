#include "cuda_allocator.cuh"
#include "helpers.h"

Error_t arena_dev_create(DevArena* arena, const u64 bsize)
{
	cudaPointerAttributes ptr_attr;
	cudaPointerGetAttributes(&ptr_attr, arena->_d_ptr);

	if (ptr_attr.type == cudaMemoryTypeDevice && ptr_attr.devicePointer != NULL) {
		return ErrorAlreadyInitialized;
	}

	CHECK_CUDA(cudaMalloc(&arena->_d_ptr, bsize));

	arena->size = bsize;
	arena->pos = sizeof *arena;

	return Success;
}

Error_t arena_dev_destroy(DevArena* arena)
{
	if (!arena->_d_ptr) {
		return ErrorInvalidValue;
	}

	CHECK_CUDA(cudaFree(arena->_d_ptr));
	arena->_d_ptr = nullptr;

	return Success;
}

Error_t arena_dev_push(DevArena* const arena, const u64 bsize, void** ptr_out)
{
	if (!arena) {
		return ErrorInvalidValue;
	}

	const u64 pos_aligned = arena->pos + PADDING_POW2(arena->pos, sizeof(void*));
	const u64 new_pos = pos_aligned + bsize;

	if (new_pos > arena->size) {
		return ErrorArenaOutOfMemory;
	}

	*ptr_out = arena->_d_ptr + pos_aligned;
	arena->pos = new_pos;

	return Success;
}

// WARN: What if bsize isn't aligned?
void mem_arena_dev_pop(DevArena* const arena, u64 bsize)
{
	bsize = MIN(bsize, arena->pos - sizeof *arena);
	arena->pos -= bsize;
}

void mem_arena_dev_pop_at(DevArena* const arena, u64 pos)
{
	u64 size = pos < arena->pos ? arena->pos - pos : 0;
	mem_arena_dev_pop(arena, size);
}

u64 mem_arena_dev_pos_get(const DevArena* const arena)
{
	return arena->pos;
}
