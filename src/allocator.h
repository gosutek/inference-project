#if !defined(ALLOCATOR_H)
#define ALLOCATOR_H

#include "cuda_allocator.cuh"
#include "helpers.h"

#if defined(__cplusplus)
extern "C"
{
#endif

	typedef struct HostArena
	{
		u64 reserve_size;
		u64 commit_size;

		u64 commit_pos;
		u64 pos;
	} HostArena;

	typedef struct ExecCtx
	{
		HostArena host_arena;
		DevArena  dev_arena;
	} ExecCtx;

	Error_t exec_ctx_create(ExecCtx** ctx);
	Error_t exec_ctx_destroy(ExecCtx** ctx);
	Error_t arena_host_create(HostArena** const arena,
		const u64                               reserve_size,
		const u64                               commit_size);
	Error_t arena_host_destroy(HostArena* arena);

	Error_t arena_host_push(HostArena* const arena, const u64 req_size, void** ptr_out);
	void    arena_host_pop(HostArena* const arena, u64 size);
	void    arena_host_pop_at(HostArena* const arena, u64 pos);

	u64 mem_arena_host_pos_get(const HostArena* const arena);

#if defined(__cplusplus)
}
#endif

#endif  // ALLOCATOR_H
