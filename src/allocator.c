#include "allocator.h"

#include <stdlib.h>  // for abort()
#include <sys/mman.h>
#include <unistd.h>

#include "helpers.h"

#ifndef NDEBUG
#include <stdio.h>
#endif

/*
 * +------------------------------------------------------------------------------+
 * |                             PLATFORM SPECIFIC                                |
 * +------------------------------------------------------------------------------+
 */

#if defined(__linux__)

static u32 vm_get_page_size(void) { return (u32)sysconf(_SC_PAGESIZE); }

// INFO: Mimics malloc in the sense that it returns a NULL ptr on an error
// instead of an error enum type
static void* vm_reserve(const u64 size)
{
	void* ptr = mmap(NULL, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (ptr == MAP_FAILED) {
		return NULL;
	}
	return ptr;
}

static i32 vm_release(void* ptr, const u64 size)
{
	return munmap(ptr, size) == 0; /* >"It is not an error if the indicated range does not contain any mapped pages" ~ So 'ptr' can be NULL here.*/
}

static i32 vm_commit(void* addr, const u64 size)
{
	return mprotect(addr, size, PROT_READ | PROT_WRITE) == 0;
}

static i32 vm_uncommit(void* addr, const u64 size)
{
	i32 ret_code = mprotect(addr, size, PROT_NONE);
	if (ret_code != 0) {
		return -1;
	}
	return madvise(addr, size, MADV_DONTNEED) == 0; /* Subsequent access will result in zero-fill-on-demand pages */
}

Error_t exec_ctx_create(ExecCtx** ctx)
{
	if (*ctx) {  // already exists
		return ErrorAlreadyInitialized;
	}

	Error_t error = arena_host_create((HostArena**)(ctx), GIB(1), MIB(1));
	if (error != Success) {
		return error;
	}

	error = arena_host_push((HostArena*)(*ctx), sizeof(*ctx)->dev_arena, (void**)&(*ctx)->dev_arena);
	if (error != Success) {
		return error;
	}
	(*ctx)->dev_arena._d_ptr = NULL;

	return Success;
}

Error_t exec_ctx_destroy(ExecCtx** ctx)
{
	if (!(*ctx)) {  // doesn't exist
		return ErrorInvalidValue;
	}

	Error_t error = Success;
	// Free the device memory first
	// if for some reason we freed the host memory first then we'd have no handle to the device chunk
	if ((*ctx)->dev_arena._d_ptr) {
		error = arena_dev_destroy(&(*ctx)->dev_arena);
		if (error != Success) {
			return error;
		}
	}

	error = arena_host_destroy((HostArena*)(*ctx));
	if (error != Success) {
		return error;
	}

	*ctx = NULL;
	return Success;
}

#else
#error "VIRTUAL MEMORY ALLOCATION NOT IMPLEMENTED FOR CURRENT PLATFORM"
#endif

/*
 * +------------------------------------------------------------------------------+
 * |                                INTERNALS                                     |
 * +------------------------------------------------------------------------------+
 */

// TODO: Implement an internal error enum and change the return types of these
// functions
//
// INFO: COMMIT SIZE SHOULD BE PAGE-SIZE ALIGNED AND DERIVED FROM AN ALLOCATION
// STRATEGY SIMILAR TO VECTOR OR SOMIN :)

Error_t arena_host_create(HostArena** const arena,
	const u64                               reserve_size,
	const u64                               commit_size)
{
	// TODO: Debug print these at some point to ensure correctness.
	const u32 page_size = vm_get_page_size();
	const u64 pa_reserve_size = reserve_size + PADDING_POW2(reserve_size, page_size);  // pa = page aligned
	const u64 pa_commit_size = commit_size + PADDING_POW2(commit_size, page_size);

	*arena = (HostArena*)vm_reserve(reserve_size);
	if (!(*arena)) {
		return ErrorMemoryAllocation;
	}

	if (!vm_commit(*arena, pa_commit_size)) { /* Allocate for the HostArena members */
		return ErrorMemoryAllocation;
	}

	(*arena)->reserve_size = pa_reserve_size;
	(*arena)->commit_size = pa_commit_size;

	(*arena)->commit_pos = pa_commit_size;
	(*arena)->pos = sizeof **arena;

	return Success;
}

Error_t arena_host_destroy(HostArena* arena)
{
	if (!vm_release(arena, arena->reserve_size)) {
		return ErrorMemoryAllocation;
	}
	arena = NULL;
	return Success;
}

Error_t arena_host_push(HostArena* const arena, const u64 req_size, void** ptr_out)
{
	const u64 aligned_pos = arena->pos + PADDING_POW2(arena->pos, sizeof(void*)); /* the pointer returned should be naturally aligned */
	const u64 new_pos = aligned_pos + req_size;

	if (new_pos > arena->reserve_size) {
		return ErrorArenaOutOfMemory;
	} else if (new_pos > arena->commit_pos) {
		u64 new_commit_pos = new_pos;
		new_commit_pos += arena->commit_size - 1;
		new_commit_pos -= new_commit_pos % arena->commit_size;
		new_commit_pos = MIN(new_commit_pos, arena->reserve_size);

		u8* mem = (u8*)arena + arena->commit_pos;
		u64 commit_size = new_commit_pos - arena->commit_pos;
		if (!vm_commit(mem, commit_size)) {
			return ErrorMemoryAllocation;
		}
		arena->commit_pos += arena->commit_size;
	}

	*ptr_out = (u8*)arena + aligned_pos;
	arena->pos = new_pos;

	return Success;
}

void arena_host_pop(HostArena* const arena, u64 size)
{
	// TODO: Should I null check the ptr here?
	size = MIN(size, arena->pos - sizeof *arena); /* don't dealloc MemArena members */
	arena->pos -= size;
}

void arena_host_pop_at(HostArena* const arena, u64 pos)
{
	u64 size = pos < arena->pos ? arena->pos - pos : 0;
	arena_host_pop(arena, size);
}

// TODO: Do I need this anymore?
u64 mem_arena_host_pos_get(const HostArena* const arena) { return arena->pos; }
