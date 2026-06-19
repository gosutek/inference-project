#if !defined(CU_ALLOCATOR_CUH)
#define CU_ALLOCATOR_CUH

#include "helpers.h"

#if defined(__cplusplus)
extern "C"
{
#endif

	typedef struct DevArena
	{
		u8* _d_ptr;

		u64 size;
		u64 pos;
	} DevArena;

	Error_t arena_dev_create(DevArena* const arena, const u64 bsize);
	Error_t arena_dev_destroy(DevArena* arena);

	Error_t arena_dev_push(DevArena* const arena, const u64 bsize, void** ptr_out);
	void    mem_arena_dev_pop(DevArena* const arena, u64 bsize);
	void    mem_arena_dev_pop_at(DevArena* const arena, u64 pos);

	u64 mem_arena_dev_pos_get(const DevArena* const arena);

#if defined(__cplusplus)
}
#endif

#endif  // CU_ALLOCATOR_CUH
