#if !defined(HELPERS_H)
#define HELPERS_H

#include <stdint.h>
#include <stdio.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CEIL_DIVI(m, n) (((m) + (n) - 1) / (n))
#define PADDING_POW2(n, p) ((((p) - ((n) & ((p) - 1))) & ((p) - 1)))
#define GIB(n) ((u64)(n) << 30)
#define MIB(n) ((u64)(n) << 20)
#define KIB(n) ((u64)(n) << 10)
#define LOWER_BITS_MASK(n) (((1u) << (n)) - 1)
#define MOD_POW2(n, p) ((n) & ((p) - 1))

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t   i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef int64_t  i64;
typedef float    f32;
typedef double   f64;

typedef i32 b32;

typedef enum Error_t
{
	Success = 0,
	ErrorMemoryAllocation = 1,
	ErrorArenaOutOfMemory = 2,
	ErrorAlreadyInitialized = 3,
	ErrorInvalidValue = 4,
	ErrorInvalidHostPtr = 5,
	ErrorInvalidDevPtr = 6,
	ErrorGeneric,

} Error_t;

#define CHECK_CUDA(x)                                                            \
	do {                                                                         \
		cudaError_t err = x;                                                     \
		if (err != cudaSuccess) {                                                \
			fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __func__, \
				__FILE__, __LINE__, cudaGetErrorString(err),                     \
				cudaGetErrorName(err), err);                                     \
			abort();                                                             \
		}                                                                        \
	} while (0)

#define CHECK_CUSPARSE(x)                                                  \
	do {                                                                   \
		cusparseStatus_t err = x;                                          \
		if (err != CUSPARSE_STATUS_SUCCESS) {                              \
			fprintf(stderr, "CUSPARSE error in %s at %s:%d: %s (%s=%d)\n", \
				__func__, __FILE__, __LINE__, cusparseGetErrorString(err), \
				cusparseGetErrorName(err), err);                           \
			abort();                                                       \
		}                                                                  \
	} while (0)

#define CHECK_ERROR(x)                                               \
	do {                                                             \
		Error_t err = x;                                             \
		if (err != Success) {                                        \
			fprintf(stderr, "Error in %s at %s:%d: %d)\n", __func__, \
				__FILE__, __LINE__, err);                            \
			abort();                                                 \
		}                                                            \
	} while (0)

#endif  // HELPERS_H
