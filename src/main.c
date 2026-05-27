#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#include "cJSON.h"

#include "allocator.h"
#include "helpers.h"

static i32 st_get_bsize(const char* safetensor_path, u64* bsize)
{
	struct stat st;
	if (stat(safetensor_path, &st) == 0) {
		*bsize = (u64)st.st_size;
		return 1;
	}
	return 0;
}

int main(void)
{
	const char* model_filepath = "gemma-4-E4B-it/model.safetensors";
	u64         st_bsize = 0;  // total file size of the safetensor in bytes
	if (st_get_bsize(model_filepath, &st_bsize) != 1) {
		fprintf(stderr, "couldn't read file size of %s\n", model_filepath);
		exit(EXIT_FAILURE);
	}

	FILE* file = fopen(model_filepath, "r");
	if (!file) {
		fprintf(stderr, "couldn't load %s\n", model_filepath);
		exit(EXIT_FAILURE);
	}

	u64 header_bsize = 0;  // header size of the safetensor in bytes
	if (fread(&header_bsize, sizeof header_bsize, 1, file) != 1) {
		fprintf(stderr, "failed read\n");
		exit(EXIT_FAILURE);
	}

	const u64 model_bsize = st_bsize - header_bsize - sizeof header_bsize;
	fclose(file);

	return 0;
}
