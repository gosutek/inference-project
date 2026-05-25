#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "cJSON.h"

#include "helpers.h"

void read_safetensor(char *safetensor_path) {
  FILE *file = fopen(safetensor_path, "r");
  if (!file) {
    fprintf(stderr, "couldn't load %s\n", safetensor_path);
    exit(EXIT_FAILURE);
  }

  u64 header_len;
  if (fread(&header_len, sizeof(uint64_t), 1, file) != 1) {
    fprintf(stderr, "failed read\n");
  }
  printf("Header size = %lu\n", header_len);
  fclose(file);
}

int main() {
  read_safetensor("gemma-4-E4B-it/model.safetensors");
  return 0;
}
