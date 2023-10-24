#include <cuda.h>

#include <stdio.h>
#include <stdlib.h>
#include <cstring>

int main(int argc, char** argv) {
  // Verifica se o parâmetro -N ou --number foi fornecido
  if (argc < 2 || (strcmp(argv[1], "-N") != 0 && strcmp(argv[1], "--number") != 0)) {
    printf("Uso: %s [-N | --number] <números>\n", argv[0]);
    return 1;
  }

  // Lê os números do input
  int* numbers = (int*)malloc(sizeof(int) * argc - 2);
  for (int i = 2; i < argc; i++) {
    numbers[i - 2] = atoi(argv[i]);
  }

  // Calcula o hash dos números
  cuda_sha256_t cuda;
  cuda_calculate_sequence(numbers, argc - 2, &cuda);

  // Imprime o hash
  printf("Hash: %s\n", hash_get_hash_str(&cuda));

  free(numbers);
  return 0;
}
