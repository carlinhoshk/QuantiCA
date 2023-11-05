#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

#include <sha256.h>


// Gera um hash SHA256 do dado de entrada
__global__ void sha256(const unsigned char *data, unsigned char *hash) {
  // Obtém o bloco de memória no dispositivo
  unsigned char *d_data = (unsigned char *)malloc(sizeof(unsigned char) * 64);
  cudaMemcpy(d_data, data, sizeof(unsigned char) * 64, cudaMemcpyHostToDevice);

  // Gera o hash
  sha256_state_t state;
  sha256_init(&state);
  sha256_update(&state, d_data, 64);
  sha256_final(&state, hash);

  // Libera a memória no dispositivo
  cudaFree(d_data);
}

int main() {
  // Obtém o dado de entrada
  unsigned char *data = (unsigned char *)malloc(sizeof(unsigned char) * 1024);
  for (int i = 0; i < 1024; i++) {
    data[i] = rand() % 256;
  }

  // Gera o hash
  unsigned char hash[32];
  cudaDeviceSynchronize();
  sha256<<<1, 1>>>(data, hash);
  cudaDeviceSynchronize();

  // Imprime o hash
  for (int i = 0; i < 32; i++) {
    printf("%02x", hash[i]);
  }
  printf("\n");

  // Libera a memória
  free(data);

  return 0;
}