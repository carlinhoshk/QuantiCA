#include <iostream>
#include <cstring>
#include <openssl/sha.h>

int main() {
    const char *data = "123456789";
    unsigned char hash[SHA256_DIGEST_LENGTH];

    SHA256((const unsigned char *)data, strlen(data), hash);

    std::cout << "SHA-256 Hash: ";
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        printf("%02x", hash[i]);
    }
    std::cout << std::endl;

    return 0;
}
