import random

talvez_resultados = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

while True:
    numero_sorteado = random.choice(talvez_resultados)  # Sortear um número aleatório da lista
    print(f'Número sorteado: {numero_sorteado}')

    if numero_sorteado == 10:
        print('certo')
        break

