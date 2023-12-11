import csv

def manter_apenas_uma_linha_a_cada_cem_epocas(entrada, saida):
    with open(entrada, 'r', newline='') as arquivo_entrada, \
         open(saida, 'w', newline='') as arquivo_saida:

        leitor = csv.reader(arquivo_entrada)
        escritor = csv.writer(arquivo_saida)

        # Copiar cabeçalhos para o arquivo de saída
        cabeçalhos = next(leitor)
        escritor.writerow(cabeçalhos)

        # Inicializar contador de linha e índice de época
        contador_linha = 0
        indice_epoca = 0

        for linha in leitor:
            # Manter apenas uma linha a cada 100 épocas
            if contador_linha == 0:
                escritor.writerow(linha)

            contador_linha += 1

            # Resetar o contador de linha e incrementar o índice de época
            if contador_linha == 100:
                contador_linha = 0
                indice_epoca += 1

if __name__ == "__main__":
    arquivo_entrada = 'out_errors.csv'
    arquivo_saida = 'out_errors_1hz.csv'

    manter_apenas_uma_linha_a_cada_cem_epocas(arquivo_entrada, arquivo_saida)