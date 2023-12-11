import pandas as pd

def transpose_csv(input_file, output_file):
    # Lê o arquivo CSV de entrada usando o pandas
    df = pd.read_csv(input_file, delimiter=';')

    # Transpõe os dados usando o método 'T' de pandas
    transposed_df = df.T

    # Escreve os dados transpostos no novo arquivo CSV
    transposed_df.to_csv(output_file, sep=';', index=False)

    print(f"Transposição concluída. Resultados salvos em {output_file}")

# Substitua 'acelerometro2.csv' pelo nome do seu arquivo CSV de entrada
# Substitua 'acelerometro2_1hz.csv' pelo nome desejado do arquivo de saída transposto
transpose_csv('gnss.csv', 'gnss_ajustado.csv')