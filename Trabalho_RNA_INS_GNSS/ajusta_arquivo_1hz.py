import pandas as pd

def downsample_and_transpose_csv(input_file, output_file):
    # Lê o arquivo CSV de entrada usando o pandas
    df = pd.read_csv(input_file, delimiter=';')

    # Seleciona a cada 100 linhas (downsampling)
    downsampled_df = df.iloc[::100]

    # Transpõe os dados usando o método 'T' de pandas
    transposed_df = downsampled_df.T

    # Remove o índice antes de escrever no novo arquivo CSV
    transposed_df.reset_index(drop=True, inplace=True)

    # Escreve os dados transpostos no novo arquivo CSV
    transposed_df.to_csv(output_file, sep=';', index=False)

    print(f"Transposição concluída. Resultados salvos em {output_file}")

# Substitua 'acelerometro2.csv' pelo nome do seu arquivo CSV de entrada
# Substitua 'acelerometro2_1hz.csv' pelo nome desejado do arquivo de saída transposto
downsample_and_transpose_csv('inprofile_100hz.csv', 'inprofile_1hz.csv')