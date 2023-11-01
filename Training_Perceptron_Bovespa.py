import numpy as np
import random
import matplotlib.pyplot as plt
import requests
import json

# Define a função 'yAdaline' que retorna a saída do Adaline
def yAdaline(Weight, bias, X_inputs):
    y_output = np.dot(Weight, X_inputs) + bias
    return y_output

# Define a função 'treina_adaline' para treinar o Adaline
def treina_adaline(W, b, X, yd, alfa, maxEpocas, tol):
    N = len(X)  # Número de amostras de entrada

    SEQ = tol
    VetorSEQ = []  # Lista para armazenar a soma dos erros quadrados para cada época

    Epoca = 1

    while (Epoca <= maxEpocas) and (SEQ >= tol):  # Continue o treinamento até maxEpocas ou até atingir a tolerância
        SEQ = 0  # Inicializa a soma dos erros quadrados para a época atual

        for i in range(N):  # Loop através de cada amostra de entrada
            X_input = X[i]
            yi = yAdaline(W, b, X_input)  # Calcula a saída do Adaline para a amostra

            erroi = yd[i] - yi  # Determina o erro para a amostra

            W = W + alfa * erroi * X_input
            b = b + alfa * erroi

            SEQ += erroi**2  # Acumula a soma dos erros quadrados
            print(SEQ)
        VetorSEQ.append(SEQ)  # Armazena a soma dos erros quadrados para a época atual
        Epoca += 1
    return W, b, VetorSEQ

response = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=PETR4.SA&apikey=EDMTY79RDZIBGQ7J")
dados_json = response.json()["Time Series (Daily)"]

valores = []
for data, informacoes in dados_json.items():
    valores_data = [
        data,
        informacoes['1. open'],
        informacoes['2. high'],
        informacoes['3. low'],
        informacoes['4. close'],
        informacoes['5. volume']
    ]
    valores.append(valores_data)

X = []
yd = []

for valor in valores:
    data, abertura, alta, baixa, fechamento, volume = valor
    X.append([float(abertura), float(alta), float(baixa), float(volume)])
    yd.append(float(fechamento))

X = np.array(X)

data = list(zip(X, yd))
random.shuffle(data)
X, yd = zip(*data)

# Converter X e yd de volta para arrays NumPy
X = np.array(X)
yd = np.array(yd)

alfa = 0.00001
maxEpocas = 1000
tol = 0.001
W = np.random.rand(X.shape[1]) * 2 - 1
b = random.uniform(-1, 1)

W, b, vetor_seq = treina_adaline(W, b, X, yd, alfa, maxEpocas, tol)

# Plota SEQ
plt.plot(vetor_seq, 'b-')
plt.xlabel('Época')
plt.ylabel('SEQ')
plt.title('Somatório dos Erros Quadráticos')
plt.show()

# Previsões
previsoes = []
for i in range(len(X)):
    previsao = yAdaline(W, b, X[i])
    previsoes.append(previsao)

# Plota as previsões em relação aos valores reais
plt.plot(range(len(yd), yd, 'b-', label='Valores Reais'))
plt.plot(range(len(previsoes), previsoes, 'r-', label='Previsões'))
plt.xlabel('Amostras')
plt.ylabel('Valor de Fechamento')
plt.legend()
plt.title('Previsão de Valores de Fechamento')
plt.show()
