import numpy as np
import random
import matplotlib.pyplot as plt
import requests
import json

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define a função Perceptron com retorno de u_value_activation
def yPerceptron(Weight, bias, X_inputs):
    u_value_activation = np.dot(Weight, X_inputs) + bias  # Calcule Threshold
    return u_value_activation

# Função Training Perceptron
def treina_perceptron(W, b, X, yd, alfa, maxEpocas, tol):
    N = X.shape[1]  # Número de amostras de entrada
    VetorSEQ = []  # Lista para armazenar o somatório dos erros quadráticos a cada época

    for epoca in range(maxEpocas):
        SEQ = 0  # Inicializa o somatório dos erros quadráticos para a época atual
        
        for i in range(N):
            yi = yPerceptron(W, b, X[:, i])           
            erroi = yd[0, i] - yi  # Determina o erro para a amostra i

            # Verifica se o erro é muito grande (evita overflow)
            if abs(erroi) > 1e3:
                print(f"Erro grande na época {epoca}, amostra {i}, erroi = {erroi}")
                #continue

            # Atualiza os pesos e o bias
            W = W + alfa * erroi * X[:, i]
            b = b + alfa * erroi
            SEQ += erroi**2  # Acumula o somatório dos erros quadráticos
        
        VetorSEQ.append(SEQ)  # Armazena o somatório dos erros quadráticos da época atual

        # Verifique se a sequência de erros é menor que a tolerância (indicando convergência)
        if SEQ < tol:
            break

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

X = np.array(X).T
yd = np.array([yd])

alfa = 0.000001
maxEpocas = 10000
tol = 0.001
W = np.array(np.random.rand(1, 4)) * 2 - 1
b = random.uniform(-1, 1)

W, b, vetor_seq = treina_perceptron(W, b, X, yd, alfa, maxEpocas, tol)

# Plota SEQ
print("Vetor seq:", vetor_seq)
plt.plot(vetor_seq, 'b-')
plt.xlabel('Época')
plt.ylabel('SEQ')
plt.title('Somatório dos Erros Quadráticos')
plt.show()

# Previsões
previsoes = []
for i in range(X.shape[1]):
    previsao = yPerceptron(W, b, X[:, i])
    previsoes.append(previsao)

# Plota as previsões em relação aos valores reais
plt.plot(range(len(yd[0])), yd[0], 'b-', label='Valores Reais')
plt.plot(range(len(previsoes)), previsoes, 'r-', label='Previsões')
plt.xlabel('Amostras')
plt.ylabel('Valor de Fechamento')
plt.legend()
plt.title('Previsão de Valores de Fechamento')
plt.show()
