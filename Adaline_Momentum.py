# Importe as bibliotecas necessárias
import numpy as np  # NumPy para operações numéricas
import random  # Random para gerar números aleatórios
import matplotlib.pyplot as plt  # Matplotlib para plotar gráficos
from random import uniform  # Random para gerar distribuição uniforme

# Define a função 'yAdaline' que retorna a saída do Adaline
def yAdaline(Weight, bias, X_inputs):
    y_output = np.dot(Weight.T, X_inputs) + bias  # Calcula a soma ponderada
    return y_output[0]  # Retorna a saída como um array NumPy

# Define a função 'treina_adaline' para treinar o Adaline
def treina_adaline(W, b, X, yd, alfa, maxEpocas, tol):
    N = len(X)  # Número de amostras de entrada

    SEQ = tol
    VetorSEQ = []  # Lista para armazenar a soma dos erros quadrados para cada época

    Epoca = 1

    while (Epoca <= maxEpocas) and (SEQ >= tol):  # Continue o treinamento até maxEpocas ou até atingir a tolerância
        SEQ = 0  # Inicializa a soma dos erros quadrados para a época atual

        for i in range(N):  # Loop através de cada amostra de entrada
            X_input = X[i].reshape(1, -1)  # Redimensiona a entrada para corresponder ao formato do peso
            yi = yAdaline(W, b, X_input)  # Calcula a saída do Adaline para a amostra

            erroi = yd[i] - yi  # Determina o erro para a amostra

            W = W + alfa * erroi * X_input.T
            b = b + alfa * erroi
            SEQ += erroi**2  # Acumula a soma dos erros quadrados
        VetorSEQ.append(SEQ[0])  # Armazena a soma dos erros quadrados para a época atual
        Epoca += 1
    return W, b, VetorSEQ

# Define a função 'treina_adaline' para treinar o Adaline
def treina_adaline_momentum(W, b, X, yd, alfa, maxEpocas, tol, beta):
    N = len(X)  # Número de amostras de entrada
    SEQ = tol
    VetorSEQ = []  # Lista para armazenar a soma dos erros quadrados para cada época
    Epoca = 1

    W_Antigo = W
    W_Atual = W
    b_antigo = b
    b_atual = b

    while (Epoca <= maxEpocas) and (SEQ >= tol):  # Continue o treinamento até maxEpocas ou até atingir a tolerância
        SEQ = 0  # Inicializa a soma dos erros quadrados para a época atual

        for i in range(N):  # Loop através de cada amostra de entrada
            X_input = X[i].reshape(1, -1)  # Redimensiona a entrada para corresponder ao formato do peso
            yi = yAdaline(W, b, X_input)  # Calcula a saída do Adaline para a amostra

            erroi = yd[i] - yi  # Determina o erro para a amostra

            W = W_Atual + alfa * erroi * X_input.T + beta*(W_Atual - W_Antigo)
            b = b + alfa * erroi + beta*(b_atual - b_antigo)

            print(W)

            W_Antigo = W_Atual
            b_antigo = b_atual
            W_Atual = W
            b_atual = b

            SEQ += erroi**2  # Acumula a soma dos erros quadrados
        VetorSEQ.append(SEQ[0])  # Armazena a soma dos erros quadrados para a época atual
        Epoca += 1
    return W, b, VetorSEQ

# Define a função 'geraData' para gerar dados de amostra com ruído
def geraData(num_amostras):
    X = []
    Yd = []

    for i in range(num_amostras):
        X.append([i])
        ruido = uniform(0, 6) * random.choice([1, -1])
        Yd.append(int(2 * i + 3) + ruido)

    data = list(zip(X, Yd))
    random.shuffle(data)
    X, Yd = zip(*data)

    return np.array(X), np.array(Yd)

# Define funções de plotagem para diferentes cenários
def plotadc1D_retaoriginal_amostra(X, Yd, a, b):
    plt.figure(figsize=(8, 6))
    plt.title("Plot 1D")
    plt.scatter(X, Yd, color='blue', label='Amostra de Treinamento')
    x_min, x_max = X.min() - 1, X.max() + 1
    x_line = np.array([x_min, x_max])
    y_line2 = 2 * x_line + 3
    plt.plot(x_line, y_line2, color='green', label='Reta Original')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

def plotadc1D_retaadaline_amostra(X, Yd, a, b):
    plt.figure(figsize=(8, 6))
    plt.title("Plot 1D")
    plt.scatter(X, Yd, color='blue', label='Amostra de Treinamento')
    x_min, x_max = X.min() - 1, X.max() + 1
    x_line = np.array([x_min, x_max])
    y_line = a * x_line + b
    plt.plot(x_line, y_line, color='red', label='Reta obtida pelo Adaline')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

def plotadc1D_retaadaline_retaoriginal(X, Yd, a, b):
    plt.figure(figsize=(8, 6))
    plt.title("Plot 1D")
    x_min, x_max = X.min() - 1, X.max() + 1
    x_line = np.array([x_min, x_max])
    y_line = a * x_line + b
    y_line2 = 2 * x_line + 3
    plt.plot(x_line, y_line, color='red', label='Reta obtida pelo Adaline')
    plt.plot(x_line, y_line2, color='green', label='Reta Original')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()

# Defina os parâmetros e execute o treinamento do Adaline
num_amostras = 200
X, Yd = geraData(num_amostras)

alfa = 0.00001
maxEpocas = 10
tol = 0.001
W = np.random.rand(1, X.shape[1]) * 2 - 1
b = random.uniform(-1, 1)
beta = 0.01

W_gerado = W
b_gerado = b

W, b, vetor_seq = treina_adaline(W, b, X, Yd, alfa, maxEpocas, tol)
W2, b2, vetor_seq2 = treina_adaline_momentum(W_gerado, b_gerado, X, Yd, alfa, maxEpocas, tol, beta)

print(W)
print(W2)


X, Yd = geraData(num_amostras)

y = [yAdaline(W, b, x) for x in X]

a = W[0, 0].item()
b_equacao = b.item()

# Plot os resultados e a soma dos erros quadrados
plotadc1D_retaoriginal_amostra(X, Yd, a, b_equacao)
plotadc1D_retaadaline_amostra(X, Yd, a, b_equacao)
plotadc1D_retaadaline_retaoriginal(X, Yd, a, b_equacao)

print("Vetor seq:", vetor_seq)
plt.plot(vetor_seq, 'b-')
plt.xlabel('Epoca')
plt.ylabel('SEQ')
plt.title('Sum of Squared Errors')
plt.show()


print("Vetor seq2:", vetor_seq2)
plt.plot(vetor_seq, 'b-')
plt.xlabel('Epoca')
plt.ylabel('SEQ')
plt.title('Sum of Squared Errors')
plt.show()