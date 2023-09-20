import numpy as np
import random
import matplotlib.pyplot as plt 

#Define Function Perceptron with return y_output
def yPerceptron(Weight, bias, X_inputs): #Array of Attributes | Weight | Bias
    u_value_activation = np.dot(Weight,X_inputs) + bias #Calcule Threshold
    y_output = [] #Initializing y_output as empty

    for n in u_value_activation: #Using For Each to 
        if(n >= 0): #Check if u value is bigger than 0
            y_output.append(1) #Add value 1 in array y_output
        else:
            y_output.append(0) #Add value 1 in array y_output
    return np.array(y_output) #Return array y_output

#Function Training Perceptron
def treina_perceptron(W, b, X, yd, alfa, maxEpocas, tol):
    N = X.shape[1]  # Número de amostras de entrada
    SEQ = tol
    VetorSEQ = []  # Lista para armazenar o somatório dos erros quadráticos a cada época

    Epoca = 1
    while (Epoca <= maxEpocas) and (SEQ >= tol):
        SEQ = 0  # Inicializa o somatório dos erros quadráticos para a época atual
        
        for i in range(0,N):
            yi = yPerceptron(W,b,X[:,i])
            erroi = yd[0,i] - yi  # Determina o erro para a amostra i

            # Atualiza os pesos e o bias
            W = W + alfa * erroi * X[:,i].T
            b = b + alfa * erroi
            SEQ += erroi**2  # Acumula o somatório dos erros quadráticos
        VetorSEQ.append(SEQ[0])  # Armazena o somatório dos erros quadráticos da época atual
        Epoca += 1
    return W, b, VetorSEQ
    

#Cria e treina um Perceptron para porta lógica AND de duas entradas
X = np.array([[0,1,0,1], [0,0,1,1]])
yd = np.array([[0.,0.,0.,1.]])
alfa = 1.2
maxEpocas = 10
tol = 0.001
W = np.array(np.random.rand(1,2))*2 - 1
b = random.uniform(-1, 1)
W, b, vetor_seq = treina_perceptron(W, b, X, yd, alfa, maxEpocas, tol)

#Plota SEQ
print("Vetor seq:", vetor_seq)
plt.plot(vetor_seq, 'b-')
plt.xlabel('Epoca')
plt.ylabel('SEQ')
plt.title('Somatório dos Erros Quadráticos')
plt.show()

y = yPerceptron(W,b,X)
print(y)



