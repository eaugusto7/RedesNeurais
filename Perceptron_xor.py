#Function AND 'n OR using Perceptron

#Import Library
import numpy as np
import matplotlib.pyplot as plt 

#Define Function Perceptron with return y_output
def yPerceptron(X_inputs, Weight, bias): #Array of Attributes | Weight | Bias
    u_value_activation = np.dot(Weight,X_inputs) + bias #Calcule Threshold
    y_output = [] #Initializing y_output as empty

    for n in u_value_activation[0]: #Using For Each to 
        if(n == 0.5): #Check if u value is bigger than 0
            y_output.append(1) #Add value 1 in array y_output
        else:
            y_output.append(0) #Add value 1 in array y_output
    return np.array(y_output) #Return array y_output


X_inputs = np.array([[0, 0, 0, 0, 1, 1, 1, 1],  # Inicializar matriz com atributos (3 entradas)
                     [0, 0, 1, 1, 0, 0, 1, 1],
                     [0, 1, 0, 1, 0, 1, 0, 1]])
Weight_XOR = np.array([[0.5, 0.5, 0.5]]) #Initializing matrix with Weight to XOR 
#Para o XOR tem que ter a saida  0 1 1 0

bias = -0

y_output_XOR = yPerceptron(X_inputs, Weight_XOR, bias) #Create variable to save output of Perceptron XOR
print("Saída XOR: ", y_output_XOR) #Show value of output of Perceptron XOR


# Plot dos gráficos para y_output_XOR
#plt.figure(figsize=(8, 4))
#plt.subplot(1, 2, 1)
#plt.title("Saída do Perceptron - Função XOR")
#plt.scatter(X_inputs[0], X_inputs[1], c=y_output_XOR, cmap='viridis')
#plt.xlabel("Entrada 1")
#plt.ylabel("Entrada 2")

#plt.tight_layout()
#plt.show()

# Preparar os dados para o gráfico 3D
x = X_inputs[0]
y = X_inputs[1]
z = X_inputs[2]
c = y_output_XOR

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotar o gráfico 3D
ax.scatter(x, y, z, c=c, cmap='viridis')
ax.set_xlabel('Entrada 1')
ax.set_ylabel('Entrada 2')
ax.set_zlabel('Entrada 3')
ax.set_title('Gráfico 3D da Saída XOR em Relação às Entradas')
plt.grid(True)
plt.show()