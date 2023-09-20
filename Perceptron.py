#Function AND 'n OR using Perceptron

#Import Library
import numpy as np
import matplotlib.pyplot as plt 

#Define Function Perceptron with return y_output
def yPerceptron(X_inputs, Weight, bias): #Array of Attributes | Weight | Bias
    u_value_activation = np.dot(Weight,X_inputs) + bias #Calcule Threshold
    y_output = [] #Initializing y_output as empty

    for n in u_value_activation[0]: #Using For Each to 
        if(n >= 0): #Check if u value is bigger than 0
            y_output.append(1) #Add value 1 in array y_output
        else:
            y_output.append(0) #Add value 1 in array y_output
    return np.array(y_output) #Return array y_output

X_inputs = np.array([[0,1,0,1], [0,0,1,1]]) #Initializing matrix with Attributes
Weight_AND = np.array([[0.4, 0.4]]) #Initializing matrix with Weight to AND 
Weight_OR = np.array([[0.6, 0.6]]) #Initializing matrix with Weight to OR
bias = -0.6

y_output_AND = yPerceptron(X_inputs, Weight_AND, bias) #Create variable to save output of Perceptron AND
y_output_OR = yPerceptron(X_inputs, Weight_OR, bias) #Create variable to save output of Perceptron OR
print("Saída AND: ", y_output_AND) #Show value of output of Perceptron AND
print("Saída OR: ", y_output_OR) #Show value of output of Perceptron OR

# Plot dos gráficos para y_output_AND
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("Saída do Perceptron - Função AND")
plt.scatter(X_inputs[0], X_inputs[1], c=y_output_AND, cmap='viridis')
plt.xlabel("Entrada 1")
plt.ylabel("Entrada 2")

# Plot dos gráficos para y_output_OR
plt.subplot(1, 2, 2)
plt.title("Saída do Perceptron - Função OR")
plt.scatter(X_inputs[0], X_inputs[1], c=y_output_OR, cmap='viridis')
plt.xlabel("Entrada 1")
plt.ylabel("Entrada 2")

plt.tight_layout()
plt.show()
