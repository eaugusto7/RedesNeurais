# Import libraries
import numpy as np  # NumPy for numerical operations
import random  # Random for generating random numbers
import matplotlib.pyplot as plt  # Matplotlib for plotting graphs
from random import shuffle 

# Define a function 'yPerceptron' that returns the Perceptron output
def yPerceptron(Weight, bias, X_inputs):  # Weight: Array of attributes weights, bias: Weight for bias, X_inputs: Input attributes
    u_value_activation = np.dot(Weight, X_inputs) + bias  # Calculate the weighted sum (activation)
    y_output = []  # Initialize the output list

    for n in u_value_activation:  # Loop through each activation value
        if n >= 0:  # If the activation is greater than or equal to 0
            y_output.append(1)  # Append 1 to the output list
        else:
            y_output.append(0)  # Otherwise, append 0 to the output list
    return np.array(y_output)  # Return the output as a NumPy array

# Define another function 'yPerceptron_final' that returns the Perceptron output
def yPerceptron_final(Weight, bias, X_inputs):
    u_value_activation = np.dot(X_inputs, Weight.T) + bias  # Calculate the weighted sum (activation)

    if u_value_activation >= 0:
        return 1
    else:
        return 0
    
# Define a function 'treina_perceptron' for training the Perceptron
def treina_perceptron(W, b, X, yd, alfa, maxEpocas, tol):
    N = len(X)  # Number of input samples

    SEQ = tol
    VetorSEQ = []  # List to store the sum of squared errors for each epoch

    Epoca = 1
    while (Epoca <= maxEpocas) and (SEQ >= tol):  # Continue training until maxEpocas or tolerance reached
        SEQ = 0  # Initialize the sum of squared errors for the current epoch
        
        for i in range(0, N):  # Loop through each input sample
            yi = yPerceptron(W, b, X[i])  # Calculate the Perceptron output for the sample
            erroi = yd[0, i] - yi  # Determine the error for the sample

            # Update the weights and bias using the delta rule
            W = W + alfa * erroi * X[i].T
            b = b + alfa * erroi
            SEQ += erroi**2  # Accumulate the sum of squared errors
        VetorSEQ.append(SEQ[0])  # Store the sum of squared errors for the current epoch
        Epoca += 1
    return W, b, VetorSEQ

def plotadc2D(X_inputs, y):
    # Crie uma lista de cores com base nos valores de y
    colors = ['red' if yi == 0 else 'blue' for yi in y]

    # Plote o gráfico de dispersão com cores apropriadas
    plt.figure(figsize=(4, 4))
    plt.subplot(1, 1, 1)
    plt.title("Plot 2D")
    plt.scatter(X_inputs[:, 0], X_inputs[:, 1], c=colors, cmap='viridis')
    plt.xlabel("Entrada 1")
    plt.ylabel("Entrada 2")

def plotareta(W,b, intervalo):
    x1 = np.linspace(intervalo[0], intervalo[1], num = 10)
    x2 = -(W[0,0]*x1 + b)/W[0,1]
    plt.plot(x1.T, x2.T, 'b-')
    plt.tight_layout()
    plt.show()

def geragauss(num_class, num_amostras_class, matriz_mid, varience):
    X = []
    Yd = []

    for i in range(num_class):
        for j in range(num_amostras_class[i]):  
            tempX1 = random.gauss(matriz_mid[0,i], varience[0,i])
            tempX2 = random.gauss(matriz_mid[1,i], varience[1,i])
                     
            X.append([tempX1, tempX2])  
            Yd.append(int(i))

    # Combine X and Yd into a list of tuples for shuffling
    data = list(zip(X, Yd))
    random.shuffle(data)
    
    # Unzip the shuffled data to get shuffled X and Yd
    X, Yd = zip(*data)

    return np.array(X), np.array(Yd)

num_class = 2 #Number of Class in geragauss
num_amostras_class = np.array([500,300]) #Number of sample for each class
matriz_mid = np.array([[1.5, 6],[8, 3.5]]).T #Midpoint of the Gaussian distribution
varience = np.array([[1.8, 1.5],[1.5, 1.8]]).T #Variance of Gaussian distribution

X, Yd = geragauss(num_class, num_amostras_class, matriz_mid, varience)

plotadc2D(X,Yd)
plt.tight_layout()
plt.show()

Yd = np.array([Yd])

alfa = 0.5  # Learning rate
maxEpocas = 100  # Maximum number of training epochs
tol = 0.001  # Tolerance for stopping training
W = np.array(np.random.rand(1, X.shape[1])) * 2 - 1  # Initialize weights randomly
b = random.uniform(-1, 1)  # Initialize bias randomly

W, b, vetor_seq = treina_perceptron(W, b, X, Yd, alfa, maxEpocas, tol)  # Train the Perceptron

# Use the trained Perceptron to make predictions
y = [yPerceptron_final(W, b, x) for x in X]

# Plot the sum of squared errors (SEQ)
print("Vetor seq:", vetor_seq)
plt.plot(vetor_seq, 'b-')
plt.xlabel('Epoca')
plt.ylabel('SEQ')
plt.title('Sum of Squared Errors')
plt.show()

# Plot the 2D Points and line
plotadc2D(X,y)
plotareta(W,b,np.array([0, 15]))