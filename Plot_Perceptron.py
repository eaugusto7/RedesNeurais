# Import libraries
import numpy as np  # NumPy for numerical operations
import random  # Random for generating random numbers
import matplotlib.pyplot as plt  # Matplotlib for plotting graphs

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
def yPerceptron_final(Weight, bias, X_inputs):  # Weight: Array of attributes weights, bias: Weight for bias, X_inputs: Input attributes
    u_value_activation = np.dot(Weight, X_inputs) + bias  # Calculate the weighted sum (activation)
    y_output = []  # Initialize the output list

    for n in u_value_activation[0]:  # Loop through each activation value (assuming single output)
        if n >= 0:  # If the activation is greater than or equal to 0
            y_output.append(1)  # Append 1 to the output list
        else:
            y_output.append(0)  # Otherwise, append 0 to the output list
    return np.array(y_output)  # Return the output as a NumPy array

# Define a function 'treina_perceptron' for training the Perceptron
def treina_perceptron(W, b, X, yd, alfa, maxEpocas, tol):
    N = X.shape[1]  # Number of input samples
    SEQ = tol
    VetorSEQ = []  # List to store the sum of squared errors for each epoch

    Epoca = 1
    while (Epoca <= maxEpocas) and (SEQ >= tol):  # Continue training until maxEpocas or tolerance reached
        SEQ = 0  # Initialize the sum of squared errors for the current epoch
        
        for i in range(0, N):  # Loop through each input sample
            yi = yPerceptron(W, b, X[:, i])  # Calculate the Perceptron output for the sample
            erroi = yd[0, i] - yi  # Determine the error for the sample

            # Update the weights and bias using the delta rule
            W = W + alfa * erroi * X[:, i].T
            b = b + alfa * erroi
            SEQ += erroi**2  # Accumulate the sum of squared errors
        VetorSEQ.append(SEQ[0])  # Store the sum of squared errors for the current epoch
        Epoca += 1
    return W, b, VetorSEQ

def plotadc2D(X_inputs,y):
    # Plot dos gráficos para y_output_AND
    plt.figure(figsize=(4, 4))
    plt.subplot(1, 1, 1)
    plt.title("Saída do Perceptron - Função OR")
    plt.scatter(X_inputs[0], X_inputs[1], c=y, cmap='viridis')
    plt.xlabel("Entrada 1")
    plt.ylabel("Entrada 2")

def plotareta(W,b, intervalo):
    x1 = np.linspace(intervalo[0], intervalo[1], num = 10)
    x2 = -(W[0,0]*x1 + b)/W[0,1]
    plt.plot(x1.T, x2.T, 'b-')
    plt.tight_layout()
    plt.show()

# Create input data and train a Perceptron for the logical AND n OR operation with two inputs
X = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])  # Input data
yd_AND = np.array([[0., 0., 0., 1.]])  # Desired outputs for OR operation
yd_OR = np.array([[0., 1., 1., 1.]])  # Desired outputs for OR operation
alfa = 1.2  # Learning rate
maxEpocas = 10  # Maximum number of training epochs
tol = 0.001  # Tolerance for stopping training
W = np.array(np.random.rand(1, 2)) * 2 - 1  # Initialize weights randomly
b = random.uniform(-1, 1)  # Initialize bias randomly
W, b, vetor_seq = treina_perceptron(W, b, X, yd_OR, alfa, maxEpocas, tol)  # Train the Perceptron


# Use the trained Perceptron to make predictions
y = yPerceptron_final(W, b, X)

# Plot the sum of squared errors (SEQ)
print("Vetor seq:", vetor_seq)
plt.plot(vetor_seq, 'b-')
plt.xlabel('Epoca')
plt.ylabel('SEQ')
plt.title('Sum of Squared Errors')
plt.show()

plotadc2D(X,y)
plotareta(W,b,np.array([0, 1.5]))

print(y)
