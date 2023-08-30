# Import Lib
import numpy as np
import matplotlib.pyplot as plt 

#Create array of number
x = np.array([[1,2,3,4,5,6,7,8,9]])
x = x.transpose()
print('x: \n', x)
print('Dimensao do x: ', x.shape)

y = 3*x + 2
print('y: \n', y)
print('Dimensao do y: ', y.shape)

#Plot X and Y
plt.plot(x, y)
plt.xlabel('Eixo x')
plt.ylabel('Eixo y')
plt.title('Gráfico xy')
#plt.show()

#Plot2 X and Y with Graphic Scatter
plt.scatter(x, y, marker='o', color='red')
plt.xlabel('Eixo x')
plt.ylabel('Eixo y')
plt.title('Gráfico xy - Scatter')
#plt.show()

#Plot2 X and Y with Array z
z = 3*x**2 + 2
print('z: \n', z)
plt.plot(x, y, label = 'Linha Y', color='green')
plt.plot(x, z, label = 'Linha Z', color='red')
plt.legend()
plt.title('Gráfico xyz')
#plt.show()

#Matriz
A = np.array([[0,1,2,3], [4,5,6,7], [8,9,10,11]])
print('A: \n', A)
N = A.shape
print('Tamanho Matriz A: ', N)

#Get Just value of [2, 3]
W = A[2,3]
print('Elemento linha 2, coluna 3: ', W)

B = np.array([A[:,0]]).T
print('B: \n', B)

C = np.array([A[2,:]])
print('C: \n', C)

#Exemplo de produto de vetores
x = np.array([[1,2,3,4]])
y = 3*x + 2
y = y.transpose()
z = np.dot(x,y)
print('z: \n', z)

#Exemplo de vetor longo
long_array_x = np.arange(11.)
print('long_array_x: \n', long_array_x)

long_array_y = np.arange(10,30,5) #Inicial, Final, Passo
print('long_array_y: \n', long_array_y)

#Matriz de numero 1
long_array_z = np.ones([2,3])
print('long_array_z: \n', long_array_z)

#Matriz de numero 0
long_array_z = np.zeros([4,3])
print('long_array_z: \n', long_array_z)

#Vetor de numeros aleatorios
randA = np.array(np.random.randn(1,5))
print('randA: \n', randA)

randB = np.array(np.random.randn(1,5))
print('randB: \n', randB)

randC = np.concatenate((randA,randB), axis=0)
print('randC: \n', randC)