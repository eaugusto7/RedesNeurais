#Teste comando if
idade = 17

if idade < 12:
    print("Criança")
elif idade < 18:
    print("Adolescente")
else:
    print("Adulto")
print("\n")

#Teste while
n = 1
while n <= 10:
    quadrado = n**2
    print(f'O quadrado de {n} é {quadrado}')
    n += 1
print("\n")

#Teste break
a = 0
while True:
    print(a)
    a += 2
    if a >= 10:
        break
print("\n")

#Teste comando for
for i in 'Hello':
    print(i)
print("\n")

#Teste for
minha_lista = [1,2,3,4,5]
for n in minha_lista:
    print(n)
print("\n")

#Teste for 2
for n in range(10):
    print(n)
print("\n")

#Teste for 3
for i in range(6,1,-2): #Parametros inicial, final e passo
    print(i)
print("\n")

#Teste for 4
for x in range(4):
    for y in range(3):
        if x < y:
            print(x,y)
print("\n")

#Exemplo de Funcao - Cria uma funcao
def imprimir_quadrado(x):
    print(x*x)

#Call function
imprimir_quadrado(10)
print("\n")

#Cria uma funcao com parametro nao obrigatorio
def pessoa(nome,idade=20):
    print(f'{nome} tem {idade} anos')

#Call function
pessoa('Maria',19)
pessoa('José')
print("\n")

#Cria funcao com retorno
def quadrado(x):
    return x*x

#Call function
y = quadrado(10)
print('y =', y)