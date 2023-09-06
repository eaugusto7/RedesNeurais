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