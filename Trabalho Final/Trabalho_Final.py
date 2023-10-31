import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler

# Gere dados fictícios para treinamento
def generate_training_data(num_samples):
    train_data = []
    labels = []

    # Abra o arquivo CSV para leitura
    with open("acelerometro.csv", "r") as file:
        linesAcelerometro = file.readlines()

    # Abra o arquivo CSV para leitura
    with open("gyro.csv", "r") as file:
        linesGyro = file.readlines()

    # Abra o arquivo CSV para leitura
    with open("pos_gnss.csv", "r") as file:
        linesPosGNSS = file.readlines()

    # Abra o arquivo CSV para leitura
    with open("vel_gnss.csv", "r") as file:
        linesVelGNSS = file.readlines()

    # Abra o arquivo CSV para leitura
    with open("in_profile.csv", "r") as file:
        InProfile = file.readlines()

    # A primeira linha contém os valores de "fx" separados por ponto e vírgula
    fx_values = linesAcelerometro[0].strip().split(";")
    fx_values = list(map(float, fx_values))

    fy_values = linesAcelerometro[1].strip().split(";")
    fy_values = list(map(float, fy_values))

    fz_values = linesAcelerometro[2].strip().split(";")
    fz_values = list(map(float, fz_values))

    # A primeira linha contém os valores de "fx" separados por ponto e vírgula
    wx_values = linesGyro[0].strip().split(";")
    wx_values = list(map(float, wx_values))

    wy_values = linesGyro[1].strip().split(";")
    wy_values = list(map(float, wy_values))

    wz_values = linesGyro[2].strip().split(";")
    wz_values = list(map(float, wz_values))

    pos_x_gnss_values = linesPosGNSS[0].strip().split(";")
    pos_x_gnss_values = list(map(float, pos_x_gnss_values))

    pos_y_gnss_values = linesPosGNSS[1].strip().split(";")
    pos_y_gnss_values = list(map(float, pos_y_gnss_values))

    pos_z_gnss_values = linesPosGNSS[2].strip().split(";")
    pos_z_gnss_values = list(map(float, pos_z_gnss_values))

    vel_x_gnss_values = linesVelGNSS[0].strip().split(";")
    vel_x_gnss_values = list(map(float, vel_x_gnss_values))

    vel_y_gnss_values = linesVelGNSS[1].strip().split(";")
    vel_y_gnss_values = list(map(float, vel_y_gnss_values))

    vel_z_gnss_values = linesVelGNSS[2].strip().split(";")
    vel_z_gnss_values = list(map(float, vel_z_gnss_values))

    in_latitude_values = InProfile[0].strip().split(";")
    in_latitude_values = list(map(float, in_latitude_values))

    in_longitude_values = InProfile[1].strip().split(";")
    in_longitude_values = list(map(float, in_longitude_values))

    in_atitude_values = InProfile[2].strip().split(";")
    in_atitude_values = list(map(float, in_atitude_values))

    for i in range(num_samples):
        # Gere dados fictícios de acelerômetro, giroscópio, posição e velocidade do GNSS
        accelerometer_data = np.array([fx_values[i], fy_values[i], fz_values[i]])
        gyroscope_data = np.array([wx_values[i], wy_values[i], wz_values[i]])
        gnss_position = np.array([pos_x_gnss_values[i], pos_y_gnss_values[i], pos_z_gnss_values[i]])
        gnss_velocity = np.array([vel_x_gnss_values[i], vel_y_gnss_values[i], vel_z_gnss_values[i]])

        # Combine os dados em uma única entrada
        input_data = np.concatenate((accelerometer_data, gyroscope_data, gnss_position, gnss_velocity))

        # Defina a saída desejada (posição NED fictícia)
        ned_position = np.array([in_latitude_values[i], in_longitude_values[i], in_atitude_values[i]])

        train_data.append(input_data)
        labels.append(ned_position)

    return np.array(train_data), np.array(labels)

# Crie a arquitetura da rede neural
model = tf.keras.Sequential([
    tf.keras.layers.Dense(276, activation='tanh', input_shape=(12,)),
    tf.keras.layers.Dropout(0.9),
    tf.keras.layers.Dense(3)  
])

custom_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
sgd_with_momentum = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
rmsprop_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
# Compile o modelo
model.compile(optimizer=sgd_with_momentum, loss='mean_squared_error')

# Gere dados de treinamento fictícios
train_data, labels = generate_training_data(num_samples=500)
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)

# Treine o modelo
model.fit(train_data, labels, epochs=20000)

dados_reais = np.array([0.153899000000000, 0.369535000000000, -9.85417400000000, 0.00158700000000000, 0.000912000000000000, -0.00208300000000000, 5.24023226937869, 5.38230778476496, 3.99309857802446, 0.104804645387574, 0.107646155695301, 0.0798619715604904])
dados_reais = np.reshape(dados_reais, (1, 12))

np.set_printoptions(suppress=True)

# Faça a previsão com os dados reais
previsao = model.predict(dados_reais)

# A previsão será uma matriz com as coordenadas NED
print("Previsão de Posição NED:", previsao[0])
