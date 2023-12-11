import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Constantes
NUM_FEATURES = 22  # Número de features de entrada
NUM_OUTPUTS = 6    # Número de variáveis de saída
EPOCHS = 100
BATCH_SIZE = 32

# Lista para armazenar os DataFrames de cada arquivo
data_frames = []

# Lista dos nomes dos arquivos para cada sensor
sensor_files = ['acelerometro_estatico.csv', 'gyro_estatico.csv', 'gnss_estatico.csv', 'in_profile_estatico.csv']

# Carregar os dados de cada arquivo e adicionar ao DataFrame
for file in sensor_files:
    sensor_data = pd.read_csv(file, delimiter=';', header=None)
    data_frames.append(sensor_data)

# Concatenar os DataFrames ao longo das colunas (eixo 1)
data = pd.concat(data_frames, axis=1)

# Adicionar os rótulos para as colunas
data.columns = ['acel_x', 'acel_y', 'acel_z', 'giro_x', 'giro_y', 'giro_z', 
                'gnss_pos_x', 'gnss_pos_y', 'gnss_pos_z', 'gnss_vel_x', 'gnss_vel_y', 'gnss_vel_z', 
                'erro_clock', 'erro_clock_drift', 'dop_norte', 'dop_leste', 'dop_vertical', 'dop_time', 
                'dop_horizontal', 'dop_position', 'dop_geometric', 'num_satelites',
                'out_longitude', 'out_latitude', 'out_altitude', 'out_vel_x', 'out_vel_y', 'out_vel_z']

# Separar features (entrada) e targets (saída)
X = data.iloc[:, :NUM_FEATURES]
y = data.iloc[:, NUM_FEATURES:]

# Dividir o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Normalizar os dados
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Construir o modelo da rede neural
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(NUM_FEATURES,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(NUM_OUTPUTS, activation='linear'))  # Saída: Longitude, Latitude, Altitude, Vel_x, Vel_y, Vel_z

# Exibir o formato da saída da última camada
print(model.output_shape)

# Compilar o modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Treinar o modelo
model.fit(X_train_scaled, y_train_scaled, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

# Avaliar o modelo no conjunto de teste
predictions_scaled = model.predict(X_test_scaled)
mse = mean_squared_error(y_test_scaled, predictions_scaled)
print(f'Mean Squared Error on Test Data: {mse}')

# Salvar o modelo
model.save('gps_model')  # Não incluir a extensão .h5

# Carregar o modelo
loaded_model = tf.keras.models.load_model('gps_model')

# Lista dos nomes dos arquivos de teste para cada sensor
test_sensor_files = ['acelerometro_estatico_test.csv', 'gyro_estatico_test.csv', 'gnss_estatico_test.csv']

# Carregar os dados de teste de cada arquivo e adicionar ao DataFrame
test_data_frames = []
for file in test_sensor_files:
    test_sensor_data = pd.read_csv(file, delimiter=';', header=None)
    test_data_frames.append(test_sensor_data)

# Concatenar os DataFrames de teste ao longo das colunas (eixo 1)
test_data = pd.concat(test_data_frames, axis=1)
X_test_data_scaled = scaler_X.transform(test_data)

# Fazer previsões
predicted_coordinates_scaled = model.predict(X_test_data_scaled)

# Desnormalizar as previsões
predicted_coordinates = scaler_y.inverse_transform(predicted_coordinates_scaled)

# Criar DataFrame com as previsões desnormalizadas
predictions_df = pd.DataFrame(data=predicted_coordinates, columns=['out_longitude', 'out_latitude', 'out_altitude', 'out_vel_x', 'out_vel_y', 'out_vel_z'])

# Adicionar as previsões ao DataFrame de teste
test_data_with_predictions = predictions_df

# Salvar as previsões em um arquivo CSV
test_data_with_predictions.to_csv('PredictedOutput_Estatico.csv', index=False)

# Imprimir as dimensões de predicted_coordinates
print('Dimensions of predicted_coordinates:', predicted_coordinates.shape)

print('Predicted Coordinates:')
print(predicted_coordinates)