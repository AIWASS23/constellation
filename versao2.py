import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input

data_folder = "images"
label_folder = "labels.csv"
shape = 50
activation = 'relu'

def load_data(folder, labels_file, target_size=(shape, shape)):
    # Carregar o arquivo CSV com as labels
    labels_df = pd.read_csv(labels_file)
    
    # Lista para armazenar os caminhos das imagens e as labels correspondentes
    images = []
    labels = []

    # Mapeamento das labels para números
    label_mapping = {label: idx for idx, label in enumerate(labels_df['class'].unique())}
    
    # Iterar sobre as imagens na pasta de teste
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Caminho completo para a imagem
            image_path = os.path.join(folder, filename)
            
            # Carregar a imagem e redimensioná-la
            img = image.load_img(image_path, target_size=target_size)
            img_array = image.img_to_array(img)
            
            # Adicionar a imagem ao conjunto de dados
            images.append(img_array)
            
            # Extrair a label correspondente do arquivo CSV
            label = labels_df[labels_df['filename'] == filename]['class'].values[0]
            label_idx = label_mapping[label]
            labels.append(label_idx)
            # labels.append(label)
    
    # Converter para arrays numpy
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels, label_mapping

# Carregar todas as imagens e labels
images, labels, label_mapping = load_data(data_folder, label_folder)

# Dividir os dados em treino e teste
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Normalizar os dados
train_images = train_images / 255.0
test_images = test_images / 255.0

# Obter o número de classes
num_classes = len(label_mapping)

# Converter as labels para one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)

# Modelos 
model1 = Sequential()
model1.add(Input(shape=(shape, shape, 3)))
model1.add(Conv2D(16, kernel_size=(3, 3), activation = activation))
model1.add(Conv2D(32, (3, 3), activation = activation))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, (3, 3), activation = activation))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, (3, 3), activation = activation))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, (3, 3), activation = activation))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))
model1.add(Flatten())
model1.add(Dense(32, activation = activation))
model1.add(Dropout(0.5))
model1.add(Dense(num_classes, activation='softmax'))

# Compilar os modelos
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # https://www.tensorflow.org/api_docs/python/tf/keras/losses # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers

# Treinar o modelo
history1 = model1.fit(train_images, train_labels, epochs = 20, batch_size=32, validation_data = (test_images, test_labels))

# Avaliar o modelo nos dados de teste e treino
results_model1_test = model1.evaluate(test_images, test_labels)
results_model1_train = model1.evaluate(train_images, train_labels)

# Previsões nos dados de teste
predictions = model1.predict(test_images)

# Converter previsões e labels para classe
predictions_classes = np.argmax(predictions, axis=1)
test_labels_classes = np.argmax(test_labels, axis=1)

precision = precision_score(test_labels_classes, predictions_classes, average = 'weighted')
recall = recall_score(test_labels_classes, predictions_classes, average = 'weighted')

# Matriz de Confusão
cm = confusion_matrix(test_labels_classes, predictions_classes)

# Calcular F1-Score e Especificidade
def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp)

f1 = f1_score(test_labels_classes, predictions_classes, average='weighted')
spec = specificity(test_labels_classes, predictions_classes)

# Imprimir os resultados
print("Resultados do modelo:")
print(f"Acurácia teste: {results_model1_test[1]}")
print(f"Acurácia treino: {results_model1_train[1]}")
print(f"Perda teste: {results_model1_test[0]}")
print(f"Perda treino: {results_model1_train[0]}")
print(f"Precisão: {precision}")
print(f"Revocação: {recall}")
print(f'Matriz de confusão: {cm}')
print(f'F1-Score: {f1}')
# print(f'Especificidade: {spec}')