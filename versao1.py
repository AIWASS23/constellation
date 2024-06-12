import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.model_selection import train_test_split


data_folder = "images"
label_folder = "labels.csv"

def load_data(folder, labels_file, target_size=(500, 500)):
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
model1.add(Conv2D(16, kernel_size=(3, 3), activation='tanh', input_shape=(500, 500, 3)))
model1.add(Conv2D(32, (3, 3), activation='tanh'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, (3, 3), activation='tanh'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, (3, 3), activation='tanh'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, (3, 3), activation='tanh'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))
model1.add(Flatten())
model1.add(Dense(32, activation='tanh'))
model1.add(Dropout(0.5))
model1.add(Dense(num_classes, activation='softmax'))
model1.summary()

model2 = Sequential()
model2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dense(num_classes, activation='softmax'))

model3 = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(500, 500, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compilar os modelos
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # https://www.tensorflow.org/api_docs/python/tf/keras/losses
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinar o modelo
history1 = model1.fit(
    train_images, train_labels,
    epochs=1,
    batch_size=32,
    validation_data=(test_images, test_labels)
)

history2 = model2.fit(
    train_images, train_labels,
    epochs=1,
    batch_size=32,
    validation_data=(test_images, test_labels)
)

history3 = model3.fit(
    train_images, train_labels,
    epochs=1, # use com moderacão
    batch_size=32,
    validation_data=(test_images, test_labels)
)

# Avaliar os modelos nos dados de teste
results_model1 = model1.evaluate(test_images, test_labels)
results_model2 = model2.evaluate(test_images, test_labels)
results_model3 = model3.evaluate(test_images, test_labels)

# Imprimir os resultados
print("Resultados do modelo 1:")
print(f"Acurácia: {results_model1[1]}")
print(f"Perda: {results_model1[0]}")

print("Resultados do modelo 2:")
print(f"Acurácia: {results_model2[1]}")
print(f"Perda: {results_model2[0]}")

print("Resultados do modelo 3:")
print(f"Acurácia: {results_model3[1]}")
print(f"Perda: {results_model3[0]}")

# Previsões nos dados de teste
predictions1 = model1.predict(test_images)
predictions2 = model2.predict(test_images)
predictions3 = model3.predict(test_images)

# Converter previsões e labels para classe
predictions1_classes = np.argmax(predictions1, axis=1)
predictions2_classes = np.argmax(predictions2, axis=1)
predictions3_classes = np.argmax(predictions3, axis=1)

test_labels_classes = np.argmax(test_labels, axis=1)

# Matriz de Confusão
cm1 = confusion_matrix(test_labels_classes, predictions1_classes)
cm2 = confusion_matrix(test_labels_classes, predictions2_classes)
cm3 = confusion_matrix(test_labels_classes, predictions3_classes)

# Exibir Matriz de Confusão
def plot_confusion_matrix(cm, model_name):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_mapping.keys())
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.show()

plot_confusion_matrix(cm1, "Modelo 1")
plot_confusion_matrix(cm2, "Modelo 2")
plot_confusion_matrix(cm3, "Modelo 3")