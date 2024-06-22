import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras import Input
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical

# Dados
data_folder = "images"
label_folder = "labels.csv"
shape = 50 # melhor 50 obs: com 100 os resultados são melhores mas dá sobreajuste

# hiperparâmetros
num_filter = 16 # melhor = 16, 32
neuron = 128 # melhor = 64, 32, 128

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
num_classes = len(label_mapping)

n_splits = 10  # Número de folds para validação cruzada
skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42)

# Armazenar métricas para cada fold
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []

precisions = []
recalls = []
f1_scores = []

# Iterar sobre cada fold
for train_index, val_index in skf.split(images, labels):
    # Dividir os dados em treino e validação
    X_train, X_val = images[train_index], images[val_index]
    y_train, y_val = labels[train_index], labels[val_index]
    
    # Normalizar os dados
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    
    # Converter labels para one-hot encoding
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)
    
    # Modelo 
    model = Sequential()
    model.add(Input(shape = (shape, shape, 3)))
    model.add(Conv2D(num_filter, (3, 3), activation = 'linear')) 
    model.add(Conv2D(num_filter, (3, 3), activation = 'tanh')) 
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(num_filter, (3, 3), activation = 'leaky_relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(num_filter, (3, 3), activation = 'selu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(num_filter, (3, 3), activation = 'selu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(neuron, activation = 'selu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation = 'sigmoid'))
    
    # Compilar o modelo
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Lion', metrics = ['accuracy'])
    
    # Treinar o modelo
    history = model.fit(X_train, y_train, epochs = 10, batch_size = 32, validation_data = (X_val, y_val), verbose = 0)
    
    # Avaliar o modelo nos dados de treino e validação
    train_results = model.evaluate(X_train, y_train, verbose = 0)
    val_results = model.evaluate(X_val, y_val, verbose = 0)
    
    # Previsões nos dados de validação
    predictions = model.predict(X_val)
    predictions_classes = np.argmax(predictions, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)
    
    precision = precision_score(y_val_classes, predictions_classes, average='weighted', zero_division=1)
    recall = recall_score(y_val_classes, predictions_classes, average='weighted', zero_division=1)
    f1 = f1_score(y_val_classes, predictions_classes, average='weighted')
    
    # Armazenar métricas de treino e validação
    train_accuracies.append(train_results[1])
    val_accuracies.append(val_results[1])
    train_losses.append(train_results[0])
    val_losses.append(val_results[0])
    
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# Calcular a média e o desvio padrão das métricas de treino e validação
mean_train_accuracy = np.mean(train_accuracies)
std_train_accuracy = np.std(train_accuracies)
mean_val_accuracy = np.mean(val_accuracies)
std_val_accuracy = np.std(val_accuracies)
mean_train_loss = np.mean(train_losses)
std_train_loss = np.std(train_losses)
mean_val_loss = np.mean(val_losses)
std_val_loss = np.std(val_losses)

mean_precision = np.mean(precisions)
std_precision = np.std(precisions)
mean_recall = np.mean(recalls)
std_recall = np.std(recalls)
mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

print(f"Resultados da validação cruzada ({n_splits} folds):")
print(f"Acurácia média de treino: {mean_train_accuracy:.4f} ± {std_train_accuracy:.4f}")
print(f"Acurácia média de teste: {mean_val_accuracy:.4f} ± {std_val_accuracy:.4f}")
print(f"Perda média de treino: {mean_train_loss:.4f} ± {std_train_loss:.4f}")
print(f"Perda média de teste: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
print(f"Precisão média: {mean_precision:.4f} ± {std_precision:.4f}")
print(f"Revocação média: {mean_recall:.4f} ± {std_recall:.4f}")
print(f"F1-Score médio: {mean_f1:.4f} ± {std_f1:.4f}")