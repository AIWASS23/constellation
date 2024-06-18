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
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
import kerastuner as kt

data_folder = "images"
label_folder = "labels.csv"
shape = 50
# activation = 'relu'

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

# # Modelos 
# model1 = Sequential()
# model1.add(Input(shape=(shape, shape, 3)))
# model1.add(Conv2D(16, kernel_size=(3, 3), activation = 'relu'))
# model1.add(Conv2D(32, (3, 3), activation = 'relu'))
# model1.add(MaxPooling2D(pool_size=(2, 2)))
# model1.add(Conv2D(32, (3, 3), activation = 'relu'))
# model1.add(MaxPooling2D(pool_size=(2, 2)))
# model1.add(Conv2D(32, (3, 3), activation = 'relu'))
# model1.add(MaxPooling2D(pool_size=(2, 2)))
# model1.add(Conv2D(32, (3, 3), activation = 'relu'))
# model1.add(MaxPooling2D(pool_size=(2, 2)))
# model1.add(Dropout(0.25))
# model1.add(Flatten())
# model1.add(Dense(32, activation = 'relu'))
# model1.add(Dropout(0.5))
# model1.add(Dense(num_classes, activation='softmax'))

# # Compilar os modelos
# model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # https://www.tensorflow.org/api_docs/python/tf/keras/losses # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers

# # Treinar o modelo
# history1 = model1.fit(train_images, train_labels, epochs = 20, batch_size=32, validation_data = (test_images, test_labels))

def create_model(dropout_rate=0.0, optimizer='adam', activations = "relu", loss = 'categorical_crossentropy'):
    model = Sequential()
    model.add(Input(shape=(shape, shape, 3)))
    model.add(Conv2D(16, kernel_size=(3, 3), activation=activations))
    model.add(Conv2D(32, (3, 3), activation=activations))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation=activations))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation=activations))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation=activations))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(32, activation=activations))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss = loss, optimizer = optimizer, metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, verbose=0)
param_grid = {
    'batch_size': [16, 32, 64],
    'activation': [
        'elu', 'exponential', 'gelu', 'hard_sigmoid', 'hard_silu', 'hard_swish', 'leaky_relu',
        'linear', 'log_softmax', 'mish', 'relu', 'relu6', 'selu', 'sigmoid', 'silu', 'softmax',
        'softplus', 'softsign', 'swish', 'tanh'
    ],
    'epochs': [10],
    'dropout_rate': [0.25, 0.5, 0.75],
    'optimizer': [
        'Adadelta', 'Adafactor', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'Ftrl', 'Lion', 'Nadam', 
        'RMSprop', 'SGD'
    ],
    'loss': [
        'KLD', 'MAE', 'MAPE', 'MSE', 'MSLE', 'binary_crossentropy', 'binary_focal_crossentropy',
        'categorical_crossentropy', 'categorical_focal_crossentropy', 'categorical_hinge', 
        'cosine_similarity', 'ctc', 'dice', 'hinge', 'huber', 'logcosh', 'poisson',
        'sparse_categorical_crossentropy', 'squared_hinge', 'tversky'
    ]
}
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(train_images, train_labels)

print("Melhor: %f usando %s" % (grid_result.best_score_, grid_result.best_params_))


def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(shape, shape, 3)))
    model.add(Conv2D(16, kernel_size=(3, 3), activation=hp.Choice('activation', [
        'elu', 'exponential', 'gelu', 'hard_sigmoid', 'hard_silu', 'hard_swish', 'leaky_relu',
        'linear', 'log_softmax', 'mish', 'relu', 'relu6', 'selu', 'sigmoid', 'silu', 'softmax',
        'softplus', 'softsign', 'swish', 'tanh'
    ])))
    model.add(Conv2D(32, (3, 3), activation=hp.Choice('activation', [
        'elu', 'exponential', 'gelu', 'hard_sigmoid', 'hard_silu', 'hard_swish', 'leaky_relu',
        'linear', 'log_softmax', 'mish', 'relu', 'relu6', 'selu', 'sigmoid', 'silu', 'softmax',
        'softplus', 'softsign', 'swish', 'tanh'
    ])))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation=hp.Choice('activation', [
        'elu', 'exponential', 'gelu', 'hard_sigmoid', 'hard_silu', 'hard_swish', 'leaky_relu',
        'linear', 'log_softmax', 'mish', 'relu', 'relu6', 'selu', 'sigmoid', 'silu', 'softmax',
        'softplus', 'softsign', 'swish', 'tanh'
    ])))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hp.Float('dropout_rate', 0.0, 0.5, step=0.1)))
    model.add(Flatten())
    model.add(Dense(32, activation=hp.Choice('activation', [
        'elu', 'exponential', 'gelu', 'hard_sigmoid', 'hard_silu', 'hard_swish', 'leaky_relu',
        'linear', 'log_softmax', 'mish', 'relu', 'relu6', 'selu', 'sigmoid', 'silu', 'softmax',
        'softplus', 'softsign', 'swish', 'tanh'
    ])))
    model.add(Dropout(hp.Float('dropout_rate', 0.0, 0.5, step=0.1)))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer=hp.Choice('optimizer', [
            'Adadelta', 'Adafactor', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'Ftrl', 'Lion', 'Nadam', 
            'RMSprop', 'SGD'
        ]),
        loss=hp.Choice('loss', [
            'categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy'
        ]),
        metrics=['accuracy']
    )
    return model

# tuner = kt.BayesianOptimization(
#     build_model,
#     objective='val_accuracy',
#     max_trials=10,
#     executions_per_trial=3,
#     directory='my_dir',
#     project_name='intro_to_kt'
# )

# tuner.search(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
# print(f"Melhor hiperparâmetros: {best_hps}")

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=30,
    hyperband_iterations=2,
    directory='my_dir',
    project_name='intro_to_kt'
)

tuner.search(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Melhor hiperparâmetros: {best_hps}")