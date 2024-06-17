import os
import cv2 
from matplotlib import pyplot as plt
import math
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
from skimage import io, color, filters

data_folder = "images"
label_folder = "labels.csv"

def main():
    # Carregar todas as imagens e labels
    images, labels, label_mapping = load_data(data_folder, label_folder)

    #Filter
    images = apply_otsu_threshold_to_dataset(images) #OTSU

    #Spliting Dataset
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    # for (index, loadedImage) in enumerate(images):
    #     value = {i for i in label_mapping if label_mapping[i]==labels[index]}
    #     cv2.imshow(f"Image with label: {value}", np.array(loadedImage, dtype=np.uint8))
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # Normalizar os dados
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Obter o número de classes
    num_classes = len(label_mapping)

    # Converter as labels para one-hot encoding
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)
    trainModel1(num_classes, train_images, train_labels, test_images, test_labels, label_mapping)

################################# KERAS #########################################

''' Model 01 '''
def createModel1(num_classes):
    model1 = Sequential()
    model1.add(Conv2D(16, kernel_size=(3, 3), activation='tanh', input_shape=(500, 500, 1)))
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

    return model1

def trainModel1(num_classes, train_images, train_labels, test_images, test_labels, label_mapping):
    model1 = createModel1(num_classes)

    # Compilar os modelos
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # https://www.tensorflow.org/api_docs/python/tf/keras/losses

    # Treinar o modelo
    model1.fit(
        train_images, train_labels,
        epochs=1,
        batch_size=32,
        validation_data=(test_images, test_labels)
    )

    results_model1 = model1.evaluate(test_images, test_labels)

    predictions1 = model1.predict(test_images)
    predictions1_classes = np.argmax(predictions1, axis=1)

    print("Resultados do modelo 1:")
    print(f"Acurácia: {results_model1[1]}")
    print(f"Perda: {results_model1[0]}")

    test_labels_classes = np.argmax(test_labels, axis=1)

    cm1 = confusion_matrix(test_labels_classes, predictions1_classes)
    plot_confusion_matrix(cm1, label_mapping, "Modelo 1")

''' Model 02 '''
def createModel2(num_classes):
    model2 = Sequential()
    model2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 1)))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Conv2D(64, (3, 3), activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Flatten())
    model2.add(Dense(128, activation='relu'))
    model2.add(Dense(num_classes, activation='softmax'))

    return model2

def trainModel2(num_classes, train_images, train_labels, test_images, test_labels, label_mapping):
    model2 = createModel2(num_classes) 

    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers

    model2.fit(
        train_images, train_labels,
        epochs=1,
        batch_size=32,
        validation_data=(test_images, test_labels)
    )

    results_model2 = model2.evaluate(test_images, test_labels)

    print("Resultados do modelo 2:")
    print(f"Acurácia: {results_model2[1]}")
    print(f"Perda: {results_model2[0]}")

    predictions2 = model2.predict(test_images)
    predictions2_classes = np.argmax(predictions2, axis=1)

    test_labels_classes = np.argmax(test_labels, axis=1)

    cm2 = confusion_matrix(test_labels_classes, predictions2_classes)
    plot_confusion_matrix(cm2, label_mapping, "Modelo 2")


def createModel3(num_classes):
    model3 = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(500, 500, 1)),
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

    return model3

def trainModel3(num_classes, train_images, train_labels, test_images, test_labels, label_mapping):
    model3 = createModel3(num_classes)

    model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Treinar o modelo
    model3.fit(
        train_images, train_labels,
        epochs=1, # use com moderacão
        batch_size=32,
        validation_data=(test_images, test_labels)
    )

    # Avaliar os modelos nos dados de teste
    results_model3 = model3.evaluate(test_images, test_labels)

    # Imprimir os resultados
    print("Resultados do modelo 3:")
    print(f"Acurácia: {results_model3[1]}")
    print(f"Perda: {results_model3[0]}")

    # Previsões nos dados de teste
    predictions3 = model3.predict(test_images)

    # Converter previsões e labels para classe
    predictions3_classes = np.argmax(predictions3, axis=1)

    test_labels_classes = np.argmax(test_labels, axis=1)

    # Matriz de Confusão
    cm3 = confusion_matrix(test_labels_classes, predictions3_classes)
    plot_confusion_matrix(cm3, label_mapping, "Modelo 3")


############################## PRE PROCESSING ###################################

def apply_otsu_threshold_to_dataset(images):
    filtered_images = []

    for image in images:
        #Original
        cv2.imshow("Original", image)

        # image = perform_high_pass_laplacian_filter_opencv(image)
        # cv2.imshow("Lalacian", np.array(image, dtype=np.uint8))

        # image = apply_dilation(image)
        # cv2.imshow("Image Dilation", np.array(image, dtype=np.uint8))

        # 2 - Erosion
        # image = apply_erosion(image)
        # cv2.imshow("Erosion", np.array(image, dtype=np.uint8))

        image = apply_top_hat_transformation(image)
        cv2.imshow("TOP HAT", np.array(image, dtype=np.uint8))

        # image = perform_high_pass_laplacian_filter_opencv(image)
        # cv2.imshow("Erosion + Laplacian", np.array(image, dtype=np.uint8))

        # image = apply_dilation(image)
        # cv2.imshow("Erosio  + Laplacian + Dilation", np.array(image, dtype=np.uint8))
        # #3 - Dilation
        # image = apply_dilation(image)
        # cv2.imshow("Image Dilation", np.array(image, dtype=np.uint8))

        # #2 - Local Lim
        # image = np.array(perform_local_mean_limiarization_filter(image.tolist()), dtype=np.uint8)
        # cv2.imshow("Local Limiarization", image)

        # #3 - OTSU
        # _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow("Image threshold", np.array(thresh, dtype=np.uint8))

        cv2.waitKey(0)
        cv2.destroyAllWindows()
 
        filtered_images.append(image)

    return np.array(filtered_images)

def apply_adaptive_threshold_images_to_dataset(images, method='gaussian'):
    thresholded_images = []

    for gray_image in images:
        if method == 'mean':
            thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        else:
            thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        thresholded_images.append(thresh)
    return np.array(thresholded_images)

def apply_sauvola_threshold_images_to_dataset(images, window_size=11, k=0.2):
    thresholded_images = []

    for gray_image in images:
        gray_image = gray_image / 255.0

        sauvola_thresh = filters.threshold_sauvola(gray_image, window_size=window_size, k=k)
        binary_sauvola = gray_image > sauvola_thresh
        thresholded_images.append(binary_sauvola.astype(np.uint8) * 255)
    
    return np.array(thresholded_images)

def perform_high_pass_laplacian_filter_opencv(image):
    kernel_size = 3
    ddepth = cv2.CV_16S
    image_filtered = cv2.Laplacian(image, ddepth, ksize=kernel_size)
    image_filtered = cv2.convertScaleAbs(image_filtered)

    return image_filtered

def apply_top_hat_transformation(image, kernel_size=(15, 15)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    top_hat_image = cv2.subtract(image, opened_image)
    
    return top_hat_image

#################################################################################

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
            img_array = cv2.cvtColor(image.img_to_array(img), cv2.COLOR_BGR2GRAY)
            
            # Adicionar a imagem ao conjunto de dados
            images.append(img_array)
            
            # Extrair a label correspondente do arquivo CSV
            label = labels_df[labels_df['filename'] == filename]['class'].values[0]
            label_idx = label_mapping[label]
            labels.append(label_idx)
            # labels.append(label)
    
    # Converter para arrays numpy
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels)
    
    return images, labels, label_mapping

def perform_local_mean_limiarization_filter(pixels_matrix):
    #2x2
    # kernel = [
    #     [1, 1],
    #     [1, 1]
    # ]
    #3x3
    # kernel = [
    #     [1, 1, 1],
    #     [1, 1, 1],
    #     [1, 1, 1]
    # ]
    #5x5
    kernel = [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ]

    result_matrix = []

    for _ in range(len(pixels_matrix)):
        result_matrix.append([])

    for i in range(len(pixels_matrix)):
        for j in range(len(pixels_matrix[0])):
            colision_matrix = calculate_colision_matrix(pixels_matrix, kernel, i, j)
            pixels_mean = math.floor(calculate_matrix_mean(colision_matrix))

            if pixels_matrix[i][j] >= pixels_mean:
                result_matrix[i].append(255)
                continue
            result_matrix[i].append(0)
    
    return result_matrix

# Exibir Matriz de Confusão
def plot_confusion_matrix(cm, label_mapping, model_name):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_mapping.keys())
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.show()


def apply_erosion(image, kernel_size=(3,3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    eroded_img = cv2.erode(image, kernel, iterations=iterations)
    return eroded_img

def apply_dilation(image, kernel_size=(3, 3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    dilated_img = cv2.dilate(image, kernel, iterations=iterations)
    return dilated_img

def calculate_matrix_mean(matrix):
    sum = 0
    for i in matrix:
        sum += i
    
    return sum / len(matrix)

def calculate_colision_matrix(matrix, kernel, baseRow, baseCol):
    colision_matrix = []
    kernel_center_row = len(kernel) // 2
    kernel_center_col = len(kernel[0]) // 2

    for m in range(len(kernel)):
        for n in range(len(kernel[0])):
            row = baseRow + (m - kernel_center_row)
            col = baseCol + (n - kernel_center_col)

            if 0 <= row < len(matrix) and 0 <= col < len(matrix[0]):
                colision_matrix.append(matrix[row][col])

    return colision_matrix

if __name__ == '__main__':
    main()