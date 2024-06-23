import os
import cv2 
from matplotlib import pyplot as plt
import math
import pandas as pd
import numpy as np
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LeakyReLU
from sklearn.model_selection import train_test_split
from skimage import io, color, filters

data_folder = "images"
label_folder = "labels.csv"

def main():
    # Carregar todas as imagens e labels
    images, labels, label_mapping = load_data(data_folder, label_folder)
    #Resize
    # images = resize_images(images)
    #Filter
    # images = apply_top_hat_transformation_to_dataset(images) #TOP-HAT
    # images = extract_lbp_features(images, labels)
    # images = extract_lbp_features(images, labels)
    # plot_svm_graph(images, labels)

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
    trainModel2(num_classes, train_images, train_labels, test_images, test_labels, label_mapping)

################################# KERAS #########################################
''' Model 01 '''
def createModel1(input_shape, num_classes):
    model1 = Sequential()
    model1.add(Conv2D(16, kernel_size=(3, 3), activation='tanh', input_shape=input_shape))
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

def create_model_grayscale(input_shape, num_classes):
    model = Sequential()
    
    # Primeira camada convolucional
    model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.1))
    
    # Segunda camada convolucional
    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Terceira camada convolucional
    model.add(Conv2D(128, (3, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Quarta camada convolucional
    model.add(Conv2D(128, (3, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Camada Flatten
    model.add(Flatten())
    
    # Primeira camada densa
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.5))
    
    # Segunda camada densa
    model.add(Dense(num_classes, activation='softmax'))
    
    model.summary()
    
    return model

def trainModel1(num_classes, train_images, train_labels, test_images, test_labels, label_mapping):
    # model1 = create_model((train_images.shape[1],) ,num_classes)
    model1 = createModel1((500, 500, 3), num_classes)
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

    y_test_classes = test_labels_classes
    accuracy = accuracy_score(y_test_classes, predictions1_classes)
    precision = precision_score(y_test_classes, predictions1_classes, average='macro')
    recall = recall_score(y_test_classes, predictions1_classes, average='macro')
    f1 = f1_score(y_test_classes, predictions1_classes, average='macro')

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-Score: {f1}')

    print('Confusion Matrix:')
    cm1 = confusion_matrix(test_labels_classes, predictions1_classes)
    plot_confusion_matrix(cm1, label_mapping, "Modelo 1")

''' Model 02 '''
def createModel2(input_shape, num_classes):
    model2 = Sequential()
    model2.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Conv2D(64, (3, 3), activation='relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Flatten())
    model2.add(Dense(128, activation='relu'))
    model2.add(Dense(num_classes, activation='softmax'))

    return model2

def trainModel2(num_classes, train_images, train_labels, test_images, test_labels, label_mapping):
    model2 = createModel2((500, 500, 3), num_classes) 

    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers

    model2.fit(
        train_images, train_labels,
        epochs=3,
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


def createModel3(input_shape, num_classes):
    model3 = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
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
    model3 = createModel3((500, 500, 3), num_classes)

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
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        filtered_images.append(thresh)

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

def apply_top_hat_transformation_to_dataset(images):
    filtered_images = []

    for image in images:
        image = apply_top_hat_transformation(image)
        filtered_images.append(image)

    return np.array(filtered_images)

def apply_erosion(image, kernel_size=(3,3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    eroded_img = cv2.erode(image, kernel, iterations=iterations)
    return eroded_img

def apply_dilation(image, kernel_size=(3, 3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    dilated_img = cv2.dilate(image, kernel, iterations=iterations)
    return dilated_img

def perform_local_mean_limiarization_filter(pixels_matrix):
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


######################## Attribute Extraction ####################################

def extract_hog_features(images):
    hog_features = []
    for baseImage in images:
        fd, hog_image = hog(
            baseImage,
            orientations=9,
            pixels_per_cell=(7, 7),
            cells_per_block=(2, 2), 
            block_norm='L2-Hys',
            visualize=True
        )

        # cv2.imshow("Hog Image", hog_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        hog_features.append(fd)
    
    hog_features = np.array(hog_features)
    hog_features = hog_features / np.max(hog_features)
    return hog_features

def extract_lbp_features(images, P=8, R=1):
    lbp_features = []
    for image in images:
        lbp = local_binary_pattern(image, P, R, method='uniform')
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
        lbp_features.append(hist)
    return np.array(lbp_features)

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
            img_array = image.img_to_array(img)#cv2.cvtColor(image.img_to_array(img), cv2.COLOR_BGR2GRAY)
            
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

# Exibir Matriz de Confusão
def plot_confusion_matrix(cm, label_mapping, model_name):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_mapping.keys())
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Matriz de Confusão - {model_name}")
    plt.show()

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

def plot_svm_graph(features, labels):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    reduced_data = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter, label='Classes')
    plt.title('Projeção 2D das Características HOG')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.show()

def resize_images(images, size=(50, 50)):
    resized_images = []
    for image in images:
        resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        resized_images.append(resized_image)

    # resized_images = np.array(resized_images)

    return resized_images

if __name__ == '__main__':
    main()