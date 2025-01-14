import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

# Carregar o dataset Cats vs Dogs
dataset, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)

# Função para pré-processar as imagens (redimensionar e normalizar)
def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))  # Redimensionar para 224x224
    image = tf.cast(image, tf.float32) / 255.0  # Normalizar para [0, 1]
    return image, label

# Função para preparar o dataset e retornar em lotes
def prepare_dataset(dataset, batch_size=32):
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# Dividir o dataset 'train' em treino e validação
train_data = dataset['train']

# Usando train_test_split do TensorFlow para dividir o conjunto de treino em treino e validação
train_dataset = train_data.take(15000)  # 15000 para treino
val_dataset = train_data.skip(15000)  # O restante para validação

# Preparar os datasets de treino e validação
train_dataset = prepare_dataset(train_dataset)
val_dataset = prepare_dataset(val_dataset)

# Carregar o modelo VGG16 pré-treinado, sem a camada final de classificação (include_top=False)
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar todas as camadas do VGG16
for layer in vgg.layers:
    layer.trainable = False

# Construir o modelo de Transfer Learning
model = models.Sequential()

# Adicionar as camadas do VGG16
model.add(vgg)

# Adicionar camadas adicionais de classificação
model.add(layers.Flatten())  # Achatar as saídas da última camada convolucional
model.add(layers.Dense(512, activation='relu'))  # Camada densa com 512 neurônios
model.add(layers.Dropout(0.5))  # Regularização
model.add(layers.Dense(1, activation='sigmoid'))  # Camada final de classificação binária (Gato ou Cachorro)

# Compilar o modelo
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Resumo do modelo
model.summary()

# Treinar o modelo
history = model.fit(
    train_dataset,
    epochs=10,  # Ajuste o número de épocas conforme necessário
    validation_data=val_dataset
)

# Visualizando a evolução da perda e precisão durante o treinamento
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plotando a perda de validação
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Loss durante o treinamento')
axes[0].set_xlabel('Épocas')
axes[0].set_ylabel('Perda')
axes[0].legend()

# Plotando a acurácia de validação
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[1].set_title('Acurácia durante o treinamento')
axes[1].set_xlabel('Épocas')
axes[1].set_ylabel('Acurácia')
axes[1].legend()

plt.show()