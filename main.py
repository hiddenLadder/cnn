# Импортирование библиотек
from keras import layers, models
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# Загрузка данных из MNIST и разбивка на обучающую и тестовую выборки
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Предобработка данных
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Создание модели
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation='softmax'))

# Настройка оптимизатора и функции потерь
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Подготовка данных для обучения с использованием генератора расширения данных
# rotation_range - диапазон для поворота изображения
# width_shift_range - диапазон для сдвига по горизонтали
# height_shift_range - диапазон для сдвига по вертикали
# zoom_range - для случайного изменения масштаба внутри изображений
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
datagen.fit(train_images)


# Обучение модели
batch_size = 64
epochs = 5
history = model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size),
                    steps_per_epoch=len(train_images) / batch_size, epochs=epochs,
                    validation_data=(test_images, test_labels))

# Сохранение модели
model.save('mnist_cnn_model.keras')

# Построение графиков изменения точности и потери
plt.plot(history.history['accuracy'], label='Точность на обучающей выборке')
plt.plot(history.history['val_accuracy'], label='Точность на валидационной выборке')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Потеря на обучающей выборке')
plt.plot(history.history['val_loss'], label='Потеря на валидационной выборке')
plt.xlabel('Эпохи')
plt.ylabel('Потеря')
plt.legend()
plt.show()

# Оценка точности модели
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Точность на тестовой выборке: {test_acc}')
print(f'Потеря на тестовой выборке: {test_loss}')

