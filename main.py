# Импортирование библиотек
from keras import layers, models
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import matplotlib.pyplot as plt

def main():
    # Загрузка данных из MNIST и разбивка на обучающую и тестовую выборки
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Предобработка данных (стандартизация)
    x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Создание модели
    # 1-й параметр - число фильтров
    # 2-й параметр - размер ядра каждого фильтра
    model = models.Sequential()
    # 32 фильтра с ядрами 3x3 пикселя каждый. 
    # (28, 28, 1) - формат изображения 28x28 пикселей с одним цветовым каналом (градации серого)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    # (2, 2) - размер окна, в котором выбирается максимальное значение
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # преобразует в одномерный вектор
    model.add(layers.Flatten())
    # выходной слой из 64 нейронов
    model.add(layers.Dense(64, activation='relu'))
    # 10 классов
    model.add(layers.Dense(10, activation='softmax'))
    # выключают случайным образом некоторые нейроны в процессе обучения модели
    # для того чтобы модель не переобучалась
    model.add(layers.Dropout(0.5))

    # print(model.summary())
    # return

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
    datagen.fit(x_train)


    # Обучение модели
    batch_size = 64 # кол-во образцов на обновление градиента
    epochs = 5 # кол-во итераций по всем предоставленным данным
    history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) / batch_size, epochs=epochs,
                        validation_data=(x_test, y_test))

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
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Точность на тестовой выборке: {test_acc}')
    print(f'Потеря на тестовой выборке: {test_loss}')

main()