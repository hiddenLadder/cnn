from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np

# Проверка обученной модели на изображении из интернета


# Обработка изображения
img = image.load_img('6.png', target_size=(28, 28))
img = img_to_array(img)[:,:,0].reshape((1, 28, 28, 1)).astype('float32') / 255

model = load_model('mnist_cnn_model.keras')

# Использование модели
result = model.predict(img)
predicted_class = np.argmax(result)
print(predicted_class)