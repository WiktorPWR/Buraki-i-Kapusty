import keras
import numpy
from keras.preprocessing import image

model = keras.models.load_model("amogrer.keras")
model = keras.Sequential([model, keras.layers.Softmax()])
img = image.load_img("data/Buraki/Image_87.jpg", target_size=(224, 224))
#img = image.load_img("data/Kapusta/Image_89.jpg", target_size=(224, 224))
img = image.img_to_array(img)
img = numpy.expand_dims(img, axis=0)

prediction_array = model.predict(img)[0]
print(prediction_array)