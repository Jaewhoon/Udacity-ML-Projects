from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("path_to_image", help = "Enter the path of the image", type = str)
parser.add_argument("saved_model", help = "name of saved model", type = str)
parser.add_argument("--top_number", help = "number of k (top choices)", type = int, default = 5)
parser.add_argument("--category_names", help = "Path to a JSON file mapping labels", type = str, default = "label_map.json")
args = parser.parse_args()
image_path = args.path_to_image
model = args.saved_model
top_k = args.top_number
json_file = args.category_names

with open(json_file, 'r') as f:
    class_names = json.load(f)

reloaded_keras_model = tf.keras.models.load_model(model, custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)

image_size = 224

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()

def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    image = np.expand_dims(image, axis = 0)
    ps = reloaded_keras_model.predict(image)[0]
    probs = np.sort(ps)[-5:]
    classes = [str(i) for i in np.argsort(ps)[-5:]]
    return probs, classes

probs, classes = predict(image_path, model, top_k)
top_classes = [class_names[str(int(x)+1)] for x in classes]
numb = len(top_classes)
for i in range(1, numb+1):
    print("{}\nName : {} (class : {})\nProbability : {}".format(i, top_classes[numb-i], classes[numb-i], probs[numb-i]))

print("\n\n\n")
print("Classes\n", classes)
print("Probabilities\n", probs)
print("Labels\n", top_classes)
