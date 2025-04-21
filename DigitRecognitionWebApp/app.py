from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import cv2
import os
from io import BytesIO
import base64
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

app = Flask(__name__)

MODEL_PATH = 'handwrittenDigit_ann.keras'

if not os.path.exists(MODEL_PATH):
    print("Training ANN model...")

    (x_train, y_train), (x_test, y_test) = mnist.load_data()


    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.reshape(-1, 28 * 28)
    x_test = x_test.reshape(-1, 28 * 28)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(28 * 28,)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=100, batch_size=30, validation_data=(x_test, y_test))
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f" Test Accuracy: {test_acc * 100:.2f}%")

    model.save(MODEL_PATH)

    print("ANN model trained and saved!")

print("Loading ANN model...")
model = tf.keras.models.load_model(MODEL_PATH)

UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html', message='Upload an image or draw a digit to predict')

@app.route('/upload_predict', methods=['POST'])
def upload_predict():
    if 'file' not in request.files:
        return render_template('index.html', message='No file uploaded')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message='No file selected')

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))

    if np.mean(img) > 127:
        img = np.invert(img)

    img = img / 255.0
    img = img.reshape(1, 28 * 28)

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    return render_template('index.html', message=f'This digit is probably a {digit}', image=filepath)


@app.route('/draw_predict', methods=['POST'])
def draw_predict():
    data = request.get_json()


    image_data = data['image']
    image_data = image_data.split(",")[1]
    img_bytes = BytesIO(base64.b64decode(image_data))


    img = Image.open(img_bytes).convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img = np.array(img)

    if np.mean(img) > 127:
        img = np.invert(img)

    img = img / 255.0
    img = img.reshape(1, 28 * 28)

    prediction = model.predict(img)
    digit = np.argmax(prediction)

    return jsonify({'message': f'This digit is probably a {digit}'})


if __name__ == '__main__':
    app.run(debug=True)
