from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import os
import numpy as np
from PIL import Image
from tensorflow import keras
from PIL import Image, ImageChops, ImageEnhance

app = Flask(__name__)
run_with_ngrok(app)  # Use Ngrok to make the local web server accessible over the internet

model = keras.models.load_model('model_casia_run1.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    ela_image_path = None
    result = None
    confidence = None
    error = None

    if request.method == 'POST':
        if 'file' not in request.files:
            error = 'No file part'
        else:
            file = request.files['file']

            if file.filename == '':
                error = 'No selected file'
            else:
                filename = os.path.join('uploads', file.filename)
                file.save(filename)

                ela_image_path = convert_and_save_ela_image(filename)
                confidence, result = predict_fake_real(ela_image_path)

    return render_template('index.html', ela_image_path=ela_image_path, confidence=confidence, result=result, error=error)

def convert_to_ela_image(image_path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'
    
    # Open the image and save a temporary JPEG version with specified quality
    image = Image.open(image_path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality=quality)
    temp_image = Image.open(temp_filename)
    
    # Compute the difference between the original and the temporary image
    ela_image = ImageChops.difference(image, temp_image)
    
    # Get the extrema of the ELA image
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    
    # Normalize the ELA image to the maximum difference
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    # Save the ELA image
    ela_image.save(ela_filename)
    
    return ela_image

def prepare_image(image_path):
    image_size = (128, 128)
    ela_image = Image.open(image_path).resize(image_size)
    
    # Convert ELA image to a NumPy array and normalize
    ela_array = np.array(ela_image).flatten() / 255.0
    
    return ela_array

def convert_and_save_ela_image(image_path):
    ela_image = convert_to_ela_image(image_path, 90)
    ela_image_path = 'static/ela_images/ela_image.png'
    ela_image.save(ela_image_path)
    return ela_image_path

def predict_fake_real(ela_image_path):
    image_array = prepare_image(ela_image_path)
    image_array = image_array.reshape(1, 128, 128, 3)
    prediction = model.predict(image_array)
    
    confidence = round(np.max(prediction) * 100, 2)
    result = 'Fake' if np.argmax(prediction) == 0 else 'Real'
    
    return confidence, result

if __name__ == '__main__':
    app.run()
