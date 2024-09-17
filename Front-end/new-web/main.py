from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
from datetime import datetime, timedelta
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import sys

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images/user'
app.secret_key = 'some_secret_key'  # Required for flash messages and session

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

DIAGNOSIS_FILE = 'diagnosis_data.json'

def get_time_ago(timestamp):
    now = datetime.now()
    diff = now - timestamp
    if diff < timedelta(minutes=1):
        return 'just now'
    elif diff < timedelta(hours=1):
        return f'{diff.seconds // 60} minutes ago'
    elif diff < timedelta(days=1):
        return f'{diff.seconds // 3600} hours ago'
    else:
        return timestamp.strftime('%B %d, %Y')

# Load the model
model = tf.keras.models.load_model('plant_village_model.h5')

# Load class names from labels.txt
with open('static/labels.txt', 'r') as f:
    class_names = [line.strip().split(' ', 1)[1] for line in f.readlines()]

def load_diagnosis_data():
    if os.path.exists(DIAGNOSIS_FILE):
        with open(DIAGNOSIS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_diagnosis_data(data):
    with open(DIAGNOSIS_FILE, 'w') as f:
        json.dump(data, f)

@app.route('/')
def home():
    images = []
    diagnosis_data = load_diagnosis_data()
    for f in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f)
        upload_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        images.append({
            'name': os.path.splitext(f)[0],
            'file': f,
            'date': get_time_ago(upload_time),
            'timestamp': upload_time  # Add this line to store the actual timestamp
        })
    
    # Sort images by timestamp in descending order (most recent first)
    images.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return render_template('index.html', images=images, diagnosis_data=diagnosis_data)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('home'))
    file = request.files['image']
    if file.filename == '':
        flash('No image selected', 'error')
        return redirect(url_for('home'))
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        flash('Image uploaded successfully', 'success')
        return redirect(url_for('diagnosis', filename=filename))

@app.route('/delete/<filename>')
def delete_image(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        # Remove diagnosis data for the deleted image
        diagnosis_data = load_diagnosis_data()
        if filename in diagnosis_data:
            del diagnosis_data[filename]
            save_diagnosis_data(diagnosis_data)
        flash('Image deleted successfully', 'success')
    else:
        flash('Image not found', 'error')
    return redirect(url_for('home'))

@app.route('/diagnosis/<filename>')
def diagnosis(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    diagnosis_data = load_diagnosis_data()
    
    if filename in diagnosis_data:
        return render_template('diagnosis.html', 
                               image_filename=filename, 
                               disease_name=diagnosis_data[filename]['disease_name'], 
                               accuracy=diagnosis_data[filename]['accuracy'])
    
    # Load and preprocess the image
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    
    # Make prediction
    predictions = model.predict(img_array)
    
    # Get the predicted class and accuracy
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    accuracy = float(100 * np.max(predictions))
    
    # Store the diagnosis data
    diagnosis_data[filename] = {
        'disease_name': predicted_class,
        'accuracy': f"{accuracy:.2f}"
    }
    save_diagnosis_data(diagnosis_data)
    
    return render_template('diagnosis.html', 
                           image_filename=filename, 
                           disease_name=predicted_class, 
                           accuracy=f"{accuracy:.2f}",)

if __name__ == '__main__':
    app.run(debug=True)