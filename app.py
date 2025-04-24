from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model('waste_classifier_model.h5')

# Class labels
class_labels = ['cardboard', 'fruitpeel', 'garden', 'paper', 'plastic', 'trash', 'vegetable']

# Nutrient database
nutrient_db = {
    'vegetable': {
        'Nitrogen (N)': '2.5-4% of dry weight',
        'Phosphorus (P)': '0.3-0.8% of dry weight',
        'Potassium (K)': '3-6% of dry weight',
        'Carbon:Nitrogen (C:N)': '15:1 (Ideal for composting)'
    },
    'fruitpeel': {
        'Nitrogen (N)': '1.5-3% of dry weight',
        'Potassium (K)': '8-12% of dry weight',
        'Calcium (Ca)': '0.5-2% of dry weight'
    },
    'garden': {
        'Nitrogen (N)': '1.5-3% of dry weight',
        'Phosphorus (P)': '0.2-0.5% of dry weight',
        'Silica (Si)': '2-5% (Strengthens plant cells)'
    },
    'paper': {
        'Carbon:Nitrogen (C:N)': '200:1 (High carbon)',
        'Lignin Content': '20-30% (Slow to decompose)'
    },
    'cardboard': {
        'Carbon:Nitrogen (C:N)': '350:1 (Very high carbon)',
        'Lignin Content': '25-35%'
    }
}

# Soil database
soil_db = {
    'clay': {
        'deficiencies': ['Nitrogen (N)', 'Phosphorus (P)', 'Organic Matter'],
        'suitable_wastes': ['fruitpeel', 'vegetable', 'garden']
    },
    'sandy': {
        'deficiencies': ['Potassium (K)', 'Magnesium (Mg)', 'Water Retention'],
        'suitable_wastes': ['cardboard', 'paper', 'garden'] 
    },
    'loamy': {
        'deficiencies': ['Calcium (Ca)', 'Sulfur (S)'],
        'suitable_wastes': ['vegetable', 'fruitpeel']
    },
    'silty': {
        'deficiencies': ['Zinc (Zn)', 'Manganese (Mn)'],
        'suitable_wastes': ['fruitpeel', 'garden']
    }
}

def get_soil_recommendations(waste_type, soil_type):
    waste_nutrients = nutrient_db.get(waste_type, {})
    soil_needs = soil_db.get(soil_type, {}).get('deficiencies', [])
    
    recommendations = []
    for nutrient, value in waste_nutrients.items():
        if any(nutrient.startswith(def_nutrient) for def_nutrient in soil_needs):
            recommendations.append(f"• {nutrient}: {value}")
    return recommendations or ["No significant nutrient match"]

def preprocess_image(img):
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html', soil_db=soil_db)

@app.route('/soil-analysis')
def soil_analysis():
    return render_template('soil_analysis.html', soil_db=soil_db)

@app.route('/predict', methods=['POST'])
def predict():
    analysis_type = request.form.get('analysis_type', 'waste')
    
    if analysis_type == 'soil':
        soil_type = request.form.get('soil_type')
        deficiencies = soil_db.get(soil_type, {}).get('deficiencies', [])
        suitable_wastes = soil_db.get(soil_type, {}).get('suitable_wastes', [])
        
        recommendations = []
        for waste in suitable_wastes:
            if waste in nutrient_db:
                recommendations.append({
                    'type': waste,
                    'nutrients': nutrient_db[waste]
                })
        
        return render_template('soil_result.html', 
                            soil_type=soil_type,
                            deficiencies=deficiencies,
                            recommendations=recommendations)
    
    else:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                img = Image.open(filepath)
        elif 'file' in request.form:
            base64_str = request.form['file']
            if base64_str.startswith('data:image'):
                base64_str = base64_str.split(',')[1]
            img_data = base64.b64decode(base64_str)
            img = Image.open(BytesIO(img_data))
            filename = 'camera_capture.jpg'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(filepath)
        else:
            return redirect(url_for('home'))
        
        img_array = preprocess_image(img)
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        soil_type = request.form.get('soil_type', 'loamy')
        
        if predicted_class in ['plastic', 'trash']:
            result = {
                'class': predicted_class,
                'message': '🚫 Non-biodegradable! Dispose properly.',
                'image_path': filepath
            }
        else:
            result = {
                'class': predicted_class,
                'message': '✅ Biodegradable! Good for composting.',
                'nutrients': nutrient_db.get(predicted_class, {}),
                'image_path': filepath,
                'soil_recommendations': get_soil_recommendations(predicted_class, soil_type),
                'soil_type': soil_type
            }
        
        return render_template('index.html', result=result, soil_db=soil_db)

if __name__ == '__main__':
    app.run(debug=True)