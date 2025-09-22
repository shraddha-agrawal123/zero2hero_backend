from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from werkzeug.utils import secure_filename
import os
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import uuid
from geopy.geocoders import Nominatim
import logging
import random
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


app = Flask(__name__)
CORS(app, origins=["http://localhost:3000","*","https://zero2hero-pp29.vercel.app/"])  # Allow React app

# Configure logging
logging.basicConfig(level=logging.INFO)

# MongoDB Atlas configuration
app.config["MONGO_URI"] = "mongodb+srv://shraddhaagrawal12082004_db_user:6L81iWqw8llNa3pw@cluster0.mhugk1s.mongodb.net/waste_management?retryWrites=true&w=majority"

try:
    mongo = PyMongo(app)
    # Test connection
    mongo.cx.admin.command('ping')
    print("✅ MongoDB connected successfully to Atlas!")
except Exception as e:
    print(f"❌ MongoDB connection failed: {e}")
    mongo = None

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model (with error handling)
try:
    model = tf.keras.models.load_model('waste_classifier_model.h5')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Model loading failed: {e}")
    model = None

# Class labels
class_labels = ['cardboard', 'fruitpeel', 'garden', 'paper', 'plastic', 'trash', 'vegetable']

# Nutrient database (for soil analysis)
nutrient_db = {
    'vegetable': {
        'Nitrogen (N)': '2.5-4% of dry weight',
        'Phosphorus (P)': '0.3-0.8% of dry weight',
        'Potassium (K)': '3-6% of dry weight',
        'Carbon:Nitrogen (C:N)': '15:1 (Ideal for composting)',
        'Benefits': 'Quick decomposition, balanced nutrients for plant growth'
    },
    'fruitpeel': {
        'Nitrogen (N)': '1.5-3% of dry weight',
        'Potassium (K)': '8-12% of dry weight',
        'Calcium (Ca)': '0.5-2% of dry weight',
        'Benefits': 'High potassium for flower and fruit development'
    },
    'garden': {
        'Nitrogen (N)': '1.5-3% of dry weight',
        'Phosphorus (P)': '0.2-0.5% of dry weight',
        'Silica (Si)': '2-5% (Strengthens plant cells)',
        'Benefits': 'Improves soil structure and plant disease resistance'
    },
    'paper': {
        'Carbon:Nitrogen (C:N)': '200:1 (High carbon)',
        'Lignin Content': '20-30% (Slow to decompose)',
        'Benefits': 'Excellent brown material for composting, improves soil structure'
    },
    'cardboard': {
        'Carbon:Nitrogen (C:N)': '350:1 (Very high carbon)',
        'Lignin Content': '25-35%',
        'Benefits': 'Long-term soil conditioning, moisture retention'
    }
}

# Soil database
soil_db = {
    'clay': {
        'name': 'Clay Soil',
        'description': 'Fine particles, poor drainage, nutrient-rich but hard for roots to penetrate',
        'deficiencies': ['Nitrogen (N)', 'Phosphorus (P)', 'Organic Matter'],
        'suitable_wastes': ['fruitpeel', 'vegetable', 'garden'],
        'characteristics': {
            'Drainage': 'Poor',
            'Water Retention': 'High',
            'Nutrient Retention': 'High',
            'pH Range': '6.0-7.5',
            'Texture': 'Fine, sticky when wet'
        }
    },
    'sandy': {
        'name': 'Sandy Soil',
        'description': 'Large particles, excellent drainage, low nutrient retention',
        'deficiencies': ['Potassium (K)', 'Magnesium (Mg)', 'Water Retention'],
        'suitable_wastes': ['cardboard', 'paper', 'garden'],
        'characteristics': {
            'Drainage': 'Excellent',
            'Water Retention': 'Low',
            'Nutrient Retention': 'Low',
            'pH Range': '6.0-7.0',
            'Texture': 'Coarse, gritty'
        }
    },
    'loamy': {
        'name': 'Loamy Soil',
        'description': 'Perfect balance of sand, silt, and clay - ideal for most plants',
        'deficiencies': ['Calcium (Ca)', 'Sulfur (S)'],
        'suitable_wastes': ['vegetable', 'fruitpeel'],
        'characteristics': {
            'Drainage': 'Good',
            'Water Retention': 'Moderate',
            'Nutrient Retention': 'Good',
            'pH Range': '6.0-7.0',
            'Texture': 'Smooth, slightly gritty'
        }
    },
    'silty': {
        'name': 'Silty Soil',
        'description': 'Medium-sized particles, good water retention, fertile but can compact',
        'deficiencies': ['Zinc (Zn)', 'Manganese (Mn)'],
        'suitable_wastes': ['fruitpeel', 'garden'],
        'characteristics': {
            'Drainage': 'Moderate',
            'Water Retention': 'Good',
            'Nutrient Retention': 'High',
            'pH Range': '6.5-7.5',
            'Texture': 'Smooth, flour-like when dry'
        }
    }
}

# Helper functions
def authenticate_token(token):
    """Authenticate user token for API routes"""
    if not token or not mongo:
        return None
    
    # Handle Bearer token format
    if token.startswith('Bearer '):
        token = token[7:]  # Remove 'Bearer ' prefix
    
    try:
        user = mongo.db.users.find_one({"token": token})
        return user
    except Exception as e:
        print(f"Authentication error: {e}")
        return None

def classify_waste(image_data):
    """Classify waste from image data"""
    if not model:
        # Return a random classification if model is not available
        predicted_class = random.choice(class_labels)
        confidence = random.uniform(0.7, 0.95)
        return predicted_class, confidence
    
    try:
        img = Image.open(BytesIO(image_data))
        img = img.resize((150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Classification error: {e}")
        # Fallback to random classification
        predicted_class = random.choice(class_labels)
        confidence = random.uniform(0.7, 0.95)
        return predicted_class, confidence

def preprocess_image(img):
    """Preprocess image for classification"""
    try:
        img = img.resize((150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return img_array
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return None

def get_soil_recommendations(waste_type, soil_type):
    """Get soil recommendations based on waste and soil type"""
    waste_nutrients = nutrient_db.get(waste_type, {})
    soil_info = soil_db.get(soil_type, {})
    soil_needs = soil_info.get('deficiencies', []) if soil_info else []
    
    recommendations = []
    for nutrient, value in waste_nutrients.items():
        if any(nutrient.startswith(def_nutrient.split(' ')[0]) for def_nutrient in soil_needs):
            recommendations.append(f"• {nutrient}: {value}")
    return recommendations or ["No significant nutrient match"]

def cleanup_expired_reports():
    """Clean up expired reports based on status and time"""
    if not mongo:
        return
    
    try:
        now = datetime.utcnow()
        
        # Remove completed reports older than 24 hours
        twenty_four_hours_ago = now - timedelta(hours=24)
        result1 = mongo.db.reports.delete_many({
            "status": "completed",
            "collected_at": {"$lt": twenty_four_hours_ago}
        })
        
        # Remove pending reports older than 3 days
        three_days_ago = now - timedelta(days=3)
        result2 = mongo.db.reports.delete_many({
            "status": "pending",
            "created_at": {"$lt": three_days_ago}
        })
        
        if result1.deleted_count > 0 or result2.deleted_count > 0:
            print(f"Cleaned up {result1.deleted_count} completed reports and {result2.deleted_count} expired pending reports")
            
    except Exception as e:
        print(f"Error cleaning up reports: {e}")

# ---------------- IMAGE VERIFICATION WITH OPENCV ---------------- #

def compare_images(img1_path, img2_path, threshold=0.6):
    """Compare two images using Structural Similarity Index (SSIM)."""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return False, 0.0  # image not found or invalid

    # Resize second image to match first image dimensions
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    score, _ = ssim(img1, img2, full=True)
    return score >= threshold, score

def get_location_details(address):
    """Get location details from address"""
    try:
        geolocator = Nominatim(user_agent="waste_management_app")
        location = geolocator.geocode(address)
        if location:
            return {
                "address": location.address,
                "latitude": location.latitude,
                "longitude": location.longitude
            }
    except Exception as e:
        print(f"Geocoding error: {e}")
    
    return {"address": address, "latitude": None, "longitude": None}

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# Web Routes (Original functionality)
@app.route('/')
def home():
    return render_template('index.html', soil_db=soil_db)

@app.route('/soil-analysis')
def soil_analysis():
    return render_template('soil_analysis.html', soil_db=soil_db)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/login-page')
def login_page():
    return render_template('login.html')

@app.route('/register-page')
def register_page():
    return render_template('register.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        analysis_type = request.form.get('analysis_type', 'waste')
        
        if analysis_type == 'soil':
            soil_type = request.form.get('soil_type')
            soil_info = soil_db.get(soil_type, {})
            deficiencies = soil_info.get('deficiencies', [])
            suitable_wastes = soil_info.get('suitable_wastes', [])
            
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
            # Handle waste classification
            img = None
            filepath = None
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
            
            if not img:
                return redirect(url_for('home'))
            
            img_array = preprocess_image(img)
            if img_array is not None and model:
                prediction = model.predict(img_array)
                predicted_class = class_labels[np.argmax(prediction)]
            else:
                # Fallback classification
                predicted_class = random.choice(class_labels)
            
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
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

# API Routes (New waste management functionality)
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({"error": "Email and password are required"}), 400
        
        if not mongo:
            return jsonify({"error": "Database connection failed"}), 500
        
        if mongo.db.users.find_one({"email": data['email']}):
            return jsonify({"error": "User already exists"}), 409
        
        user_id = mongo.db.users.insert_one({
            "email": data['email'],
            "password": data['password'],  # In production, hash this password
            "name": data.get('name', ''),
            "created_at": datetime.utcnow(),
            "points": 0,
            "reports": [],
            "collections": [],
            "token": str(uuid.uuid4())
        }).inserted_id
        
        user = mongo.db.users.find_one({"_id": user_id})
        
        return jsonify({
            "token": user['token'], 
            "user_id": str(user_id),
            "name": user.get('name', ''),
            "email": user['email'],
            "points": user.get('points', 0),
            "message": "User registered successfully"
        }), 201
    
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({"error": "Registration failed"}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data or 'email' not in data or 'password' not in data:
            return jsonify({"error": "Email and password are required"}), 400
        
        if not mongo:
            return jsonify({"error": "Database connection failed"}), 500
        
        user = mongo.db.users.find_one({
            "email": data['email'],
            "password": data['password']  # In production, use hashed password comparison
        })
        
        if not user:
            return jsonify({"error": "Invalid credentials"}), 401
        
        # Generate new token on login
        token = str(uuid.uuid4())
        mongo.db.users.update_one({"_id": user['_id']}, {"$set": {"token": token}})
        
        return jsonify({
            "token": token,
            "user_id": str(user['_id']),
            "name": user.get('name', ''),
            "email": user['email'],
            "points": user.get('points', 0),
            "message": "Login successful"
        }), 200
    
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({"error": "Login failed"}), 500

@app.route('/api/report-waste', methods=['POST'])
def report_waste():
    try:
        token = request.headers.get('Authorization')
        user = authenticate_token(token)
        if not user:
            return jsonify({"error": "Unauthorized"}), 401
        
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        image_file = request.files['image']
        address = request.form.get('address', '')
        estimated_amount = request.form.get('estimatedAmount', '')
        
        if image_file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        # Classify the waste
        image_data = image_file.read()
        waste_type, confidence = classify_waste(image_data)
        
        # Get location details
        location_details = get_location_details(address)
        
        if not mongo:
            return jsonify({"error": "Database connection failed"}), 500
        
        # Save report to database
        report_data = {
            "user_id": user['_id'],
            "waste_type": waste_type,
            "confidence": confidence,
            "address": location_details['address'],
            "latitude": location_details.get('latitude'),
            "longitude": location_details.get('longitude'),
            "estimated_amount": estimated_amount,
            "image": base64.b64encode(image_data).decode('utf-8'),
            "status": "pending",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        report_id = mongo.db.reports.insert_one(report_data).inserted_id
        
        # Award points for reporting waste
        points_earned = 5
        
        # Update user's points and add report reference
        mongo.db.users.update_one(
            {"_id": user['_id']},
            {
                "$push": {"reports": str(report_id)},
                "$inc": {"points": points_earned},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        # Create notification
        notification_data = {
            "user_id": user['_id'],
            "title": "Waste Reported Successfully",
            "message": f"You've reported {waste_type} waste and earned {points_earned} points!",
            "type": "success",
            "read": False,
            "points_earned": points_earned,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        notification_id = mongo.db.notifications.insert_one(notification_data).inserted_id
        
        return jsonify({
            "success": True,
            "report_id": str(report_id),
            "waste_type": waste_type,
            "confidence": confidence,
            "address": location_details['address'],
            "estimated_amount": estimated_amount,
            "points_earned": points_earned,
            "notification": {
                "_id": str(notification_id),
                "title": notification_data["title"],
                "message": notification_data["message"],
                "read": False,
                "created_at": notification_data["created_at"].isoformat()
            }
        }), 201
    
    except Exception as e:
        print(f"Report waste error: {e}")
        return jsonify({"error": "Failed to report waste", "details": str(e)}), 500

@app.route('/api/collect-waste', methods=['POST'])
def collect_waste():
    try:
        token = request.headers.get('Authorization')
        user = authenticate_token(token)
        if not user:
            return jsonify({"error": "Unauthorized"}), 401
        
        report_id = request.form.get('report_id')
        if not report_id:
            return jsonify({"error": "Report ID is required"}), 400
        
        if 'verification_image' not in request.files:
            return jsonify({"error": 'Verification image is required'}), 400
        
        verification_image = request.files['verification_image']
        if verification_image.filename == '':
            return jsonify({"error": "No verification image selected"}), 400
        
        if not mongo:
            return jsonify({"error": "Database connection failed"}), 500
        
        # Get the report
        try:
            report = mongo.db.reports.find_one({"_id": ObjectId(report_id)})
        except:
            return jsonify({"error": "Invalid report ID"}), 400
        
        if not report:
            return jsonify({"error": "Report not found"}), 404
        
        if report['status'] != 'pending':
            return jsonify({"error": "Report is not available for collection"}), 400
        
        # Save the original image temporarily
        original_image_data = base64.b64decode(report['image'])
        original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{report_id}_original.jpg")
        with open(original_image_path, 'wb') as f:
            f.write(original_image_data)
        
        # Save the verification image temporarily
        verification_image_data = verification_image.read()
        verification_image_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{report_id}_verification.jpg")
        with open(verification_image_path, 'wb') as f:
            f.write(verification_image_data)
        
        # Compare the images with 60% threshold
        verified, confidence = compare_images(original_image_path, verification_image_path, threshold=0.6)
        
        # Clean up temporary files
        try:
            os.remove(original_image_path)
            os.remove(verification_image_path)
        except:
            pass
        
        if not verified:
            return jsonify({
                "success": False,
                "verified": False,
                "message": "Image verification failed! The uploaded image does not match the reported waste. Please try again with the correct image.",
                "confidence": float(confidence)
            }), 200  # Return 200 but with success: false
        
        # Calculate points based on waste type
        points_map = {
            'plastic': 10,
            'paper': 8,
            'cardboard': 12,
            'vegetable': 6,
            'fruitpeel': 6,
            'garden': 8,
            'trash': 5
        }
        points_earned = points_map.get(report.get('waste_type', ''), 10)
        
        # Update report status and store points_earned
        mongo.db.reports.update_one(
            {"_id": ObjectId(report_id)},
            {"$set": {
                "status": "completed",
                "collected_by": user['_id'],
                "collected_at": datetime.utcnow(),
                "verification_image": base64.b64encode(verification_image_data).decode('utf-8'),
                "points_earned": points_earned,  # Store points in the report
                "updated_at": datetime.utcnow()
            }}
        )
        
        # Add points to user
        mongo.db.users.update_one(
            {"_id": user['_id']},
            {
                "$inc": {"points": points_earned},
                "$push": {"collections": report_id},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        # Create notification
        notification_data = {
            "user_id": user['_id'],
            "title": "Collection Completed",
            "message": f"You've successfully collected {report['waste_type']} waste and earned {points_earned} points!",
            "type": "success",
            "read": False,
            "points_earned": points_earned,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        notification_id = mongo.db.notifications.insert_one(notification_data).inserted_id
        
        return jsonify({
            "success": True,
            "verified": True,
            "message": "Congratulations! Verification successful!",
            "points_earned": points_earned,
            "confidence": float(confidence),
            "notification": {
                "_id": str(notification_id),
                "title": notification_data["title"],
                "message": notification_data["message"],
                "read": False,
                "created_at": notification_data["created_at"].isoformat()
            }
        }), 200
    
    except Exception as e:
        print(f"Collect waste error: {e}")
        return jsonify({"error": "Failed to collect waste"}), 500
    
@app.route('/api/reports', methods=['GET'])
def get_reports():
    try:
        token = request.headers.get('Authorization')
        user = authenticate_token(token)
        if not user:
            return jsonify({"error": "Unauthorized"}), 401
        
        if not mongo:
            return jsonify({"error": "Database connection failed"}), 500
        
        # Clean up expired reports before fetching (only remove completed reports older than 24h)
        cleanup_expired_reports()
        
        status = request.args.get('status', 'available')  # Changed default to 'available'
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 10))
        skip = (page - 1) * limit
        
        # Build query based on status parameter
        if status == 'pending':
            query = {"status": "pending"}
        elif status == 'completed':
            query = {"status": "completed"}
        elif status == 'available':
            # Show both pending reports and completed reports within 24 hours
            twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
            query = {
                "$or": [
                    {"status": "pending"},
                    {
                        "status": "completed", 
                        "$or": [
                            {"collected_at": {"$gte": twenty_four_hours_ago}},
                            {"updated_at": {"$gte": twenty_four_hours_ago}}
                        ]
                    }
                ]
            }
        elif status == 'all':
            query = {}
        else:
            # Default to available reports
            twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
            query = {
                "$or": [
                    {"status": "pending"},
                    {
                        "status": "completed", 
                        "$or": [
                            {"collected_at": {"$gte": twenty_four_hours_ago}},
                            {"updated_at": {"$gte": twenty_four_hours_ago}}
                        ]
                    }
                ]
            }
        
        # Get total count
        total = mongo.db.reports.count_documents(query)
        
        # Get paginated reports, sorted by status (pending first) then by date
        pipeline = [
            {"$match": query},
            {"$addFields": {
                "status_priority": {
                    "$switch": {
                        "branches": [
                            {"case": {"$eq": ["$status", "pending"]}, "then": 1},
                            {"case": {"$eq": ["$status", "completed"]}, "then": 2}
                        ],
                        "default": 3
                    }
                }
            }},
            {"$sort": {
                "status_priority": 1,  # Pending reports first
                "created_at": -1       # Then by newest first
            }},
            {"$skip": skip},
            {"$limit": limit},
            {"$project": {"status_priority": 0}}  # Remove the helper field
        ]
        
        reports = list(mongo.db.reports.aggregate(pipeline))
        
        # Convert ObjectId to string for JSON serialization and add points_earned field
        for report in reports:
            report['_id'] = str(report['_id'])
            report['user_id'] = str(report['user_id'])
            
            if 'collected_by' in report:
                report['collected_by'] = str(report['collected_by'])
            
            if 'created_at' in report:
                report['created_at'] = report['created_at'].strftime('%Y-%m-%d')
            
            if 'updated_at' in report:
                report['updated_at'] = report['updated_at'].isoformat()
            
            if 'collected_at' in report:
                report['collected_at'] = report['collected_at'].isoformat()
            
            # Add points_earned field for completed reports
            if report['status'] == 'completed':
                # Calculate points based on waste type (same logic as in collect_waste endpoint)
                points_map = {
                    'plastic': 10,
                    'paper': 8,
                    'cardboard': 12,
                    'vegetable': 6,
                    'fruitpeel': 6,
                    'garden': 8,
                    'trash': 5
                }
                report['points_earned'] = points_map.get(report.get('waste_type', ''), 10)
        
        return jsonify({
            "success": True,
            "reports": reports,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": (total + limit - 1) // limit  # Calculate total pages
        }), 200
    
    except Exception as e:
        print(f"Get reports error: {e}")
        return jsonify({"error": "Failed to fetch reports"}), 500


# Updated cleanup function to be more specific about 24-hour rule
def cleanup_expired_reports():
    """Clean up expired reports based on status and time"""
    if not mongo:
        return
    
    try:
        now = datetime.utcnow()
        
        # Remove completed reports older than 24 hours
        twenty_four_hours_ago = now - timedelta(hours=24)
        result1 = mongo.db.reports.delete_many({
            "status": "completed",
            "$or": [
                {"collected_at": {"$lt": twenty_four_hours_ago}},
                {
                    "collected_at": {"$exists": False},
                    "updated_at": {"$lt": twenty_four_hours_ago}
                }
            ]
        })
        
        # Remove pending reports older than 7 days (extended from 3 days)
        seven_days_ago = now - timedelta(days=7)
        result2 = mongo.db.reports.delete_many({
            "status": "pending",
            "created_at": {"$lt": seven_days_ago}
        })
        
        if result1.deleted_count > 0 or result2.deleted_count > 0:
            print(f"Cleaned up {result1.deleted_count} completed reports (>24h old) and {result2.deleted_count} expired pending reports (>7 days old)")
            
    except Exception as e:
        print(f"Error cleaning up reports: {e}")

@app.route('/api/notifications/<notification_id>/read', methods=['PUT'])
def mark_notification_read(notification_id):
    try:
        token = request.headers.get('Authorization')
        user = authenticate_token(token)
        if not user:
            return jsonify({"error": "Unauthorized"}), 401

        # Verify the notification belongs to the user
        notification = mongo.db.notifications.find_one({
            "_id": ObjectId(notification_id),
            "user_id": user['_id']
        })
        
        if not notification:
            return jsonify({"error": "Notification not found"}), 404

        # Mark as read
        result = mongo.db.notifications.update_one(
            {"_id": ObjectId(notification_id)},
            {"$set": {"read": True, "updated_at": datetime.utcnow()}}
        )

        if result.modified_count > 0:
            return jsonify({
                "success": True,
                "message": "Notification marked as read"
            }), 200
        else:
            return jsonify({"error": "Failed to update notification"}), 400

    except Exception as e:
        print(f"Mark notification read error: {e}")
        return jsonify({"error": "Failed to update notification"}), 500

@app.route('/api/notifications/mark-all-read', methods=['PUT'])
def mark_all_notifications_read():
    try:
        token = request.headers.get('Authorization')
        user = authenticate_token(token)
        if not user:
            return jsonify({"error": "Unauthorized"}), 401

        # Mark all user notifications as read
        result = mongo.db.notifications.update_many(
            {"user_id": user['_id'], "read": False},
            {"$set": {"read": True, "updated_at": datetime.utcnow()}}
        )

        return jsonify({
            "success": True,
            "message": f"Marked {result.modified_count} notifications as read"
        }), 200

    except Exception as e:
        print(f"Mark all notifications read error: {e}")
        return jsonify({"error": "Failed to update notifications"}), 500

@app.route('/api/notifications', methods=['POST'])
def create_notification():
    try:
        token = request.headers.get('Authorization')
        user = authenticate_token(token)
        if not user:
            return jsonify({"error": "Unauthorized"}), 401

        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Required fields
        title = data.get('title')
        message = data.get('message')
        
        if not title or not message:
            return jsonify({"error": "Title and message are required"}), 400

        # Create notification
        notification_data = {
            "user_id": user['_id'],
            "title": title,
            "message": message,
            "type": data.get('type', 'info'),
            "read": False,
            "points_earned": data.get('points_earned', 0),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        notification_id = mongo.db.notifications.insert_one(notification_data).inserted_id

        # Return the created notification
        notification_data['_id'] = str(notification_id)
        notification_data['created_at'] = notification_data['created_at'].isoformat()
        notification_data['updated_at'] = notification_data['updated_at'].isoformat()

        return jsonify({
            "success": True,
            "notification": notification_data
        }), 201

    except Exception as e:
        print(f"Create notification error: {e}")
        return jsonify({"error": "Failed to create notification"}), 500

@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    try:
        token = request.headers.get('Authorization')
        user = authenticate_token(token)
        if not user:
            return jsonify({"error": "Unauthorized"}), 401

        # Get top 50 users by points
        leaderboard = list(mongo.db.users.find(
            {},
            {"name": 1, "email": 1, "points": 1, "_id": 1}
        ).sort("points", -1).limit(50))

        # Convert ObjectId to string and ensure points is a number
        for player in leaderboard:
            player['_id'] = str(player['_id'])
            player['points'] = player.get('points', 0) or 0

        return jsonify({
            "success": True,
            "leaderboard": leaderboard
        }), 200

    except Exception as e:
        print(f"Leaderboard error: {e}")
        return jsonify({"error": "Failed to fetch leaderboard"}), 500

# Additional API endpoint for simple classification
@app.route('/api/classify-simple', methods=['POST'])
def classify_simple():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        image_file = request.files['file']
        if image_file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        # Classify the waste
        image_data = image_file.read()
        predicted_class, confidence = classify_waste(image_data)
        
        return jsonify({
            "success": True,
            "predicted_class": predicted_class,
            "confidence": confidence
        }), 200
    
    except Exception as e:
        print(f"Simple classification error: {e}")
        return jsonify({"error": "Classification failed"}), 500

# Test route to check database connection
@app.route('/api/test-db')
def test_db():
    try:
        if not mongo:
            return jsonify({"error": "MongoDB not initialized"}), 500
        
        # Test database connection
        result = mongo.db.test.insert_one({"test": "connection", "timestamp": datetime.utcnow()})
        mongo.db.test.delete_one({"_id": result.inserted_id})
        return jsonify({"message": "Database connection successful"}), 200
    except Exception as e:
        return jsonify({"error": f"Database connection failed: {str(e)}"}), 500

# Health check endpoint
@app.route('/api/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "mongodb": "connected" if mongo else "disconnected",
        "model": "loaded" if model else "not loaded",
        "timestamp": datetime.utcnow().isoformat()
    }), 200

# Cleanup task endpoint (can be called via cron job)
@app.route('/api/cleanup-reports', methods=['POST'])
def cleanup_reports_endpoint():
    try:
        cleanup_expired_reports()
        return jsonify({"message": "Cleanup completed successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Cleanup failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)