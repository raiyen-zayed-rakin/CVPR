import cv2
import numpy as np
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
from tensorflow import keras
from datetime import datetime
import os

# Load the saved model
model = keras.models.load_model('face_recognition_model(2) copy.h5')
print("Model loaded successfully!")

# Option 1: Load from pickle file (requires scikit-learn)
try:
    import pickle
    with open('label_encoder(2) copy.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    class_names = label_encoder.classes_
    print(f"Classes loaded from pickle: {class_names}")
except (ImportError, FileNotFoundError) as e:
    # Option 2: Hardcode class names (must be in ALPHABETICAL order!)
    print("Warning: Could not load pickle file, using hardcoded class names")
    class_names = ['avoy', 'navin', 'rakin', 'yousha']  # Alphabetical order!
    print(f"Classes: {class_names}")

# Load face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Text file setup
TEXT_FILE = 'face_detections.txt'
logged_faces = set()  # Track which faces have been logged this session

def initialize_text_file():
    """Initialize text file with header if it doesn't exist"""
    try:
        if not os.path.exists(TEXT_FILE):
            with open(TEXT_FILE, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("FACE DETECTION LOG\n")
                f.write("=" * 60 + "\n\n")
            print(f"✓ Created new text file: {TEXT_FILE}")
        else:
            print(f"✓ Using existing text file: {TEXT_FILE}")
        return True
    except Exception as e:
        print(f"✗ Error creating text file: {e}")
        return False

def log_to_text_file(name, confidence):
    """Log detected face to text file"""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(TEXT_FILE, 'a') as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Name: {name}\n")
            f.write(f"Confidence: {confidence:.2%}\n")
            f.write("-" * 60 + "\n\n")
        
        print(f"✓ Logged to file: {name} ({confidence:.2%}) at {timestamp}")
        return True
    except Exception as e:
        print(f"✗ Error logging to file: {e}")
        return False

def preprocess_face(face_roi):
    """Preprocess the detected face for the model - matches notebook preprocessing"""
    # Resize to 224x224 to match model input
    resized = cv2.resize(face_roi, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1] range - matching notebook normalization
    normalized = resized.astype('float32') / 255.0
    
    # Reshape to match model input: (1, 224, 224, 3)
    model_input = normalized.reshape(1, 224, 224, 3)
    
    return model_input, resized

# Initialize text file
print("\n--- Initializing Text File ---")
initialize_text_file()
print(f"Current directory: {os.getcwd()}")
print(f"Text file path: {os.path.abspath(TEXT_FILE)}")

# Initialize webcam
cap = cv2.VideoCapture(0)
print("\n--- Starting Face Recognition ---")
print("Press 'q' to quit")
print("Show your face to the webcam for recognition")
print("Faces with >98% confidence will be logged once\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )
    
    # Process each detected face
    detections = []
    
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        if face_roi.size > 0:
            # Preprocess for model
            model_input, processed = preprocess_face(face_roi)
            
            # Make prediction
            predictions = model.predict(model_input, verbose=0)
            predicted_class = np.argmax(predictions)
            predicted_name = class_names[predicted_class]
            confidence = predictions[0][predicted_class]
            
            # Check if confidence > 98% and not logged yet
            if confidence > 0.98 and predicted_name not in logged_faces:
                if log_to_text_file(predicted_name, confidence):
                    logged_faces.add(predicted_name)
                    print(f"First detection of {predicted_name} logged!")
            
            # Only show predictions with reasonable confidence
            if confidence > 0.8:  # Increased threshold - adjust between 0.6-0.9
                detections.append({
                    'bbox': (x, y, w, h),
                    'name': predicted_name,
                    'confidence': confidence,
                    'predictions': predictions[0],
                    'logged': predicted_name in logged_faces
                })
    
    # Draw detections on frame
    for detection in detections:
        x, y, w, h = detection['bbox']
        name = detection['name']
        confidence = detection['confidence']
        is_logged = detection['logged']
        
        # Choose color based on confidence
        if confidence > 0.9:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence > 0.7:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 165, 255)  # Orange for low confidence
        
        # Draw bounding box around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Draw label background with logged indicator
        label = f"{name}: {confidence:.2%}"
        if is_logged:
            label += " [LOGGED]"
        
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (x, y - label_h - 15), (x + label_w + 10, y), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, (0, 0, 0), 2)
        
        # Display all class probabilities below the face
        prob_y = y + h + 25
        for i, class_name in enumerate(class_names):
            prob_text = f"{class_name}: {detection['predictions'][i]:.2%}"
            cv2.putText(frame, prob_text, (x, prob_y + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, prob_text, (x, prob_y + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Display instructions
    instructions = [
        "Press 'q' to quit",
        f"Faces detected: {len(detections)}",
        f"Logged this session: {len(logged_faces)}"
    ]
    
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 255), 2)
        cv2.putText(frame, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 0), 1)
    
    # Display the main frame
    cv2.imshow('Face Recognition', frame)
    
    # Display processed face (for debugging)
    if detections and 'processed' in locals():
        processed_display = cv2.resize(processed, (224, 224))
        cv2.imshow('Processed Face', processed_display)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print(f"\n--- Application Closed ---")
print(f"Total faces logged this session: {len(logged_faces)}")
if logged_faces:
    print(f"Logged faces: {', '.join(logged_faces)}")
print(f"Log file location: {os.path.abspath(TEXT_FILE)}")