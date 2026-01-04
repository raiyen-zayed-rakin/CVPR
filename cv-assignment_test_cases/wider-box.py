import cv2
import numpy as np
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
from tensorflow import keras

# Load the saved model (.h5 format)
model = keras.models.load_model('face_recognition_model.h5')
print("./Model loaded successfully!")

# Option 1: Load from pickle file (requires scikit-learn)
try:
    import pickle
    with open('label_encoder.pkl', 'rb') as f:
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

# Initialize webcam
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")
print("Show your face to the webcam for recognition")

def preprocess_face(face_roi):
    """Preprocess the detected face for the model - matches notebook preprocessing"""
    # Resize to 224x224 to match model input
    resized = cv2.resize(face_roi, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 1] range - matching notebook normalization
    normalized = resized.astype('float32') / 255.0
    
    # Reshape to match model input: (1, 224, 224, 3)
    model_input = normalized.reshape(1, 224, 224, 3)
    
    return model_input, resized

def expand_bbox(x, y, w, h, frame_shape, scale_w=0.8, scale_h=1.2):
    """
    Expand bounding box to capture almost the entire upper body/head area
    
    Args:
        x, y, w, h: Original face bounding box
        frame_shape: Shape of the frame (height, width)
        scale_w: How much to expand horizontally (0.8 = 80% on each side - captures shoulders)
        scale_h: How much to expand vertically (1.2 = 120% extra on top for full head/hair)
    
    Returns:
        Expanded x, y, w, h
    """
    frame_height, frame_width = frame_shape[:2]
    
    # Calculate expansion - much larger now
    expand_w = int(w * scale_w)
    expand_h_top = int(h * scale_h)    # Lots of space on top for hair/head
    expand_h_bottom = int(h * 0.6)     # Some space on bottom for neck/shoulders
    
    # New coordinates
    new_x = max(0, x - expand_w)
    new_y = max(0, y - expand_h_top)
    new_w = min(frame_width - new_x, w + 2 * expand_w)
    new_h = min(frame_height - new_y, h + expand_h_top + expand_h_bottom)
    
    return new_x, new_y, new_w, new_h

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
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
        # Draw the original face detection box (green, thin)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
        # Expand bounding box to capture almost entire upper body/head area
        exp_x, exp_y, exp_w, exp_h = expand_bbox(x, y, w, h, frame.shape, 
                                                   scale_w=0.8, scale_h=1.2)
        
        # Extract expanded ROI (full head)
        face_roi = frame[exp_y:exp_y+exp_h, exp_x:exp_x+exp_w]
        
        if face_roi.size > 0:
            # Preprocess for model
            model_input, processed = preprocess_face(face_roi)
            
            # Make prediction
            predictions = model.predict(model_input, verbose=0)
            predicted_class = np.argmax(predictions)
            predicted_name = class_names[predicted_class]
            confidence = predictions[0][predicted_class]
            
            # Debug: Print all predictions
            print(f"\n=== Prediction Debug ===")
            for i, name in enumerate(class_names):
                print(f"{name}: {predictions[0][i]:.4f}")
            print(f"Winner: {predicted_name} ({confidence:.2%})")
            
            # Only show predictions with reasonable confidence
            if confidence > 0.5:  # Lowered threshold to see more predictions
                detections.append({
                    'bbox': (exp_x, exp_y, exp_w, exp_h),
                    'name': predicted_name,
                    'confidence': confidence,
                    'predictions': predictions[0],
                    'processed': processed
                })
    
    # Draw detections on frame
    for detection in detections:
        x, y, w, h = detection['bbox']
        name = detection['name']
        confidence = detection['confidence']
        
        # Choose color based on confidence
        if confidence > 0.9:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence > 0.7:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 165, 255)  # Orange for low confidence
        
        # Draw expanded bounding box around full head
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Draw label background
        label = f"{name}: {confidence:.2%}"
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
        "Green box = original face detection",
        "Thick box = expanded capture (almost entire pic)"
    ]
    
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)
        cv2.putText(frame, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 0, 0), 1)
    
    # Display the main frame
    cv2.imshow('Face Recognition', frame)
    
    # Display processed face (for debugging)
    if detections:
        processed_display = cv2.resize(detections[0]['processed'], (400, 400))
        cv2.imshow('Processed Face (What model sees)', processed_display)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Application closed")