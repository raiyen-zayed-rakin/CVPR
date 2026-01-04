import cv2
import numpy as np
import tensorflow as tf
print(tf.__version__)

from tensorflow import keras

# Load the saved model
model = keras.models.load_model('mnist_model.h5')  # or 'mnist_model.keras'
print("Model loaded successfully!")

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")
print("Show digits written large on white paper to the webcam")

def preprocess_digit(roi):
    """Preprocess the detected region for the model"""
    # Convert to grayscale if needed
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to get binary image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    # Resize to 28x28
    resized = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize
    normalized = resized.astype('float32') / 255.0
    
    return normalized, resized

def find_digit_contours(frame):
    """Find potential digit regions in the frame"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, thresh

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Find contours
    contours, thresh_display = find_digit_contours(frame)
    
    # Process each contour
    detections = []
    
    for contour in contours:
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter based on size and aspect ratio
        # Adjust these values based on your setup
        min_size = 50  # Minimum width/height
        max_size = 400  # Maximum width/height
        min_aspect = 0.3  # Minimum aspect ratio (w/h)
        max_aspect = 3.0  # Maximum aspect ratio (w/h)
        
        aspect_ratio = w / float(h) if h > 0 else 0
        
        # Check if contour meets criteria
        if (min_size < w < max_size and 
            min_size < h < max_size and 
            min_aspect < aspect_ratio < max_aspect):
            
            # Add padding around the digit
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(width, x + w + padding)
            y2 = min(height, y + h + padding)
            
            # Extract ROI
            roi = frame[y1:y2, x1:x2]
            
            if roi.size > 0:
                # Preprocess for model
                normalized, processed = preprocess_digit(roi)
                model_input = normalized.reshape(1, 28*28)
                
                # Make prediction
                predictions = model.predict(model_input, verbose=0)
                predicted_digit = np.argmax(predictions)
                confidence = predictions[0][predicted_digit]
                
                # Only show predictions with reasonable confidence
                if confidence > 0.95:  # Confidence threshold
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'digit': predicted_digit,
                        'confidence': confidence,
                        'processed': processed
                    })
    
    # Draw detections on frame
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        digit = detection['digit']
        confidence = detection['confidence']
        
        # Choose color based on confidence
        if confidence > 0.9:
            color = (0, 255, 0)  # Green for high confidence
        elif confidence > 0.7:
            color = (0, 255, 255)  # Yellow for medium confidence
        else:
            color = (0, 165, 255)  # Orange for low confidence
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background
        label = f"{digit}: {confidence:.2%}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 0, 0), 2)
    
    # Display instructions
    instructions = [
        "Press 'q' to quit",
        f"Detections: {len(detections)}"
    ]
    
    for i, text in enumerate(instructions):
        cv2.putText(frame, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)
        cv2.putText(frame, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 0, 0), 1)
    
    # Display the main frame
    cv2.imshow('Digit Detection', frame)
    
    # Display threshold view (for debugging)
    cv2.imshow('Threshold View', thresh_display)
    
    # Display processed digits
    if detections:
        # Show the first detected digit's processed version
        processed_img = detections[0]['processed']
        processed_display = cv2.resize(processed_img, (140, 140))
        cv2.imshow('Processed Digit', processed_display)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Application closed")