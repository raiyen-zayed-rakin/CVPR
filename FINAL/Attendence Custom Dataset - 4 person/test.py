import cv2
import os

INPUT_DIR = "navin_new"
OUTPUT_DIR = "navin"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

count = 0

for img_name in os.listdir(INPUT_DIR):
    img_path = os.path.join(INPUT_DIR, img_name)
    
    # Skip if not a file
    if not os.path.isfile(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (80, 80))
        save_path = os.path.join(OUTPUT_DIR, f"face_{count}.jpg")
        cv2.imwrite(save_path, face)
        count += 1

print(f"✅ Done! {count} face images saved to '{OUTPUT_DIR}/'")