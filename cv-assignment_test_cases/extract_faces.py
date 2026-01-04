import cv2
import os

# Paths
INPUT_DIR = './22-46590-1'
OUTPUT_DIR = './22-46590-1-new'

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print("Cascade loaded:", not face_cascade.empty())
if face_cascade.empty():
    print("❌ Haar cascade not loaded")
    exit()

os.makedirs(OUTPUT_DIR, exist_ok=True)

for person in os.listdir(INPUT_DIR):
    person_path = os.path.join(INPUT_DIR, person)
    if not os.path.isdir(person_path):
        continue

    output_person_dir = os.path.join(OUTPUT_DIR, person)
    os.makedirs(output_person_dir, exist_ok=True)

    count = 0

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, (80, 80))

            save_path = os.path.join(
                output_person_dir, f"face_{count}.jpg"
            )
            cv2.imwrite(save_path, face)
            count += 1

    print(f"✅ {person}: {count} face images saved")
