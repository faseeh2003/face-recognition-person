
import face_recognition
import cv2
import os
import pyttsx3
import sys
print("Current working directory:", os.getcwd())
import sys
print("Python executable:", sys.executable)
print("Python path:", sys.path)


print("face")
print("Starting face recognition...")

engine = pyttsx3.init()
engine.setProperty('rate', 130)  

def speak(text):
    """Speak out the given text"""
    engine.say(text)
    engine.runAndWait()

known_faces_dir = known_faces_dir = os.path.join(os.getcwd(), 'dataset')
tolerance = 0.6
model = "hog"  

known_faces = []
known_names = []

for name in os.listdir(known_faces_dir):
    person_dir = os.path.join(known_faces_dir, name)
    if not os.path.isdir(person_dir):
        continue
    for filename in os.listdir(person_dir):
        image_path = os.path.join(person_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            known_faces.append(encodings[0])
            known_names.append(name)

print(f"Loaded {len(known_faces)} known faces.")
speak(f"Loaded {len(known_faces)} known faces.")

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    locations = face_recognition.face_locations(small_frame, model=model)
    encodings = face_recognition.face_encodings(small_frame, locations)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, tolerance)
        name = "Unknown"

        if True in results:
            match_index = results.index(True)
            name = known_names[match_index]

        print(f"Identified: {name}")
        speak(name)
        video.release()
        cv2.destroyAllWindows()
        sys.exit()

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
