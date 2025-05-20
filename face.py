import dlib
import cv2
import numpy as np
import os

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def encode_faces(directory):
    encodings = {}
    for person in os.listdir(directory):
        person_path = os.path.join(directory, person)
        if not os.path.isdir(person_path):
            continue

        encodings[person] = []
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            img = cv2.imread(image_path)
            faces = detector(img)

            for face in faces:
                shape = sp(img, face)
                face_descriptor = facerec.compute_face_descriptor(img, shape)
                encodings[person].append(np.array(face_descriptor))

    return encodings

face_encodings = encode_faces("data")
np.save("face_encodings.npy", face_encodings)
