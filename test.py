'''
PyPower Projects
Emotion Detection Using AI
'''

# USAGE: python test.py

from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import os

# Define data paths and labels
data_dir = "E:/AI/Emotion-Detection-master/archive"
images = []
labels = []

for emotion_folder in os.listdir(data_dir):
    emotion_path = os.path.join(data_dir, emotion_folder)
    for image_file in os.listdir(emotion_path):
        image_path = os.path.join(emotion_path, image_file)
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error loading image: {image_path}")
                continue
            img = cv2.resize(img, (48, 48))  # Adjust based on your model input size
            images.append(img)
            labels.append(emotion_folder)  # Assuming label is folder name
        except Exception as e:
            print(f"Error processing image: {image_path}")
            print(e)

face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier = load_model('./Emotion_Detection.h5')

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # make a prediction on the ROI, then lookup the class
            preds = classifier.predict(roi)[0]
            print("\nprediction = ", preds)
            label = class_labels[np.argmax(preds)]
            print("\nprediction max = ", np.argmax(preds))
            print("\nlabel = ", label)
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        print("\n\n")
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
