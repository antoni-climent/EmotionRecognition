import cv2
import numpy as np
from keras.models import model_from_json

#Emotion list
emotion = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

# Load model generated in main.py
json_file = open('trained_model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Weight loading
emotion_model.load_weights("trained_model2.h5")

# Get web cam
my_cam = cv2.VideoCapture(0)

while True:
    # Searching for faces
    ret, frame = my_cam.read()
    frame = cv2.resize(frame, (1920, 1080))
    if not ret:
        break
    track_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face
    num_faces = track_face.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    #for (x, y, w, h) in num_faces:
    if(len(num_faces) != 0):
        x, y, w, h = num_faces[0] #There is only one face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 10, 100), 3)
        gray_face = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray_face, (48, 48)), -1), 0) #WIP

        # Use the model to make the prediction
        predict = emotion_model.predict(cropped_img)
        print(predict)
        ind = int(np.argmax(predict))

        #Write it into the frame before showing
        text = str(emotion[ind]) + ": " + str(np.amax(predict))
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (10, 255, 10), 3)

    cv2.imshow('Result', frame)

    c = cv2.waitKey(1)

my_cam.release()
cv2.destroyAllWindows()