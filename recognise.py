import os
import cv2 as cv
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

# loading model
model = model_from_json(open("fer.json", "r").read())
# loading weights
model.load_weights('fer.h5')

# cascade classifier to find faces in the image
fhc = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# starting the video
capture=cv.VideoCapture(0)

# looping over each frame one by one
while True:
    
    # capturing the frame ret = false if not captured
    ret,img=capture.read()
    
     if not ret:
         continue
    # converting in gray scale for further processing as cascade classifier deals with gray scale images
    g_img= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # using cascade classifier to find faces
    faces_found = fhc.detectMultiScale(g_img, 1.32, 5)

    # looping over all the faces found
    for (x,y,w,h) in faces_found:
        
        # drawing rectangle around the face
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        
        #cropping the face from the image and resizing
        face = g_img[y:y+w,x:x+h]
        face =cv2.resize(face,(48,48))
        
        img_pixels = image.img_to_array(face)
        
        # expanding dimension to 3 channel as gray has 2 channel only and model is trained on colored image
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255
        
        # getting the predictions
        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])
        
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]
        
        # putting text of emotion
        cv.putText(img, predicted_emotion, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    resized_img = cv.resize(img, (1000, 700))
    cv.imshow('Facial emotion analysis ',resized_img)

    if cv2.waitKey(10) == ord('q'):
         break

capture.release()
cv.destroyAllWindows
