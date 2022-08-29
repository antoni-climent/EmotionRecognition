# EmotionRecognition
This is a model trained to recognize seven types of emotion by watching face images. 
The emotions are: Anger, desgust, fearness, happyness, neutrality, sadness and surprise.
The training and test datasets used are uploaded in the train and test folders respectively and the Models folder contains some model weights saved. 

File description:
main.py holds the code used for the training phase 
reaTimeApp.py uses the model "trained_model2.h5" to decide which emotion is showing the face detected throw the pc camera. To define the square containing the face the library openCV is used.
