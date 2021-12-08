import cv2
#Load some pre-trained data on face frontals from opencv
trained_face_data= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('RDJ.png')
#Convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)



cv2.imshow('Face detection',grayscaled_img)
cv2.waitKey()
print("Code completed")