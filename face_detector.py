import cv2
#Load some pre-trained data on face frontals from opencv
trained_face_data= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('RDJ.png')
#Convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#Draw rectangles around the Face
(x,y,w,h) = face_coordinates[0]
cv2.rectangle(img,(x,y),(x+w,y+h),(255, 255, 0),7)
#print(face_coordinates)


cv2.imshow('Face detection',img)
cv2.waitKey()
print("Code completed")