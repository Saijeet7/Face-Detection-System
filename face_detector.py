import cv2
from random import randrange
#Load some pre-trained data on face frontals from opencv
trained_face_data= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#img = cv2.imread('download.jpg')
#Convert to grayscale
webcam = cv2.VideoCapture(0)

while True:
    ########Read Current frame
    successful_frame_read, frame = webcam.read()
    ###Convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#Draw rectangles around the Face
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(randrange(256), randrange(256), randrange(256)),3)

    cv2.imshow('Face detection',frame)
    key = cv2.waitKey(1)

    #Press Q for quit
    if key ==81 or key==113:
        break

webcam.release()



#Detect Faces
#face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#Draw rectangles around the Face
# for (x,y,w,h) in face_coordinates:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(256), randrange(256), randrange(256)),3)
#print(face_coordinates)


# cv2.imshow('Face detection',img)
# cv2.waitKey()
print("Code completed")