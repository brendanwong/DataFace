import numpy as np
import cv2

# import pretrained classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# sets video source to default webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read() # Capture frame by frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect face coords and store
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)

    # display frame with drawn rectangle
    cv2.imshow('Video', frame)

    # search for face captured?
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    video_capture.release()
    cv2.destroyAllWindows()
'''
img = cv2.imread('dudes.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''