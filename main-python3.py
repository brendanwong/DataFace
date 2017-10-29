import numpy as numpy
import cv2



def main():
    # import pre-trained haar-cascade classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    # possible use for eye detection for increased accuracy?

    # sets video source to default webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()  # Capture frame by frame
        if ret is True:
            # if true, then it was read correctly
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale
        else:
            continue

        # store and detect face coordinates
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor = 1.7,
                                          minNeighbors = 5, minSize = (30, 30),
                                          flags = cv2.CASCADE_SCALE_IMAGE)

        # larger scale factor means smaller photo sample and faster face tracking/detection
        # faces returned as a list of rectangles

        for (x, y, w, h) in faces:
            # draw rectangle around each detected face
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
            # create a crop of just the face
            cropped = frame[y:y+h, x:x+w]

        try:
            # try to display the cropped photo
            cv2.imshow('cropped', cropped)

           # save
            cv2.imwrite('face.png', cropped)
            # from here, call functions to send to face api and parse incoming data
        except:
            # if error, just show the whole frame
            cv2.imshow('frame', frame)


        # search for face captured?
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
	main()