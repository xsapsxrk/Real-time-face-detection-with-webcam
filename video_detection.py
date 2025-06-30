import cv2 as cv
import numpy as np

face_cascade=cv.CascadeClassifier('haar_face.xml')
eyes_cascade=cv.CascadeClassifier('haar_eyes.xml')



def detectAndDisplay(frame):
    
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    
    
    #### For Face---->
    faces_ellipse=face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=7)
    for id,(x,y,w,h) in enumerate(faces_ellipse):
        center=(x+w//2,y+h//2)    
        frame=cv.ellipse(frame,center, (w//2, h//2), 0, 0, 360, (255, 0, 255),3)

    ###Putting text above the face---->
        label1=f"Face{id+1}"
        text_position1=(x,y-10)
        cv.putText(frame,label1,text_position1,cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1, cv.LINE_AA)


        faceROI = gray[y:y+h,x:x+w]
    ### For Eyes----->
        eyes=eyes_cascade.detectMultiScale(faceROI)
        for idx, (x2,y2,w2,h2) in enumerate(eyes):
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
        
        ###Putting text above the eyes---->
            label=f"Eye{idx+1}"
            text_position=(x+x2,y+y2 - 10)
            cv.putText(frame,label,text_position,cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1, cv.LINE_AA)

    cv.imshow('capture' ,frame)


cap=cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('(Error)No captured frame---Break')
        break


    detectAndDisplay(frame)

    if cv.waitKey(10) == 25:
        break

    

cap.release()
cv.destroyAllWindows()
