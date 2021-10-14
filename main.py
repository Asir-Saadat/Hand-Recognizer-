import cv2
import time
import mediapipe as mp

mpHand = mp.solutions.hands
hand = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils
cap_video=cv2.VideoCapture(0)

cTime=0
pTime=0

while True:
    success,img = cap_video.read() # Default, it stays in BGR mode, so we need to convert it to RGB

    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)

    result = hand.process(imgRGB)
    #print(result.multi_hand_landmarks) Gets the landmarks of the hands.

    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:

            for id,lm in enumerate(landmarks.landmark):
                #print(id)
                #print(lm.x,lm.y,lm.z)
                imw,imh,imc=img.shape
                cx,cy =int((lm.x*lm.y)), int(lm.y*imh)
                print(id,cx,cy)

                if id == 1:
                    cv2.circle(img,(cx,cy),15,(255,0,0),cv2.FILLED)

            #print(landmarks)
            mpDraw.draw_landmarks(img, landmarks, mpHand.HAND_CONNECTIONS)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime  # This means that, we can get the frame per second


    cv2.putText(img,str(int(fps)), (10,70),cv2.FONT_ITALIC,3,(255,0,250),3)
    cv2.imshow("Image",img)



    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): #If I press q, this means the vidoe fill stop.
        break