# Copyright (c) 2016 Tom Scanlan
# Inspired from code by Matthew Earl's faceswap and Rajeev
# Ratan's yawn counting program.

import cv2
import dlib

from emot_func import get_landmarks
from emot_func import mount_open
from emot_func import face_length
from emot_func import edge_lips
from emot_func import mouth_low
from emot_func import hunch_brows

#Detectable emotions: Surprise, Happy, Sad, Plain, Angry

#Surprise: toplip, bottomlip, height, mouth compute
#Happy: edgelips, width, smil compute
#Sad: edge lips lower than top
#Angry: upper eyes, top brows


PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)
yawns = 0
yawn_status = False

while True:
    ret, frame = cap.read()
    #image_landmarks, lip_distance, face_height_percent = mount_open(frame)
    landmarks = get_landmarks(frame)
    mouth_return = mount_open(landmarks)
    f_length = face_length(landmarks, frame.shape)
    hunch_brows(landmarks)
    print "Calculating Expression"
    a = (f_length[1] * frame.shape[1] * float(.065))
    b = mouth_low(landmarks)
    # 22 percent of the face height for mouth open
    if mouth_return[1] > (f_length[0]*frame.shape[0]*float(.2)):
        cv2.putText(frame, "Surprised Expression", (50,450), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

    # 45% threshold to smile
    elif edge_lips(landmarks, 0) > (f_length[1] * frame.shape[1] * float(.45)):
        cv2.putText(frame, "Happy Expression", (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)

    elif hunch_brows(landmarks) < (f_length[0] * frame.shape[0] * float(.1)):
        cv2.putText(frame, "Angry Expression", (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    elif a > b:
        cv2.putText(frame, "Sad Expression", (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (200, 50, 200), 2)

    else:
        cv2.putText(frame, "No Expression", (50, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Expression Detection', frame)

    #cv2.imshow('Live Landmarks', mouth_return[0])

    if cv2.waitKey(1) == 13:
        break
cap.release()
cv2.destroyAllWindows()