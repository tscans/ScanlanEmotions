import cv2
import dlib
import numpy as np

PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def get_landmarks(im):
    recognition = detector(im,1)
    if len(recognition) > 1:
        return "error"
    if len(recognition) == 0:
        return "error"
    return np.matrix([[p.x,p.y] for p in predictor(im, recognition[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0,0], point[0,1])
        cv2.putText(im, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=.4,color=(0,0,255))
        cv2.circle(im, pos, 3, color=(0,255,255))
    return im

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def edge_lips(landmarks, lorr):
    if landmarks == "error":
        return 0
    #48 right, 54 left
    right = [landmarks[48]]
    left = [landmarks[54]]
    right = np.squeeze(np.asarray(right))
    left = np.squeeze(np.asarray(left))
    if lorr == 0:
        edge_dist = left[lorr] - right[lorr]
        return edge_dist
    else:
        edge_average = (left[lorr] + right[lorr]) / 2
        return edge_average

def face_length(landmarks, image):
    if landmarks == "error":
        return [0,0]
    top_brow = []
    bottom_chin = []
    left_ear = []
    right_ear =[]
    #0-2, 14-16
    for i in range(18, 25):
        top_brow.append(landmarks[i])
    for i in range(7, 10):
        bottom_chin.append(landmarks[i])
    for i in range(0, 3):
        right_ear.append(landmarks[i])
    for i in range(14, 17):
        left_ear.append(landmarks[i])


    top_brow_all_pts = np.squeeze(np.asarray(top_brow))
    top_brow_mean = np.mean(top_brow, axis=0)

    bottom_chin_all_pts = np.squeeze(np.asarray(bottom_chin))
    bottom_chin_mean = np.mean(bottom_chin, axis=0)

    left_ear_all_pts = np.squeeze(np.asarray(left_ear))
    left_ear_mean = np.mean(left_ear, axis=0)

    right_ear_all_pts = np.squeeze(np.asarray(right_ear))
    right_ear_mean = np.mean(right_ear, axis=0)

    face_height = [int(top_brow_mean[:, 1]), int(bottom_chin_mean[:,1])]
    face_height = face_height[1] - face_height[0]
    face_height_percent = float(face_height) / float(image[0])

    face_width = [int(left_ear_mean[:, 0]), int(right_ear_mean[:, 0])]
    face_width = face_width[0] - face_width[1]
    face_width_percent = float(face_width) / float(image[1])
    return [face_height_percent, face_width_percent]

def mount_open(landmarks):
    if landmarks == "error":
        print "Too many or missing faces"
        return 0, 0
    #image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return 0, lip_distance

def mouth_wide(landmarks):
    if landmarks == "error":
        return 0, 0
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    return 0, lip_distance

def mouth_low(landmarks):
    if landmarks == "error":
        return 0, 0
    top_lip_center = top_lip(landmarks)
    edge_lips_avg = edge_lips(landmarks, 1)
    difference = edge_lips_avg - top_lip_center
    return difference

def hunch_brows(landmarks):
    if landmarks == "error":
        return 0
    # 18-20 right brow, 23-25 left brow, 37-38 right eye, 43-43 left eye
    right_brow = []
    left_brow = []
    right_eye = []
    left_eye = []
    for i in range(18, 21):
        right_brow.append(landmarks[i])
    for i in range(23, 26):
        left_brow.append(landmarks[i])
    for i in range(37,39):
        right_eye.append(landmarks[i])
    for i in range(43,45):
        left_eye.append(landmarks[i])

    right_brow = np.squeeze(np.asarray(right_brow))
    left_brow = np.squeeze(np.asarray(left_brow))
    right_eye = np.squeeze(np.asarray(right_eye))
    left_eye = np.squeeze(np.asarray(left_eye))

    right_brow = np.mean(right_brow, axis=0)
    left_brow = np.mean(left_brow, axis=0)
    right_eye = np.mean(right_eye, axis=0)
    left_eye = np.mean(left_eye, axis=0)

    right = int(right_eye[1]) - int(right_brow[1])
    left = int(left_eye[1]) - int(left_brow[1])
    avg_distance = (right + left) /2
    return avg_distance
