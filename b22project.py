import cv2
import mediapipe as mp
import numpy as np 
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
cap = cv2.VideoCapture(0)

# variables for curl counter
cntr = 0 
stg = None

def calculate_angle(a,b,c):
    a = np.array(a) # top landmark
    b = np.array(b) # mid landmark
    c = np.array(c) # end landmark
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = int(np.abs(radians*180.0/np.pi))
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 
