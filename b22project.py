    
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
    
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = int(np.abs(rad*180.0/np.pi))
    
    if ang >180.0:
        ang = 360-ang
        
    return ang

# Setting up mediapipe object

with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read() #The image is read and obtained in BGR format
        
        # Recoloring image to RGB for mp_pose.Pose to read
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #We set writeable flag to False so that image is not modified while being processed
        #as pose.process() expect read only image as input
        image.flags.writeable = False  
      
        # Making detection of body joints 
        results = pose.process(image)
    
        #Now we set it back to true which allows modification to the image 
        image.flags.writeable = True
        # Recolor back to BGR so that other OpenCV functions become compatible
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extracting landmarks of various body parts
        try:
            landmarks = results.pose_landmarks.landmark

            
            # Getting coordinates of the required points
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            # Calculating angle made by the three points
            angle = calculate_angle(shoulder, elbow, wrist)
            
            
            # Curl counter 
            if ang > 160:
                stg = "down"
            if ang < 30 and stg =='down':
                stg="up"
                cntr +=1
                       
        except:
            pass
        
        # Rendering the curl counter
        cv2.rectangle(image, (0,0), (225,73), (245,117,16),-1) #Creating the rectangle to
                                                               #to show the counter
        
        # Representing the required data 
        cv2.putText(image, 'REPS', (15,15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(cntr), 
                    (30,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)        
        cv2.putText(image, 'STAGE', (95,15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stg, 
                    (80,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
        
        
        # Rendering detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
