
def right_rep():
    import cv2
    import mediapipe as mp
    import numpy as np
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    window_name = "Right_Rep"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 960)

    # variables for curl counter
    cntr = 0
    stg = None
    reset_message = ""
    reset_timer = 0

    def calculate_angle(a, b, c):
        a = np.array(a)  # top landmark
        b = np.array(b)  # mid landmark
        c = np.array(c)  # end landmark

        p1 = np.arctan2(c[1]-b[1], c[0]-b[0])
        p2 = np.arctan2(a[1]-b[1], a[0]-b[0])
        rad = p1 - p2

        ang = int(np.abs(rad*180.0/np.pi))
        if ang > 180.0:
            ang = 360-ang

        return ang

    # Setting up mediapipe object

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pos:
        while cap.isOpened():
            ret, frame = cap.read()
            # The image is read and obtained in BGR format

            # Recoloring image to RGB for mp_pose.Pose to read
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # We set writeable flag to False so that image
            # is not modified while being processed
            # as pose.process() expect read only image as input
            img.flags.writeable = False

            # Making detection of body joints
            results = pos.process(img)

            # Now we set it back to true which allows modification to the image
            img.flags.writeable = True
            # Recolor back to BGR so that other OpenCV functions work
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Getting coordinates of the required points
                rs = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                re = mp_pose.PoseLandmark.RIGHT_ELBOW.value
                rw = mp_pose.PoseLandmark.RIGHT_WRIST.value
                shoulder = [landmarks[rs].x, landmarks[rs].y]
                elbow = [landmarks[re].x, landmarks[re].y]
                wrist = [landmarks[rw].x, landmarks[rw].y]

                # Calculating angle made by the three points
                angle = calculate_angle(shoulder, elbow, wrist)

                # Curl counter
                if angle > 160:
                    stg = "down"
                if angle < 40 and stg == 'down':
                    stg = "up"
                    cntr += 1

            except:
                pass

            # Rendering the curl counter
            cv2.rectangle(img, (0, 0), (1280, 73),  # Creating the rectangle to
                          (0, 0, 0),-1)             # to show the counter

            # Representing the required data
            cv2.putText(img, 'REPS', (15, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, str(cntr),
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(img, 'STAGE', (95, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, stg,
                        (80, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # Render detections
            mp_drawing.draw_landmarks(
                                    img, results.pose_landmarks,
                                    mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(
                                        color=(245, 117, 66),
                                        thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(
                                        color=(245, 66, 230),
                                        thickness=2, circle_radius=2)
                                    )

            if reset_timer > 0:
                cv2.putText(img, reset_message,
                            (200, 60),
                            cv2.FONT_HERSHEY_TRIPLEX, 0.75,
                            (255, 255, 255), 1, cv2.LINE_AA)
                reset_timer -= 1

            cv2.imshow(window_name, img)

            key = cv2.waitKey(10)
            if key == ord('q'):
                break
            elif key == ord('r'):
                cntr = 0
                reset_message = "Counter has been reset"
                reset_timer = 2*15

        cap.release()
        cv2.destroyAllWindows()
