import cv2 
import mediapipe as mp
import math 
import numpy as np

class poseDetection():

    def __init__(self, mode=False, model_complexity=1, upper_body = False, smooth_landmarks = True,
                min_detection_confidence = 0.5, min_tracking_confidence = 0.5):
        self.mode = mode # Video mode
        self.model_complexity = model_complexity # The higher the more accurate but slower
        self.upper_body = upper_body  # Detects the whole body not just the upper body
        self.smooth_landmarks = smooth_landmarks # Removing the noise and smoothing the landmarks, improving stability over time
        self.min_detection_confidence = min_detection_confidence # Score required to be considered successfull
        self.min_tracking_confidence = min_tracking_confidence # Score required for pose tracking to continue
        self.mediapipe_draw = mp.solutions.drawing_utils # Module for drawing landmarks on images
        self.mediapipe_pose = mp.solutions.pose # Pose module for post estimation
        self.posture = self.mediapipe_pose.Pose(self.mode,self.model_complexity,self.upper_body,self.smooth_landmarks,
                                                self.min_detection_confidence,self.min_tracking_confidence) # Creating instance of pose class
        self.landmark_list = self.mediapipe_pose.PoseLandmark # Stores the pose landmarks from mediapipe pose module into a list

    def backPosture(self, image, mp_results, exercise):
        landmarks = mp_results.pose_world_landmarks.landmark
        
        # Check if the relevant landmarks are detected
        if all(lm.visibility > self.min_detection_confidence for lm in [landmarks[11], landmarks[13], landmarks[14],
         landmarks[15], landmarks[12], landmarks[16], landmarks[23], landmarks[24]]):
            # Extracting the relevant landmarks
            nose = [landmarks[0].x, landmarks[0].y, landmarks[0].z]
            right_shoulder = [landmarks[11].x, landmarks[11].y, landmarks[11].z]
            left_shoulder = [landmarks[12].x, landmarks[12].y, landmarks[12].z]
            right_hip = [landmarks[23].x, landmarks[23].y, landmarks[23].z]
            left_hip = [landmarks[24].x, landmarks[24].y, landmarks[24].z]
            right_elbow = [landmarks[13].x, landmarks[13].y, landmarks[13].z]
            left_elbow = [landmarks[14].x, landmarks[14].y, landmarks[14].z]
            right_wrist = [landmarks[16].x, landmarks[16].y, landmarks[16].z]
            left_wrist = [landmarks[15].x, landmarks[15].y, landmarks[15].z]
            right_knee = [landmarks[25].x, landmarks[25].y, landmarks[25].z]
            left_knee = [landmarks[26].x, landmarks[26].y, landmarks[26].z]

            left_side_angle = self.calculateAngle(left_shoulder, left_hip, left_knee)
            
            # Calculate the angle between nose, left shoulder, and left hip
            left_angle = self.calculateAngle(left_shoulder, nose, left_hip)
            # print(left_angle)

            # Calculate the angle between nose, right shoulder, and right hip
            right_angle = self.calculateAngle(right_shoulder, nose, right_hip)
            
            left_arm_angle = self.calculateAngle(left_wrist, left_elbow, left_shoulder)
            
            right_arm_angle = self.calculateAngle(right_wrist, right_elbow, right_shoulder)

            # Check if the angles exceed the threshold
            if exercise == "Bicep Curls" or exercise == "Squats":
                angle_threshold = 40
            if exercise == "Shoulder Press" or exercise == "Front Raise":
                if left_arm_angle < 45:
                    angle_threshold = 45
                elif right_arm_angle < 70:
                    angle_threshold = 44
                else:
                    angle_threshold = 40

            if left_angle > angle_threshold or right_angle > angle_threshold:
                return False
            else:
                return True
        else:
            return False

    def calculateAngle(self,a, b, c):
        # Calculate the angle between vectors a->b and b->c using the dot product
        ab = [a[i] - b[i] for i in range(min(len(a), len(b)))]
        bc = [c[i] - b[i] for i in range(min(len(b), len(c)))]
        dot_product = sum([ab[i] * bc[i] for i in range(len(ab))]) # Measures the similarity between vectors
        magnitude_ab = math.sqrt(sum([ab[i] ** 2 for i in range(len(ab))]))
        magnitude_bc = math.sqrt(sum([bc[i] ** 2 for i in range(len(bc))]))
        cosine_angle = dot_product / (magnitude_ab * magnitude_bc)

        # Check if the cosine angle is within the valid range
        if cosine_angle > 1:
            cosine_angle = 1
        elif cosine_angle < -1:
            cosine_angle = -1

        angle = math.degrees(math.acos(cosine_angle))
        return angle

    def checkHeadPosition(self, mp_results):
        landmarks = mp_results.pose_landmarks.landmark

        # Check if relevant landmarks are detected
        if landmarks[self.mediapipe_pose.PoseLandmark.NOSE.value].visibility > self.min_detection_confidence and \
                landmarks[self.mediapipe_pose.PoseLandmark.LEFT_EAR.value].visibility > self.min_detection_confidence and \
                landmarks[self.mediapipe_pose.PoseLandmark.RIGHT_EAR.value].visibility > self.min_detection_confidence:

            # Calculate the angle between the ear points and the nose point
            nose = [landmarks[self.mediapipe_pose.PoseLandmark.NOSE.value].x,
                    landmarks[self.mediapipe_pose.PoseLandmark.NOSE.value].y]
            left_ear = [landmarks[self.mediapipe_pose.PoseLandmark.LEFT_EAR.value].x,
                        landmarks[self.mediapipe_pose.PoseLandmark.LEFT_EAR.value].y]
            right_ear = [landmarks[self.mediapipe_pose.PoseLandmark.RIGHT_EAR.value].x,
                        landmarks[self.mediapipe_pose.PoseLandmark.RIGHT_EAR.value].y]

            # Calculate the angle between the ear points and the nose point
            angle = self.calculateAngle(nose, left_ear, right_ear)

            # print(angle)
            # Check if the head is tilted
            if angle > 29 or angle < 13:
                return False
            else:
                return True
        else:
            return False
        
    def detectPose(self, image, exercise, draw_flag = True):

        # Load the face detection model from OpenCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        # Recoloring Image to RGB
        image_in_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detecting faces in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        # Loop over each detected face
        for (x, y, w, h) in self.faces:

            # Crop the image to the face region
            face_image = image[y:y+h, x:x+w]
            # Creating Detection
            self.mp_results = self.posture.process(image_in_RGB)
            

            if self.mp_results.pose_landmarks:
                # Check head position
                head_position = self.checkHeadPosition(self.mp_results)
                back_posture_status = self.backPosture( image, self.mp_results, exercise)
                # If landmarks are detected and head position is good
                if head_position:
                    if draw_flag:
                        self.mediapipe_draw.draw_landmarks(image, self.mp_results.pose_landmarks,
                                                        self.mediapipe_pose.POSE_CONNECTIONS )
                else:
                    cv2.putText(image, "Head Not In Neutral Position", (120, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if exercise == "Bicep Curls" or exercise == "Shoulder Press" or exercise == "Front Raise" or exercise == "Squats":
                    if back_posture_status:
                        cv2.putText(image, "Good Back Posture!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(image, "Bad Back Posture!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
            else:
                cv2.putText(image, "No Pose Landmarks Detected", (340, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
        return image

    def detectPosition(self, image, draw_flag=True):
        self.landmark_list = []

        # Extracting Landmarks
        try:
            if self.mp_results.pose_landmarks:
                for id, landMark in enumerate(self.mp_results.pose_landmarks.landmark):
                    # height, width, c = image.shape
                    cx, cy, cz = int(landMark.x * width), int(landMark.y * height),(landMark.z * width)
                    self.landmark_list.append([id, cx, cy,cz])
        except:
            pass

        return self.landmark_list
    
    def detectAngle(self, image, lm1, lm2, lm3, draw_flag = True ):
        # Slicing the list and getting the landmarks
        x1, y1, z1 = self.landmark_list[lm1][1:]
        x2, y2, z2 = self.landmark_list[lm2][1:]
        x3, y3, z3 = self.landmark_list[lm3][1:]
    
        # # Display error if multiple body or no face is detected
        # if len(self.faces) > 2:
        #     cv2.rectangle(image, (350, 20), (1000, 200), (0, 0, 0), cv2.FILLED)
        #     cv2.putText(image, f'Multiple Body Detected!', (500, 70), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
        #     cv2.putText(image, f'Please only one person at a time!', (400, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
        if len(self.faces) <= 0:
            cv2.rectangle(image, (200, 20), (1100, 200), (0, 0, 0), cv2.FILLED)
            cv2.putText(image, 'No Face detected!', (500, 70), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
            cv2.putText(image, 'Please position yourself in front of the camera!', (220, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
            return None
                    
        # Check if hip is detected and if not then don't draw line landmarks
        left_hip_visibility = self.mp_results.pose_landmarks.landmark[self.mediapipe_pose.PoseLandmark.LEFT_HIP].visibility
        right_hip_visibility = self.mp_results.pose_landmarks.landmark[self.mediapipe_pose.PoseLandmark.RIGHT_HIP].visibility
                    
        if left_hip_visibility < 0.5 and right_hip_visibility < 0.5: 
            # Displaying error and box           
            cv2.rectangle(image, (400, 400), (900, 520), (0, 0, 0), cv2.FILLED)
            cv2.putText(image, 'Please keep body in view!', (420, 450), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
            cv2.putText(image, 'Move Backward!', (500, 500), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
            
            angle = 0

        else:
            # Calculating the angle of selected landmarks and converting to degree
            angle = math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))
            
            if angle < 0:
                angle += 360
            if angle > 180:
                angle = 360 - angle

            # Drawing the landmarks circle and line
            if draw_flag:
                cv2.line(image, (x1, y1),(x2, y2), (255, 255, 255), 3)
                cv2.line(image, (x3, y3),(x2, y2), (255, 255, 255), 3)
                cv2.circle(image, (x1, y1), 10, (48, 255, 48), cv2.FILLED)
                cv2.circle(image, (x1, y1), 15, (48, 255, 48), 2)
                cv2.circle(image, (x2, y2), 10, (48, 255, 48), cv2.FILLED)
                cv2.circle(image, (x2, y2), 15, (48, 255, 48), 2)
                cv2.circle(image, (x3, y3), 10, (48, 255, 48), cv2.FILLED)
                cv2.circle(image, (x3, y3), 15, (48, 255, 48), 2)
                cv2.putText(image, str(int(angle)), (x2 - 60, y2 + 70),
                            cv2.FONT_HERSHEY_DUPLEX, 2, (48, 255, 48), 2)

        return angle


