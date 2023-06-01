import cv2
import Module as m
import numpy as np
import pygame
import pyttsx3
import threading
import time
import concurrent.futures
from PIL import Image, ImageTk
from PIL.Image import Resampling
import tkinter as tk
import datetime
import math

# Initialize Pygame mixer
pygame.mixer.init()

# Load the sound file
ding_sound = pygame.mixer.Sound("ding.wav")
# Sound credit: 'Ding 3' by Andersmmg
# Source: https://freesound.org/people/andersmmg/sounds/523424/
# License:  https://creativecommons.org/licenses/by/4.0/  Attribution 4.0 International (CC BY 4.0)

    
class Items:

    def __init__(self):
        self.engine = pyttsx3.init()
        self.lock = threading.Lock()
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.bar_color_right = (0, 0, 255)
        self.bar_color_left = (0, 0, 255)

    def sayRepsCount(self, reps_count, side):
        self.engine.say(f"{side} {reps_count}")
        self.engine.runAndWait()
            
    def repetitionCounterRight(self, percentage_work_right, reps_count_right, direction_right):      
        # Checking reps percentage for right arm
        if percentage_work_right == 100 and direction_right == 0:
                reps_count_right += 0.5
                direction_right = 1

        if percentage_work_right == 0 and direction_right == 1:
                reps_count_right += 0.5
                direction_right = 0
                
                # Once the repetitions are complete, play the sound
                ding_sound.play()

                self.thread_pool.submit(self.sayRepsCount, int(reps_count_right), "right")

        return {"reps_count_right": reps_count_right, "direction_right": direction_right}

    def repetitionCounterLeft(self, percentage_work_left, reps_count_left, direction_left):       
        # Checking reps percentage for right arm
        if percentage_work_left == 100 and direction_left == 0:
                reps_count_left += 0.5
                direction_left = 1
        if percentage_work_left == 0 and direction_left == 1:
                reps_count_left += 0.5
                direction_left = 0

                # Once the repetitions are complete, play the sound
                ding_sound.play()

                self.thread_pool.submit(self.sayRepsCount, int(reps_count_left), "left")

        return {"reps_count_left": reps_count_left, "direction_left": direction_left}

    def displayRightCountRep(self, image, reps_count_right):
        # Displaying a rectangle background for repetition count for right arm
        text_size_right, _ = cv2.getTextSize(f'{int(reps_count_right)} reps', cv2.FONT_HERSHEY_DUPLEX, 1.5, 2)
        reps_box_right = np.zeros((text_size_right[1] + 30, text_size_right[0] + 30, 3), np.uint8)
        reps_box_right[:] = (230, 224, 176)
        
        # Displaying the repetition count for right arm
        text_org_right = (int((reps_box_right.shape[1] - text_size_right[0]) / 2), int((reps_box_right.shape[0] + text_size_right[1]) / 2))
        cv2.putText(reps_box_right, f'{int(reps_count_right)} reps', text_org_right, cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)

        image[620:(620 + reps_box_right.shape[0]), 965:(965 + reps_box_right.shape[1])] = reps_box_right

    def displayLeftCountRep(self, image, reps_count_left):
        # Displaying a rectangle background for repetition count for left arm
        text_size_left, _ = cv2.getTextSize(f'{int(reps_count_left)} reps', cv2.FONT_HERSHEY_DUPLEX, 1.5, 2)
        reps_box_left = np.zeros((text_size_left[1] + 30, text_size_left[0] + 30, 3), np.uint8)
        reps_box_left[:] = (230, 224, 176)

        # Displaying the repetition count for right arm
        text_org_left = (int((reps_box_left.shape[1] - text_size_left[0]) / 2), int((reps_box_left.shape[0] + text_size_left[1]) / 2))
        cv2.putText(reps_box_left, f'{int(reps_count_left)} reps', text_org_left, cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)


        image[620:(620 + reps_box_left.shape[0]), 120:(120 + reps_box_left.shape[1])] = reps_box_left

    # Creating a progress bar for right arm        
    def displayPerformanceBarRight(self, image, percentage_work_right, progress_bar_right, bar_color_right):
        cv2.rectangle(image, (1190, int(progress_bar_right)), (1230, 660), bar_color_right, cv2.FILLED)
        cv2.putText(image, f'{int(percentage_work_right)} %', (1160, 90), cv2.FONT_HERSHEY_DUPLEX, 1, bar_color_right, 1)
        cv2.putText(image, f'R', (1197, 690), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

    # Creating a progress bar for left arm
    def displayPerformanceBarLeft(self, image, percentage_work_left, progress_bar_left, bar_color_left):
        cv2.rectangle(image, (50, int(progress_bar_left)), (90, 660), bar_color_left, cv2.FILLED)
        cv2.putText(image, f'{int(percentage_work_left)} %', (30, 90), cv2.FONT_HERSHEY_DUPLEX, 1, bar_color_left, 1)
        cv2.putText(image, f'L', (57, 690), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)

    def displayBarColor(self, percentage_bar):
        color = (0, 205, 205)
        if 0 < percentage_bar <= 30:
            color = (51, 51, 255)
        if 30 < percentage_bar <= 60:
            color = (0, 165, 255)
        if 60 <= percentage_bar <= 100:
            color = (0, 255, 255) 

        return color

    def drawResetButton(self, image, color):
        cv2.rectangle(image, (1280 - 110, 10), (1280 - 20, 50), color, -1)
        cv2.putText(image, "Reset", (1280 - 110, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

class Exercises:
    
    def __init__(self, button_color, main_window): # Initializes various attributes
        self.engine = pyttsx3.init()  # Text to speech
        self.reps_count_left = 0 # Repetition Counters
        self.direction_left = 0
        self.reps_count_right = 0
        self.direction_right = 0
        self.cap = None
        self.pose_detector = m.poseDetection() # Initialize the pose detector
        self.initCamera() # Camera initialization
        self.button_color = button_color # Button color
        self.initial_left_elbow_hip_distance = None
        self.initial_right_elbow_hip_distance = None
        self.main_window = main_window
        self.previous_percentage_work_right = 0 
        self.last_percentage_work_time_right = 0 
        self.previous_percentage_work_left = 0 
        self.last_percentage_work_time_left = 0  
        self.initial_left_elbow = None
        
    def initCamera(self):
        # Initialize the camera and pose detector
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.pose_detector = m.poseDetection()

    def openCamera(self):     
        # Restart the camera and pose detector
        self.cap.release()
        self.initCamera()

    def sayExercise(self, exercise):
        self.engine.say(f"Exercise {exercise}, Please Position Yourself")
        self.engine.runAndWait()

    def resetCounters(self):
        self.reps_count_left = 0
        self.direction_left = 0
        self.reps_count_right = 0
        self.direction_right = 0
    
    def setInitialElbowPosition(self, landmark_list): # Sets the initial position of both elbows
        if len(landmark_list) != 0:
            self.initial_left_elbow = (
                landmark_list[14][1],
                landmark_list[14][2]
            )
            self.initial_right_elbow = (
                landmark_list[13][1],
                landmark_list[13][2]
            )

    def resetElbowHipDistance(self): # Resetting the initial distance between the elbows and hips
        self.initial_left_elbow_hip_distance = None
        self.initial_right_elbow_hip_distance = None

    def showGif(self, gif_path, duration):
        top = tk.Toplevel(self.main_window)
        top.geometry("550x920")
        top.configure(bg="#141414")

        label = tk.Label(top, bg="#141414")
        label.pack()

        self.animateGif(gif_path, label)

        # Create the text label
        text_label = tk.Label(top, text="Exercise Tutorial", bg="#141414", fg="white", font=("Helvetica", 18, 'bold'))
        text_label.pack()
        self.main_window.after(duration, top.destroy)

    def animateGif(self, gif_path, label): # Updates the GIF frames and display them in the window
        resize_factor= 1.5 # Resizing factor
        img = Image.open(gif_path) # Opening the GIF file
        frames = []  # Initializing empty list for frames
        # Resizes the opened image and convert them to a PhotoImage object
        frame = ImageTk.PhotoImage(img.resize((int(img.width * resize_factor), int(img.height * resize_factor)), Image.ANTIALIAS))
        frames.append(frame) # Adding the new frame to the list
        
        try:
            while True:
                img.seek(img.tell() + 1) # Advances to the next frame and returns the index of the current frame then moved to the next frame
                frame = ImageTk.PhotoImage(img.resize((int(img.width * resize_factor), int(img.height * resize_factor)), Image.ANTIALIAS))
                frames.append(frame)
        except EOFError: # If no more frames to read from the GIF
            pass

        def update(index): # Updating and displaying the frames of the GIF.
            if index == len(frames): # If all frames have been displayed
                index = 0  # Reset to 0
            photo = frames[index] # Retrieves the PhotoImage object from the frames list and assigns to photo variable.
            label.config(image=photo)  # Updates the image property of the label widget to display current frame.
            label.image = photo # Assigns the photo object to the image attribute of the label widget to ensure it's not garbage collected.
            label.after(20, update, index + 1) # Schedules the next update of the frame after 20 milliseconds.

        update(0) # Initiates the first call to the update function with an index of 0.

    def showGifAndRunExercise(self, gif_path, duration, exercise_function, exercise_name):
        self.main_window.withdraw()
        self.showGif(gif_path, duration)

        # Create a separate thread to run the text-to-speech engine
        tts_thread = threading.Thread(target=self.sayExercise, args=(exercise_name,))
        tts_thread.start()
        
        self.main_window.after(duration, exercise_function)

    def showSummary(self, start_time, exercise_name):
        # Function to close the window
        def close_summary_window():
            summary_window.destroy()
            # Show user interface again
            self.main_window.deiconify()

        # Create a new window
        summary_window = tk.Toplevel(self.main_window)
        summary_window.geometry("300x400")
        summary_window.title("Exercise Summary")
        summary_window.configure(bg="#141414")

        # Bind the close button to close_summary_window() function
        summary_window.protocol("WM_DELETE_WINDOW", close_summary_window)

        # Schedule the window to close after 10 seconds
        summary_window.after(10000, close_summary_window)

        # Create the title label
        title_label = tk.Label(summary_window, text="S U M M A R Y", 
            font=("Baskerville", 24), height=2, bg="#808080", fg="#141414")
        title_label.pack(fill=tk.X)

        # Get the current time and calculate the time difference
        current_time = datetime.datetime.now()
        start_datetime = datetime.datetime.fromtimestamp(start_time)
        time_diff = current_time - start_datetime

        exercise_label = tk.Label(summary_window, text=f"Exercise: {exercise_name}", font=("Baskerville", 14),
            fg="white", bg="#141414", height=2)
        exercise_label.pack()

        # Display the repetition counts and the current time
        left_label = tk.Label(summary_window, text=f"Left Reps: {int(self.reps_count_left)}", font=("Baskerville", 14),
            fg="white", bg="#141414", height=2)
        left_label.pack()
        right_label = tk.Label(summary_window, text=f"Right Reps: {int(self.reps_count_right)}", font=("Baskerville", 14), 
            fg="white", bg="#141414", height=2)
        right_label.pack()
        time_label = tk.Label(summary_window, text=f"Time: {time_diff}", font=("Baskerville", 14), 
            fg="white", bg="#141414", height=2)
        time_label.pack()

        summary_window.mainloop()

    # Define the function for the Bicep Curl workout
    def bicepCurl(self):

        # Define a function to handle mouse clicks
        def mouseCallback(event, x, y, flags, param):
            nonlocal self
            button_start_x = 1280 - 110 
            button_end_x = 1280 - 30
            button_start_y = 10
            button_end_y = 50
            if button_start_x <= x <= button_end_x and button_start_y <= y <= button_end_y:  # Check if the click is within the reset button region
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.resetCounters()
                    print("Reset clicked")
                    # Reset the initial left elbow to left hip distance value
                    self.resetElbowHipDistance()

                    # Change color when the button is clicked
                    self.button_color = (130, 129, 129) 
                elif event == cv2.EVENT_LBUTTONUP:
                    self.button_color = (176, 175, 175) 
        
        print("Performing Bicep Curl...")

        # Initializes the camera
        self.openCamera()

        # Reset counters to 0
        self.resetCounters()

        # Reset the initial left elbow to left hip distance value
        self.resetElbowHipDistance()

        # Define variables for counting repetitions
        left_in_repetition = False
        right_in_repetition = False

        left_elbow_y_initial = None
        right_elbow_y_initial = None

        # Set the start time of the exercise
        self.start_time = time.time()

        last_too_fast_left = None
        last_too_fast_right = None

        max_speed = 450

        # Loop through the frames
        while True:
            # Calculate elapsed time
            elapsed_time = time.time() - self.start_time

            success, image = self.cap.read()
            image = cv2.flip(image, 1)
            image = self.pose_detector.detectPose(image,"Bicep Curls", False)
            landmark_list = self.pose_detector.detectPosition(image, False)

            # Draw the elapsed time on the image
            cv2.putText(image, f"Time: {elapsed_time:.2f}s", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            self.setInitialElbowPosition(landmark_list)
            left_elbow_movement_threshold = 26
            right_elbow_movement_threshold = 26

            if len(landmark_list) != 0:

                # Left Arm Angle
                left_angle = self.pose_detector.detectAngle(image, 12, 14, 16)
                # Right Arm Angle
                right_angle = self.pose_detector.detectAngle(image, 11, 13, 15)

                # Left Arm
                if left_angle == None or right_angle == None:
                    cv2.rectangle(image, (200, 20), (1100, 200), (0, 0, 0), cv2.FILLED)
                    cv2.putText(image, f'No Face detected!', (500, 70), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
                    cv2.putText(image, f'Please position yourself in front of the camera!', (220, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
                else:
                    percentage_work_left = np.interp(left_angle,(30,90),(100,0))
                    progress_bar_left = np.interp(left_angle, (30, 90), (100, 650))

                    left_elbow_y_current = landmark_list[14][2] if landmark_list and landmark_list[14] else None
                    if left_elbow_y_initial is None:
                        left_elbow_y_initial = left_elbow_y_current
                        
                    # Change percentage bar color
                    left_color = Items().displayBarColor(percentage_work_left)

                    if left_elbow_y_initial is not None and abs(left_elbow_y_current - left_elbow_y_initial) > left_elbow_movement_threshold:
                            cv2.putText(image, "Elbow Too High!", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Checking reps percentage for left arm
                    if percentage_work_left == 100 or percentage_work_left == 0:
                        left_color = (0, 255, 0)
                        epsilon = 1e-8
                        left_speed = abs((percentage_work_left - self.previous_percentage_work_left) / (time.time() - self.last_percentage_work_time_left + epsilon))
                        # print(left_speed)
                        self.previous_percentage_work_left = percentage_work_left
                        self.last_percentage_work_time_left = time.time()
                        
                        # print(abs(left_elbow_y_current - left_elbow_y_initial))
                        if left_elbow_y_initial is not None and abs(left_elbow_y_current - left_elbow_y_initial) > left_elbow_movement_threshold:
                            cv2.putText(image, "Don't Swing Your Arm!", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            left_in_repetition = False
                        else:
                            if not left_in_repetition:
                                left_in_repetition = True
                                self.lastRepTimeLeft = time.time()
                            elif left_speed > max_speed:  
                                last_too_fast_left = datetime.datetime.now()
                                too_fast = True
                            elif time.time() - self.lastRepTimeLeft > 0.4:
                                repetition_left = Items().repetitionCounterLeft(percentage_work_left, self.reps_count_left, self.direction_left)
                                self.reps_count_left = repetition_left["reps_count_left"]
                                self.direction_left = repetition_left["direction_left"]
                                left_in_repetition = False                 

                        if last_too_fast_left is not None and (datetime.datetime.now() - last_too_fast_left).total_seconds() < 1.5:
                            cv2.putText(image, "Too Fast! Slow Down!", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
 
                    # Creating a progress bar for left arm
                    Items().displayPerformanceBarLeft(image, percentage_work_left, progress_bar_left, left_color)

                    # Displaying the repetition count for left arm
                    Items().displayLeftCountRep(image, self.reps_count_left)

                    # Right Arm
                    percentage_work_right = np.interp(right_angle,(30,90),(100,0))
                    progress_bar_right = np.interp(right_angle, (30, 90), (100, 650))
                    
                    right_elbow_y_current = landmark_list[13][2] if landmark_list and landmark_list[13] else None
                    if right_elbow_y_initial is None:
                        right_elbow_y_initial = right_elbow_y_current
                    
                    right_color = Items().displayBarColor(percentage_work_right)
                    
                    if right_elbow_y_initial is not None and abs(right_elbow_y_current - right_elbow_y_initial) > right_elbow_movement_threshold:
                                cv2.putText(image, "Elbow Too High!", (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Checking reps percentage for right arm
                    if percentage_work_right == 100 or percentage_work_right == 0:
                            right_color = (0, 255, 0)
                            epsilon = 1e-8
                            right_speed = abs((percentage_work_right - self.previous_percentage_work_right) / (time.time() - self.last_percentage_work_time_right + epsilon))
                            
                            self.previous_percentage_work_right = percentage_work_right
                            self.last_percentage_work_time_right = time.time()
                            # print(abs(right_elbow_y_current - right_elbow_y_initial))
                            if right_elbow_y_initial is not None and abs(right_elbow_y_current - right_elbow_y_initial) > right_elbow_movement_threshold:
                                cv2.putText(image, "Don't Swing Your Arm!", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                right_in_repetition = False
                            else:
                                if not right_in_repetition:
                                    right_in_repetition = True
                                    self.lastRepTimeRight = time.time()
                                elif right_speed > max_speed:  
                                    last_too_fast_right = datetime.datetime.now()
                                elif time.time() - self.lastRepTimeRight > 0.4:
                                    repetition_right = Items().repetitionCounterRight(percentage_work_right, self.reps_count_right, self.direction_right)
                                    self.reps_count_right = repetition_right["reps_count_right"]
                                    self.direction_right = repetition_right["direction_right"]
                                    right_in_repetition = False   

                            if last_too_fast_right is not None and (datetime.datetime.now() - last_too_fast_right).total_seconds() < 1.5:
                                cv2.putText(image, "Too Fast! Slow Down!", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Creating a progress bar for right arm
                    Items().displayPerformanceBarRight(image, percentage_work_right, progress_bar_right, right_color)
                    
                    # Displaying the repetition count for right arm
                    Items().displayRightCountRep(image, self.reps_count_right)

            # Draw the reset button
            Items().drawResetButton(image, self.button_color)
            
            # cv2.namedWindow("Bicep Curl")    
            cv2.imshow("Bicep Curl", image)

            cv2.setMouseCallback("Bicep Curl", mouseCallback)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Closes the window
        self.close()

        # Show the summary window
        self.showSummary(start_time=self.start_time, exercise_name="Bicep Curl")


    # Define the function for the Shoulder Press workout
    def shoulderPress(self):
        # Define a function to handle mouse clicks
        def mouseCallback(event, x, y, flags, param):
            nonlocal self
            button_start_x = 1280 - 110 
            button_end_x = 1280 - 30
            button_start_y = 10
            button_end_y = 50
            if button_start_x <= x <= button_end_x and button_start_y <= y <= button_end_y:  # Check if the click is within the reset button region
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.resetCounters()
                    print("Reset clicked")
                    # Change color when the button is clicked
                    self.button_color = (130, 129, 129) 
                elif event == cv2.EVENT_LBUTTONUP:
                    self.button_color = (176, 175, 175) 

        print("Performing Shoulder Press...")
        
        # Initializes the camera
        self.openCamera()

        # Reset counters to 0
        self.resetCounters()
        time_threshold = 0.8

        # Set the start time of the exercise
        self.start_time = time.time()

        last_too_fast_left = None
        last_too_fast_right = None

        max_speed = 160

        # Loop through the frames
        while True:
            # Calculate elapsed time
            elapsed_time = time.time() - self.start_time

            success, image = self.cap.read()
            image = cv2.flip(image, 1)
            image = self.pose_detector.detectPose(image,"Shoulder Press", False)
            landmark_list = self.pose_detector.detectPosition(image, False)
                
            # Draw the elapsed time on the image
            cv2.putText(image, f"Time: {elapsed_time:.2f}s", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if len(landmark_list) != 0:

                # Draw lines and calculate angles for left arm (landmarks 16, 14, 12)
                left_arm_angle = self.pose_detector.detectAngle(image, 16, 14, 12)

                # Draw lines and calculate angles for right arm (landmarks 11, 13, 15)
                right_arm_angle = self.pose_detector.detectAngle(image, 11, 13, 15)

                # Left Arm Angle
                left_angle = self.pose_detector.detectAngle(image, 24, 12, 14)
                # Right Arm Angle
                right_angle = self.pose_detector.detectAngle(image, 23, 11, 13)
                
                # Left Arm
                if left_angle == None or right_angle == None or left_arm_angle == None or right_arm_angle == None:
                    cv2.rectangle(image, (200, 20), (1100, 200), (0, 0, 0), cv2.FILLED)
                    cv2.putText(image, f'No Face detected!', (500, 70), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
                    cv2.putText(image, f'Please position yourself in front of the camera!', (220, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
                else:
                    percentage_work_left = np.interp(left_angle, (85, 160), (0, 100))
                    progress_bar_left = np.interp(left_angle, (85, 160), (650, 100))
                    
                    # Change percentage bar color
                    left_color = Items().displayBarColor(percentage_work_left)

                    # Initialize a variable for forearm alignment
                    left_forearm_aligned = True

                    if left_angle > 110 and left_angle < 140:
                        if left_arm_angle < 70 or left_arm_angle > 140:         
                            left_color = (0, 255, 0)
                            cv2.putText(image, "Fix Left Forearm Vertically!", (150, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            left_forearm_aligned = False

                    if left_angle > 110 and left_arm_angle < 70:
                        left_color = (0, 255, 0)
                        cv2.putText(image, "Fix Left Forearm Vertically!", (150, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        right_forearm_aligned = False

                    # Initialize last rep time variables
                    self.last_rep_time_left = None
                    self.last_rep_time_right = None

                    # Checking reps percentage for left arm
                    if left_forearm_aligned and (percentage_work_left == 100 or percentage_work_left == 0):
                        epsilon = 1e-8
                        left_speed = abs((percentage_work_left - self.previous_percentage_work_left) / (time.time() - self.last_percentage_work_time_left + epsilon))
                        
                        self.previous_percentage_work_left = percentage_work_left
                        self.last_percentage_work_time_left = time.time()

                        if left_speed > max_speed:  
                                last_too_fast_left = datetime.datetime.now()
                        if not left_angle < 110 and not left_arm_angle > 70:
                            pass
                        else:
                            # Change percentage bar color
                            left_color = (0, 255, 0)
                            repetition_left = Items().repetitionCounterLeft(percentage_work_left, self.reps_count_left, self.direction_left)
                            self.reps_count_left = repetition_left["reps_count_left"]
                            self.direction_left = repetition_left["direction_left"]

                        if last_too_fast_left is not None and (datetime.datetime.now() - last_too_fast_left).total_seconds() < 1.5:
                                cv2.putText(image, "Too Fast! Slow Down!", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


                    # Creating a progress bar for left arm
                    Items().displayPerformanceBarLeft(image, percentage_work_left, progress_bar_left, left_color)

                    # Displaying the repetition count for left arm
                    Items().displayLeftCountRep(image, self.reps_count_left)

                    # Right Arm
                    percentage_work_right = np.interp(right_angle,(85,160),(0,100))
                    progress_bar_right = np.interp(right_angle, (85, 160), (650, 100))
                        
                    right_color = Items().displayBarColor(percentage_work_right)

                    # Initialize a variable for forearm alignment
                    right_forearm_aligned = True

                    if right_angle > 110 and right_angle < 140:
                        if right_arm_angle < 70 or right_arm_angle > 140:
                            right_color = (0, 255, 0)
                            cv2.putText(image, "Fix Right Forearm Vertically!", (700, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            right_forearm_aligned = False

                    if right_angle > 110 and right_arm_angle < 70:
                        right_color = (0, 255, 0)
                        cv2.putText(image, "Fix Right Forearm Vertically!", (700, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        right_forearm_aligned = False

                    # Checking reps percentage for right arm
                    if right_forearm_aligned and (percentage_work_right == 100 or percentage_work_right == 0):
                        epsilon = 1e-8
                        right_speed = abs((percentage_work_right - self.previous_percentage_work_right) / (time.time() - self.last_percentage_work_time_right + epsilon))
                        # print(right_speed)

                        self.previous_percentage_work_right = percentage_work_right
                        self.last_percentage_work_time_right = time.time()

                        if right_speed > max_speed:  
                                last_too_fast_right = datetime.datetime.now()
                        if not right_angle < 110 and not right_arm_angle > 70:
                            pass
                        else:
                            right_color = (0, 255, 0)
                            repetition_right = Items().repetitionCounterRight(percentage_work_right, self.reps_count_right, self.direction_right)
                            self.reps_count_right = repetition_right["reps_count_right"]
                            self.direction_right = repetition_right["direction_right"]

                        if last_too_fast_right is not None and (datetime.datetime.now() - last_too_fast_right).total_seconds() < 1.5:
                                cv2.putText(image, "Too Fast! Slow Down!", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Creating a progress bar for right arm
                    Items().displayPerformanceBarRight(image, percentage_work_right, progress_bar_right, right_color)
                    
                    # Displaying the repetition count for right arm
                    Items().displayRightCountRep(image, self.reps_count_right)

                    if left_angle > 180:
                        cv2.putText(image, "Don't Lock Left Elbow!", (150, 260), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    if left_angle < 75:
                        cv2.putText(image, "Left Arm Too Low!", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    if right_angle > 180:
                        cv2.putText(image, "Don't Lock Right Elbow!", (800, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    if right_angle < 75:
                        cv2.putText(image, "Right Arm Too Low!", (800, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    if left_angle < 100:
                        if left_arm_angle > 94:
                            cv2.putText(image, "Move Left Forearm to the Right!", (150, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        if left_arm_angle < 60:
                            cv2.putText(image, "Move Left Forearm to the Left!", (150, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    if right_angle < 100:
                        if right_arm_angle > 90:
                            cv2.putText(image, "Move Right Forearm to the Left!", (700, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        if right_arm_angle < 60:
                            cv2.putText(image, "Move Right Forearm to the Right!", (700, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Draw the reset button
            Items().drawResetButton(image, self.button_color)

            cv2.imshow("Shoulder Press", image)

            cv2.setMouseCallback("Shoulder Press", mouseCallback)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        # Closes the window
        self.close()

        # Show the summary window
        self.showSummary(start_time=self.start_time, exercise_name="Shoulder Press")

    # Define the function for the Squats workout
    def squats(self):
        # Define a function to handle mouse clicks
        def mouseCallback(event, x, y, flags, param):
            nonlocal self
            button_start_x = 1280 - 110 
            button_end_x = 1280 - 30
            button_start_y = 10
            button_end_y = 50
            if button_start_x <= x <= button_end_x and button_start_y <= y <= button_end_y:  # Check if the click is within the reset button region
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.resetCounters()
                    print("Reset clicked")
                    # Change color when the button is clicked
                    self.button_color = (130, 129, 129) 
                elif event == cv2.EVENT_LBUTTONUP:
                    self.button_color = (176, 175, 175) 

        print("Performing Squats...")

        # Initializes the camera
        self.openCamera()

        # Reset counter to 0
        self.resetCounters()

        # Set the start time of the exercise
        self.start_time = time.time()

        last_too_fast_left = None
        last_too_fast_right = None

        max_speed = 210

        # Loop through the frames
        while True:
            # Calculate elapsed time
            elapsed_time = time.time() - self.start_time

            success, image = self.cap.read()
            image = cv2.flip(image, 1)
            image = self.pose_detector.detectPose(image,"Squats", False)
            landmark_list = self.pose_detector.detectPosition(image, False)

            # Draw the elapsed time on the image
            cv2.putText(image, f"Time: {elapsed_time:.2f}s", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if len(landmark_list) != 0:
                
                # Calculate left leg squat angle
                left_squat_angle = self.pose_detector.detectAngle(image, 26, 24, 12)

                # Calculate right squat angle
                right_squat_angle = self.pose_detector.detectAngle(image, 25, 23, 11)

                # Right Leg
                if right_squat_angle == None or left_squat_angle == None:
                    cv2.rectangle(image, (200, 20), (1100, 200), (0, 0, 0), cv2.FILLED)
                    cv2.putText(image, f'No Face detected!', (500, 70), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
                    cv2.putText(image, f'Please position yourself in front of the camera!', (220, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
                else:
                    if left_squat_angle < 50 or right_squat_angle < 50:
                        cv2.putText(image, "Squat is too Low!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    percentage_work_right = np.interp(right_squat_angle, (75, 165), (100, 0))
                    progress_bar_right = np.interp(right_squat_angle, (75, 165), (100, 650))

                    # Change percentage bar color
                    bar_color_right = Items().displayBarColor(percentage_work_right)

                    if left_squat_angle > 120 and right_squat_angle < 100:
                        cv2.putText(image, "Straigthen your body!", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        # Checking reps percentage for squat
                        if percentage_work_right == 100 or percentage_work_right == 0:
                            bar_color_right = (0, 255, 0)
                            epsilon = 1e-8
                            right_speed = abs((percentage_work_right - self.previous_percentage_work_right) / (time.time() - self.last_percentage_work_time_right + epsilon))
                            # print(right_speed)
                            self.previous_percentage_work_right = percentage_work_right
                            self.last_percentage_work_time_right = time.time()

                            if right_speed > max_speed:  
                                    last_too_fast_right = datetime.datetime.now()
                            else:
                                repetition_right = Items().repetitionCounterRight(percentage_work_right, self.reps_count_right, self.direction_right)
                                self.reps_count_right = repetition_right["reps_count_right"]
                                self.direction_right = repetition_right["direction_right"]
                            
                            if last_too_fast_right is not None and (datetime.datetime.now() - last_too_fast_right).total_seconds() < 1.5:
                                cv2.putText(image, "Too Fast! Slow Down!", (120, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Creating a progress bar for squat
                    Items().displayPerformanceBarRight(image, percentage_work_right, progress_bar_right, bar_color_right)

                    # Displaying the right repetition count for squat
                    Items().displayRightCountRep(image, self.reps_count_right)

                    # Left Leg
                    percentage_work_left = np.interp(left_squat_angle, (90, 155), (100, 0))
                    progress_bar_left = np.interp(left_squat_angle, (90, 155), (100, 650))
                    
                    # Change percentage bar color
                    bar_color_left = Items().displayBarColor(percentage_work_left)

                    # Checking reps percentage for squat
                    if right_squat_angle > 120 and left_squat_angle < 100:
                        cv2.putText(image, "Straigthen your body!", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        if percentage_work_left == 100 or percentage_work_left == 0:
                            bar_color_left = (0, 255, 0)
                            repetition_left = Items().repetitionCounterLeft(percentage_work_left, self.reps_count_left, self.direction_left)
                            self.reps_count_left = repetition_left["reps_count_left"]
                            self.direction_left = repetition_left["direction_left"]

                    # Creating a progress bar for squat
                    Items().displayPerformanceBarLeft(image, percentage_work_left, progress_bar_left, bar_color_left)

                    # Displaying the left repetition count for squat
                    Items().displayLeftCountRep(image, self.reps_count_left)

                # Checking feet angle distance
                feet_landmarks = [27, 28]
                if all(landmark_list[i][3] < 700 for i in feet_landmarks):
                    feet_angle = self.pose_detector.detectAngle(image, 28, 0, 27, draw_flag=False)
                    if feet_angle == None:
                        cv2.rectangle(image, (200, 20), (1100, 200), (0, 0, 0), cv2.FILLED)
                        cv2.putText(image, f'No Face detected!', (500, 70), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
                        cv2.putText(image, f'Please position yourself in front of the camera!', (220, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
                    else:
                        if (left_squat_angle < 140 and feet_angle > 30) and feet_angle > 17 or feet_angle < 12 or (left_squat_angle > 150 and feet_angle > 17):
                            cv2.putText(image, "Adjust feet to shoulder-width apart!", (380, 570), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                else:
                    cv2.rectangle(image, (390, 650), (900, 770), (0, 0, 0), cv2.FILLED)
                    cv2.putText(image, f'Please show feet distance!', (410, 700), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)

                # knee_angle = self.pose_detector.detectAngle(image, 26, 0, 25)
                       
                # if knee_angle < 15:
                #     cv2.putText(image, "Don't Bend Knee Inwards!", (400, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # print(feet_angle)
                # for i in feet_landmarks:
                #     print(f"Landmark {i}: ({landmark_list[i][1]}, {landmark_list[i][2]}), {landmark_list[i][3]})")
                
            # Draw the reset button
            Items().drawResetButton(image, self.button_color)

            cv2.imshow("Squat", image)

            cv2.setMouseCallback("Squat", mouseCallback)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Closes the window
        self.close()
        
        # Show the summary window
        self.showSummary(start_time=self.start_time, exercise_name="Squat")

    # Define the function for the Front Raise workout
    def frontRaise(self):
        # Define a function to handle mouse clicks
        def mouseCallback(event, x, y, flags, param):
            nonlocal self
            button_start_x = 1280 - 110 
            button_end_x = 1280 - 30
            button_start_y = 10
            button_end_y = 50
            if button_start_x <= x <= button_end_x and button_start_y <= y <= button_end_y:  # Check if the click is within the reset button region
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.resetCounters()
                    print("Reset clicked")
                    # Change color when the button is clicked
                    self.button_color = (130, 129, 129) 
                elif event == cv2.EVENT_LBUTTONUP:
                    self.button_color = (176, 175, 175) 
    
        print("Performing Front Raise...")

        # Initializes the camera
        self.openCamera()

        # Reset counters to 0
        self.resetCounters()

        # Set the start time of the exercise
        self.start_time = time.time()

        prev_left_wrist_x, prev_left_wrist_y = None, None

        last_too_fast_left = None
        last_too_fast_right = None

        max_speed = 160 

        # Loop through the frames
        while True:
            # Calculate elapsed time
            elapsed_time = time.time() - self.start_time

            success, image = self.cap.read()
            image = cv2.flip(image, 1)
            image = self.pose_detector.detectPose(image,"Front Raise", False)
            landmark_list = self.pose_detector.detectPosition(image, False)
            
            # Draw the elapsed time on the image
            cv2.putText(image, f"Time: {elapsed_time:.2f}s", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if len(landmark_list) != 0:
                # Threshold for Lateral Raise
                lateral_raise_threshold = 160
                
                # Left Arm
                left_angle = self.pose_detector.detectAngle(image, 24, 12, 14)
                # Check if left arm going the right direction
                left_arm_going_right_angle = self.pose_detector.detectAngle(image, 11, 12, 16, draw_flag=False)
                # Checking the angle of the left arm
                left_arm = self.pose_detector.detectAngle(image, 16, 8, 7, draw_flag=False)

                # Right Arm
                right_angle = self.pose_detector.detectAngle(image, 23, 11, 13)
                # Check if right arm going the left direction
                right_arm_going_left_angle = self.pose_detector.detectAngle(image, 12, 11, 15, draw_flag=False)
                # Checking the angle of the right arm
                right_arm = self.pose_detector.detectAngle(image, 15, 7, 8, draw_flag=False)

                left_forearm_angle = self.pose_detector.detectAngle(image, 12, 14, 16, draw_flag=False)
                right_forearm_angle = self.pose_detector.detectAngle(image, 11, 13, 15, draw_flag=False)

                if left_forearm_angle == None or right_forearm_angle == None or right_angle == None or left_angle == None or left_arm_going_right_angle == None or \
                    left_arm == None or right_arm_going_left_angle == None or right_arm == None:
                    cv2.rectangle(image, (200, 20), (1100, 200), (0, 0, 0), cv2.FILLED)
                    cv2.putText(image, f'No Face detected!', (500, 70), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
                    cv2.putText(image, f'Please position yourself in front of the camera!', (220, 150), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1)
                else:

                    # Left Arm
                    percentage_work_left = np.interp(left_angle, (30, 105), (0, 100))
                    progress_bar_left = np.interp(left_angle, (30, 105), (650, 100))

                    left_hand_x = landmark_list[16][1]
                    left_shoulder_x = landmark_list[12][1]
                    
                    # Check if it's a lateral raise 
                    is_lateral_raise_left = abs(left_hand_x - left_shoulder_x) > lateral_raise_threshold

                    # Change percentage bar color
                    left_color = Items().displayBarColor(percentage_work_left)
                    
                    # If left arm going wrong direction
                    if left_arm_going_right_angle < 80 or (left_angle > 79 and left_arm < 145): 
                        cv2.putText(image, "Move Left Hand Left!", (100, 330), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    if is_lateral_raise_left or left_angle > 125:
                        left_color = (0, 255, 0)
                        cv2.putText(image, "Please raise your arm in front!", (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        if left_forearm_angle < 100:
                            cv2.putText(image, "Straigthen your arm front!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        else:
                            # Change percentage bar color
                            left_color = Items().displayBarColor(percentage_work_left)
                            # Checking reps percentage for left arm
                            if percentage_work_left == 100 or percentage_work_left == 0:
                                left_color = (0, 255, 0)
                                epsilon = 1e-8
                                left_speed = abs((percentage_work_left - self.previous_percentage_work_left) / (time.time() - self.last_percentage_work_time_left + epsilon))
                                # print(left_speed)
                                self.previous_percentage_work_left = percentage_work_left
                                self.last_percentage_work_time_left = time.time()

                                if left_speed > max_speed:  
                                        last_too_fast_left = datetime.datetime.now()
                                else:
                                    repetition_left = Items().repetitionCounterLeft(percentage_work_left, self.reps_count_left, self.direction_left)
                                    self.reps_count_left = repetition_left["reps_count_left"]
                                    self.direction_left = repetition_left["direction_left"]

                                if last_too_fast_left is not None and (datetime.datetime.now() - last_too_fast_left).total_seconds() < 1.5:
                                    cv2.putText(image, "Too Fast! Slow Down!", (120, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Creating a progress bar for left arm
                    Items().displayPerformanceBarLeft(image, percentage_work_left, progress_bar_left, left_color)

                    # Displaying the repetition count for left arm
                    Items().displayLeftCountRep(image, self.reps_count_left)


                    # Right Arm
                    percentage_work_right = np.interp(right_angle, (25, 90), (0, 100))
                    progress_bar_right = np.interp(right_angle, (25, 90), (650, 100))

                    right_hand_x = landmark_list[15][1]
                    right_shoulder_x = landmark_list[11][1]
                    
                    # Check if it's a lateral raise
                    is_lateral_raise_right = abs(right_hand_x - right_shoulder_x) > lateral_raise_threshold
                    
                    # Change percentage bar color
                    right_color = Items().displayBarColor(percentage_work_right)

                    # Check right arm is going to the wrong direction
                    if right_arm_going_left_angle < 80 or (right_angle > 65 and right_arm < 145): 
                        cv2.putText(image, "Move Right Hand Right!", (800, 330), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)      

                    if is_lateral_raise_right or right_angle > 125:
                        right_color = (0, 255, 0)
                        cv2.putText(image, "Please raise your arm in front!", (90, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        if right_forearm_angle < 100:
                            cv2.putText(image, "Straigthen your arm front!", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        else:
                            # Change percentage bar color
                            right_color = Items().displayBarColor(percentage_work_right)
                            
                            # Checking reps percentage for right arm
                            if percentage_work_right == 100 or percentage_work_right == 0:
                                right_color = (0, 255, 0)
                                epsilon = 1e-8
                                right_speed = abs((percentage_work_right - self.previous_percentage_work_right) / (time.time() - self.last_percentage_work_time_right + epsilon))
                                
                                self.previous_percentage_work_right = percentage_work_right
                                self.last_percentage_work_time_right = time.time()

                                if right_speed > max_speed:  
                                    last_too_fast_right = datetime.datetime.now()
                                else:        
                                    repetition_right = Items().repetitionCounterRight(percentage_work_right, self.reps_count_right, self.direction_right)
                                    self.reps_count_right = repetition_right["reps_count_right"]
                                    self.direction_right = repetition_right["direction_right"] 

                                if last_too_fast_right is not None and (datetime.datetime.now() - last_too_fast_right).total_seconds() < 1.5:
                                    cv2.putText(image, "Too Fast! Slow Down!", (120, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    # Creating a progress bar for right arm
                    Items().displayPerformanceBarRight(image, percentage_work_right, progress_bar_right, right_color)

                    # Displaying the repetition count for right arm
                    Items().displayRightCountRep(image, self.reps_count_right)

                    if left_angle > 130:
                        cv2.putText(image, "Left Arm Too High!", (150, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    if right_angle > 130:
                        cv2.putText(image, "Right Arm Too High!", (800, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Draw the reset button
            Items().drawResetButton(image, self.button_color)

            cv2.imshow("Front Raise", image)
            
            cv2.setMouseCallback("Front Raise", mouseCallback)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Closes the window
        self.close()

        # Show the summary window
        self.showSummary(start_time=self.start_time, exercise_name="Front Raise")

    def close(self):
        # Release the camera
        self.cap.release()

        # Close all windows
        cv2.destroyAllWindows()

