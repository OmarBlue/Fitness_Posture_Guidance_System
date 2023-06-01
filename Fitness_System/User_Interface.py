import tkinter as tk
from Fitness_System import Exercises

class FitnessPostureSystem:
    def __init__(self):
        self.main = tk.Tk()
        self.main.geometry("400x500")
        self.main.configure(bg="#141414")
        self.main.title("Fitness Posture Assistance System")
        self.title = tk.Label(self.main, text="W O R K O U T   L I S T", 
        font=("Baskerville", 24), height=2, bg="#0B69B0", fg="#141414")
        self.title.pack(fill=tk.X)
        self.exercises = Exercises(button_color=(176, 175, 175), main_window=self.main)
        self.createWorkoutButtons()
        self.main.mainloop()

    def createWorkoutButtons(self):
        def onHover(button, bg, fg):
            button['background'] = bg
            button['foreground'] = fg
            
        def onMouseLeave(button, bg, fg):
            button['background'] = fg
            button['foreground'] = bg

        def workoutButtons(text, command, bg, fg):
            button = tk.Button(self.main, width=45, height=3, font=("Baskerville", 12), 
                                    text=text, command=command,
                                    fg=fg, bg=bg,
                                    border=0, activeforeground=fg,
                                    activebackground=bg)
            button.bind('<Leave>', lambda event: onMouseLeave(button, bg, fg))
            button.bind('<Enter>', lambda event: onHover(button, bg, fg))
            button.pack(fill=tk.X)

        workoutButtons("B I C E P   C U R L", lambda: self.exercises.showGifAndRunExercise("GIF_Images/bicep_curl.gif",
            6000, self.exercises.bicepCurl, "Bicep Curl"), "#9F73AB", "#141414")
        workoutButtons("S H O U L D E R   P R E S S", lambda: self.exercises.showGifAndRunExercise("GIF_Images/shoulder_press.gif",
            6000, self.exercises.shoulderPress, "Shoulder Press"), "#6D67E4","#141414")
        workoutButtons("S Q U A T", lambda: self.exercises.showGifAndRunExercise("GIF_Images/squat.gif",
            6000, self.exercises.squats, "Squat"), "#46C2CB","#141414")
        workoutButtons("F R O N T   R A I S E",  lambda: self.exercises.showGifAndRunExercise("GIF_Images/front_raise.gif",
            6000, self.exercises.frontRaise, "Front Raise"), "#F2F7A1","#141414")
    
user_interface = FitnessPostureSystem()
