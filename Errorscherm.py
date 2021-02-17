import tkinter as tk
import serial
import time

count = 1
ignore = False


def error_msg():
    frequency = 2000  # Set Frequency To 2000 Hertz
    duration = 800  # Set Duration To 800 ms == 0.8 second
    winsound.Beep(frequency, duration)
    time.sleep(0.1)
    winsound.Beep(frequency, duration)
    time.sleep(0.1)
    winsound.Beep(frequency, duration)

    root = tk.Tk()
    root.geometry('381x210')
    root.title("Error detector")
    root.configure(background="black")
    canvas=tk.Canvas(root, width=381, height=210, bg="black")
    canvas.pack()

    img = tk.PhotoImage(file="errorscreen.png")
    myimg = canvas.create_image(0,0, anchor="nw", image=img)

# Counter on screen
    tekst = tk.Label(root, text="Dit scherm sluit automatisch in:", bg="#ece9d9", fg="grey")
    tekst.pack()
    canvas.create_window(100, 190, window=tekst)

    def countdown(cntdwn):
        # change text in label
        countdowntekst['text'] = cntdwn

        if cntdwn > 0:
            # call countdown again after 1000ms (1s)
            root.after(1000, countdown, cntdwn - 1)
        else:
            root.destroy()

    countdowntekst = tk.Label(root, bg="#ece9d9", fg="grey")
    countdowntekst.place(x=185, y=180)

    # call countdown first time
    countdown(10)
    # root.after(0, countdown, 10)

# Code to stop 3D-printer
    def stopcode():
        global ignore
        """ser = serial.Serial('COM3', 115200, dsrdtr=True)
        time.sleep(2)
        ser.write(str.encode("G28 X0 Y0\r\n"))  #Home X- and Y-axis
        ser.write(str.encode("M84\r\n"))        #Disable steppers
        ser.write(str.encode("M106 S0\r\n"))    #Turn off fan
        ser.write(str.encode("M104 S0\r\n"))    #Turn of hot-end
        ser.write(str.encode("M140 S0\r\n"))    #Turn off bed
        time.sleep(1)"""
        print("stopping print")
        root.destroy()
        ignore = True

    stopknop = tk.Button(root, text="Stop printer", padx=10, pady=5, fg="black", bg="red", command=stopcode)
    stopknop.place(x=220, y=130)

# Code to ignore error screen
    def negeercode():
        global ignore
        ignore = True
        root.destroy()
        time.sleep(10)   # Time to snooze error screen
        ignore = False

    ignorebutton = tk.Button(root, text="Negeer errors", padx=10, pady=5, fg="black", bg="#ece9d9", command=negeercode)
    ignorebutton.place(x=100, y=130)


    root.resizable(False, False)
    root.mainloop()

# Delay to show user interface after detection (Eliminate false positives)
# outside the loop
from datetime import datetime, timedelta
import winsound
first_seen= None

# inside the loop
while True:
    if count > 0 and first_seen is None:
        first_seen = datetime.now()
    elif count == 0:
        first_seen = None
    elif count > 0 and datetime.now() - first_seen > timedelta(seconds=3) and ignore == False:
        print("Count has been greater than 0 for more than 3 seconds")
        error_msg()











