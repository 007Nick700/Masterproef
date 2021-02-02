import tkinter as tk
import serial
import time

count = 1
negeer = False


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
    #frame = tk.Frame(root, bg="white")
    #frame.place(relwidth=0.9, relheight=0.8, relx=0.05, rely=0.1)

#####################################################################################################

    img = tk.PhotoImage(file="errorscreen.png")
    myimg = canvas.create_image(0,0, anchor="nw", image=img)

########################################TELLER OP SCHERM#############################################################
    tekst = tk.Label(root, text="Dit scherm sluit automatisch in:", bg="#ece9d9", fg="grey")
    tekst.pack()
    canvas.create_window(100, 190, window=tekst)



    def countdown(teller):
        # change text in label
        countdowntekst['text'] = teller

        if teller > 0:
            # call countdown again after 1000ms (1s)
            root.after(1000, countdown, teller - 1)
        else:
            root.destroy()

    countdowntekst = tk.Label(root, bg="#ece9d9", fg="grey")
    countdowntekst.place(x=185, y=180)

    # call countdown first time
    countdown(60)
    # root.after(0, countdown, 10)

########################################CODE OM TE STOPPEN#######################################################
    def stopcode():
        global negeer
        """ser = serial.Serial('COM3', 115200)
        time.sleep(2)
        ser.write(str.encode("G28\r\n"))
        time.sleep(1)"""
        print("stopping print")
        root.destroy()
        negeer = True

    stopknop = tk.Button(root, text="Stop printer", padx=10, pady=5, fg="black", bg="red", command=stopcode)
    stopknop.place(x=220, y=130)

#######################################CODE OM SCHERM TE NEGEREN########################################################
    def negeercode():
        global negeer
        negeer = True
        root.destroy()
        time.sleep(10)   #Time to snooze ignorebutton
        negeer = False

    negeerknop = tk.Button(root, text="Negeer errors", padx=10, pady=5, fg="black", bg="#ece9d9", command=negeercode)
    negeerknop.place(x=100, y=130)

#####################################################################################################################

    root.resizable(False, False)
    root.mainloop()

#######################################VERTRAGING OM GUI TE LATEN ZIEN##########################################
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
    elif count > 0 and datetime.now() - first_seen > timedelta(seconds=3) and negeer == False:
        print("Count has been greater than 0 for more than 3 seconds")
        error_msg()











