import winsound
import time

frequency = 2000  # Set Frequency To 2500 Hertz
duration = 800  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)
time.sleep(0.1)
winsound.Beep(frequency, duration)
time.sleep(0.1)
winsound.Beep(frequency, duration)