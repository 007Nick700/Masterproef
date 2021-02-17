import serial
import time

ser = serial.Serial('COM3', 115200, dsrdtr=True)

time.sleep(2)
ser.write(str.encode("G28 X0 Y0\r\n"))  #Home X- and Y-axis
ser.write(str.encode("M84\r\n"))        #Disable steppers
ser.write(str.encode("M106 S0\r\n"))    #Turn off fan
ser.write(str.encode("M104 S0\r\n"))    #Turn of hot-end
ser.write(str.encode("M140 S0\r\n"))    #Turn off bed
time.sleep(1)
