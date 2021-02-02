import serial
import time

ser = serial.Serial('COM3', 115200)

time.sleep(2)
ser.write(str.encode("G28\r\n"))
time.sleep(1)
