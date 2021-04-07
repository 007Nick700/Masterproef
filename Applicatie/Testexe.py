import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import cv2
import argparse
import time
import winsound
import serial

import tensorflow as tf
import numpy as np
import tkinter as tk

from object_detection.utils import label_map_util
from PIL import Image

tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Folder that the Saved Model is Located In',
                    default='exported-models/my_mobilenet_model')
parser.add_argument('--labels', help='Where the Labelmap is Located',
                    default='exported-models/my_mobilenet_model/saved_model/label_map.pbtxt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.3)

args, unknown = parser.parse_known_args()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)



def everything():
    startscreen.destroy()
    # PROVIDE PATH TO MODEL DIRECTORY
    PATH_TO_MODEL_DIR = 'C:/tensorflow2/models/research/object_detection/inference_graph'

    # PROVIDE PATH TO LABEL MAP
    PATH_TO_LABELS = 'C:/tensorflow2/models/research/object_detection/training/label_map.pbtxt'

    # PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
    MIN_CONF_THRESH = float(args.threshold)

    PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

    print('Loading model...', end='')
    start_time = time.time()

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)

    def load_image_into_numpy_array(path):
        """Load an image from file into a numpy array.
        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.
        Args:
          path: the file path to the image
        Returns:
          uint8 numpy array with shape (img_height, img_width, 3)
        """
        return np.array(Image.open(path))

    print('Running inference for Webcam', end='')

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
        canvas = tk.Canvas(root, width=381, height=210, bg="black")
        canvas.pack()

        img = tk.PhotoImage(file="errorscreen.png")
        myimg = canvas.create_image(0, 0, anchor="nw", image=img)

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
        countdown(20)

        # root.after(0, countdown, 10)

        # Code to stop 3D-printer
        def stopcode():
            global ignore
            global stop
            root.destroy()
            print("Stopping print")
            ignore = True
            stop = True
            cv2.destroyAllWindows()

            """ser = serial.Serial('COM3', 115200, dsrdtr=True)
            time.sleep(2)
            ser.write(str.encode("G28 X0 Y0\r\n"))  # Home X- and Y-axis
            ser.write(str.encode("M84\r\n"))  # Disable steppers
            ser.write(str.encode("M106 S0\r\n"))  # Turn off fan
            ser.write(str.encode("M104 S0\r\n"))  # Turn of hot-end
            ser.write(str.encode("M140 S0\r\n"))  # Turn off bed
            time.sleep(1)"""

        stopknop = tk.Button(root, text="Stop printer", padx=10, pady=5, fg="black", bg="red", command=stopcode)
        stopknop.place(x=220, y=130)

        # Code to ignore error screen
        def ignorecode():
            global ignore
            global ignorecount
            ignore = True
            root.destroy()
            ignorecount = 0

        ignorebutton = tk.Button(root, text="Negeer errors", padx=10, pady=5, fg="black", bg="#ece9d9",
                                 command=ignorecode)
        ignorebutton.place(x=100, y=130)

        root.resizable(False, False)
        root.mainloop()

    videostream = cv2.VideoCapture(0)
    ret = videostream.set(3, 1000)
    ret = videostream.set(4, 720)

    outputFrameIndices = []
    frame_counter = 0
    Threshold = 5
    ignorecount = 0
    ignore = False
    stop = False

    while True:
        ignorecount = ignorecount + 1
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = videostream.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)
        imH, imW, _ = frame.shape

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(frame)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        # SET MIN SCORE THRESH TO MINIMUM THRESHOLD FOR DETECTIONS

        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        scores = detections['detection_scores']
        boxes = detections['detection_boxes']
        classes = detections['detection_classes']
        count = 0
        for i in range(len(scores)):
            if ((scores[i] > MIN_CONF_THRESH) and (scores[i] <= 1.0)):
                # increase count
                count += 1
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
                # Draw label
                object_name = category_index[int(classes[i])][
                    'name']  # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))  # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size
                label_ymin = max(ymin, labelSize[1] + 10)  # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                              (xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255),
                              cv2.FILLED)  # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),
                            2)  # Draw label text

        cv2.putText(frame, 'Errors Detected : ' + str(count), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (70, 235, 52), 2,
                    cv2.LINE_AA)
        cv2.imshow('Error Detector', frame)

        if count >= 1 and ignore == False and stop == False:
            frame_counter = frame_counter + 1
            ret, frame = videostream.read()  # read current frame
            outputFrameIndices.append(frame_counter)
            print(frame_counter)
            if frame_counter > Threshold:
                error_msg()
        else:
            frame_counter = 0

        if ignore == True and ignorecount >= 150:
            ignore = False

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    print(" Done")


startscreen = tk.Tk()
startscreen.geometry('381x210')
startscreen.title("Error detector")
startscreen.configure(background="black")
canvass = tk.Canvas(startscreen, width=381, height=210, bg="white")
canvass.pack()
startknop = tk.Button(startscreen, text="Start shit", padx=10, pady=5, fg="black", bg="red", command=everything)
startknop.place(x=10, y=10)
startscreen.mainloop()