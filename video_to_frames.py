from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('/content/drive/MyDrive/framesss/', exist_ok=True)

import cv2
import numpy as np

video_path = '/content/drive/MyDrive/front-video.mp4'
save_path = '/content/drive/MyDrive/framesss/'

cap = cv2.VideoCapture(video_path)

frame_count = 0
read_first_frame = True

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    if read_first_frame:
        read_first_frame = False
        continue

    filename = save_path + 'frame{}.jpg'.format(frame_count)
    cv2.imwrite(filename, frame)

    print('Processed frame {}...'.format(frame_count))

cap.release()

