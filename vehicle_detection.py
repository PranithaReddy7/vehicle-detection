import cv2
import numpy as np
from google.colab.patches import cv2_imshow

from google.colab import drive
drive.mount('/content/drive')

frame_path = '/content/drive/MyDrive/frames/'

net = cv2.dnn.readNet('/content/drive/MyDrive/yolov3-tiny.weights', '/content/drive/MyDrive/yolov3-tiny.cfg')

with open('coco.names') as f:
    classes = f.read().splitlines()

for i in range(21,23):

    print(frame_path + 'frame{}.jpg'.format(i))
    frame = cv2.imread(frame_path + 'frame{}.jpg'.format(i+1))

    height, width, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)

    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)

    # Loop through the detections and draw bounding boxes around the vehicles
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 2 and confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with bounding boxes around the vehicles
    cv2_imshow(frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()

