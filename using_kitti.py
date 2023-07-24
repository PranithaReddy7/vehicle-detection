from google.colab import drive
drive.mount('/content/drive')
!mkdir /content/drive/MyDrive/yolov3-tiny-kitti-resultsss/

import cv2
import numpy as np
import os

from google.colab import drive
drive.mount('/content/drive')

zip_path = '/content/drive/MyDrive/data_object_image_2.zip/data_object_image_2/testing/image_2/'

import zipfile
with zipfile.ZipFile('/content/drive/MyDrive/data_object_image_2.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/data_object_image_2')

classes = ["Truck", "Van", "Car"]
with open("kitti.names", "w") as f:
    f.write("\n".join(classes))

image_path = '/content/data_object_image_2/testing/image_2/'
model_path = '/content/drive/MyDrive/yolov3-tiny.weights'
config_path = '/content/drive/MyDrive/yolov3-tiny.cfg'
class_path = '/content/drive/MyDrive/kitti.names'

net = cv2.dnn.readNet(model_path, config_path)

with open('kitti.names') as f:
    classes = f.read().splitlines()

for filename in sorted(os.listdir(image_path)):

    img = cv2.imread(os.path.join(image_path, filename))

    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)

    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    pred_boxes = []
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
                 boxes.append([x, y, w, h])
                 confidences.append(float(confidence))
                 class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0 and len(classes) > 0 and len(confidences) > 0:
       for i in indices.flatten():
          x, y, w, h = boxes[i]
          label = '{}: {:.2f}'.format(classes[class_ids[i]], confidences[i])
          cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
          cv2.putText(img, label, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite('/content/drive/MyDrive/yolov3-tiny-kitti-resultsss/' + filename, img)

