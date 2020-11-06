import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Yolo:

    def __init__(self, names="cfg/person.names", weights="cfg/yolov3.weights", cfg="cfg/yolov3.cfg"):
        """
          Yolo initialization
          param:
                  names: mapping between class and names.
                  weights: file containing the pre-trained weights.
                  cfg: YOLO's config file
        """
        with open(names, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.net = cv2.dnn.readNet(weights, cfg)
        self.reso = (416, 416)

    def detect(self, image):
        """
        Start the detection
        param:
                image: input image to perform detection on.
        """
        Width = image.shape[1]
        Height = image.shape[0]
        # create input blob
        # set input blob for the network
        self.net.setInput(cv2.dnn.blobFromImage(image, 0.00392, self.reso, (0, 0, 0), True, crop=False))
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        outs = self.net.forward(output_layers)
        class_ids = []
        confidences = []
        boxes = []
        # create bounding box
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
        boxlist = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            if class_ids[i] == 0:
                boxlist.append(box)
        return boxlist
