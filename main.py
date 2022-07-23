#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
import cv2
import numpy as np
import onnxruntime

class HAWP:
    def __init__(self, conf_thres=0.95):
        self.conf_threshold = conf_thres
        # Initialize model
        self.onnx_session = onnxruntime.InferenceSession("hawp_512x512_float32.onnx")
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

        self.input_shape = self.onnx_session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1,1,3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1,1,3)
    def prepare_input(self, image):
        input_image = cv2.resize(image, dsize=(self.input_width, self.input_height))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = (input_image.astype(np.float32) / 255.0 - self.mean) / self.std
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        return input_image

    def detect(self, image):
        input_image = self.prepare_input(image)

        # Perform inference on the image
        lines, scores = self.onnx_session.run(None, {self.input_name: input_image})

        # Post process:squeeze, RGB->BGR, Transpose, uint8 cast
        image_width, image_height = image.shape[1], image.shape[0]
        for line in lines:
            line[0] = int(line[0] / 128 * image_width)
            line[1] = int(line[1] / 128 * image_height)
            line[2] = int(line[2] / 128 * image_width)
            line[3] = int(line[3] / 128 * image_height)
        # Draw Line
        dst_image = copy.deepcopy(image)
        for line, score in zip(lines, scores):
            if score < self.conf_threshold:
                continue
            x1, y1 = int(line[0]), int(line[1])
            x2, y2 = int(line[2]), int(line[3])
            cv2.line(dst_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return dst_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='sample.png', help="image path")
    parser.add_argument('--confThreshold', default=0.95, type=float, help='class confidence')
    args = parser.parse_args()

    detector = HAWP(conf_thres=args.confThreshold)
    srcimg = cv2.imread(args.imgpath)

    dstimg = detector.detect(srcimg)
    cv2.namedWindow('srcimg', 0)
    cv2.imshow('srcimg', srcimg)
    winName = 'Deep learning Holistically-Attracted Wireframe Parsing in ONNXRuntime'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, dstimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()