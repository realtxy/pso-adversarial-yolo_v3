import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from core import utils
import cv2


class YoloTest:
    def __init__(self):
        self.IMAGE_H = 416
        self.IMAGE_W = 416
        self.classes = utils.read_coco_names('./data/coco.names')
        self.num_classes = len(self.classes)
        self.gpu_nms_graph = tf.Graph()
        self.input_tensor, self.output_tensors = utils.read_pb_return_tensors(
            self.gpu_nms_graph,
            "./checkpoint/yolov3_gpu_nms.pb",
            ["Placeholder:0", "concat_10:0", "concat_11:0", "concat_12:0"])
        self.sess = tf.Session(graph=self.gpu_nms_graph)
        self.last_PIL_image = None  # 原图
        self.last_boxes = None
        self.last_scores = None
        self.last_labels = None
        self.colors = [[254.0, 254.0, 254], [248.92, 228.6, 127], [243.84, 203.20000000000002, 0], [238.76, 177.79999999999998, -127], [233.68, 152.4, -254], [228.6, 127.0, 254], [223.52, 101.60000000000001, 127], [218.44, 76.20000000000002, 0], [213.35999999999999, 50.79999999999999, -127], [208.28000000000003, 25.399999999999995, -254], [203.20000000000002, 0.0, 254], [198.12, -25.400000000000023, 127], [193.04, -50.79999999999999, 0], [187.96, -76.20000000000002, -127], [182.88, -101.59999999999998, -254], [177.79999999999998, -127.0, 254], [172.71999999999997, -152.40000000000003, 127], [167.64, -177.79999999999998, 0], [162.56, -203.20000000000002, -127], [157.48, -228.59999999999997, -254], [152.4, -254.0, 254], [147.32000000000002, -279.40000000000003, 127], [142.24, -304.80000000000007, 0], [137.16, -330.19999999999993, -127], [132.08, -355.59999999999997, -254], [127.0, 254.0, 254], [121.92, 228.6, 127], [116.83999999999999, 203.20000000000002, 0], [111.75999999999999, 177.79999999999998, -127], [106.68, 152.4, -254], [101.60000000000001, 127.0, 254], [96.52, 101.60000000000001, 127], [91.44, 76.20000000000002, 0], [86.35999999999999, 50.79999999999999, -127], [81.27999999999999, 25.399999999999995, -254], [76.20000000000002, 0.0, 254], [71.12, -25.400000000000023, 127], [66.04, -50.79999999999999, 0], [60.96, -76.20000000000002, -127], [55.879999999999995, -101.59999999999998, -254], [50.79999999999999, -127.0, 254], [45.72000000000001, -152.40000000000003, 127], [40.64000000000001, -177.79999999999998, 0], [35.56, -203.20000000000002, -127], [30.48, -228.59999999999997, -254], [25.399999999999995, -254.0, 254], [20.31999999999999, -279.40000000000003, 127], [15.240000000000013, -304.80000000000007, 0], [10.160000000000009, -330.19999999999993, -127], [5.0800000000000045, -355.59999999999997, -254], [0.0, 254.0, 254], [-5.0800000000000045, 228.6, 127], [-10.160000000000009, 203.20000000000002, 0], [-15.240000000000013, 177.79999999999998, -127], [-20.320000000000018, 152.4, -254], [-25.400000000000023, 127.0, 254], [-30.480000000000025, 101.60000000000001, 127], [-35.559999999999974, 76.20000000000002, 0], [-40.63999999999998, 50.79999999999999, -127], [-45.719999999999985, 25.399999999999995, -254], [-50.79999999999999, 0.0, 254], [-55.879999999999995, -25.400000000000023, 127], [-60.96, -50.79999999999999, 0], [-66.04, -76.20000000000002, -127], [-71.12, -101.59999999999998, -254], [-76.20000000000002, -127.0, 254], [-81.28000000000002, -152.40000000000003, 127], [-86.36000000000001, -177.79999999999998, 0], [-91.44000000000003, -203.20000000000002, -127], [-96.51999999999997, -228.59999999999997, -254], [-101.59999999999998, -254.0, 254], [-106.67999999999998, -279.40000000000003, 127], [-111.75999999999999, -304.80000000000007, 0], [-116.83999999999999, -330.19999999999993, -127], [-121.92, -355.59999999999997, -254], [-127.0, 254.0, 254], [-132.08, 228.6, 127], [-137.16, 203.20000000000002, 0], [-142.24, 177.79999999999998, -127], [-147.32000000000002, 152.4, -254]]
        self.label_name = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
                           'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                           'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                           'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    def read_image(self, image_path):
        img = Image.open(image_path)
        self.last_PIL_image = img.resize(size=(self.IMAGE_W, self.IMAGE_H))
        img_resized = np.array(self.last_PIL_image, dtype=np.float32)
        # img_resized = img_resized / 255.
        return img_resized

    def predict_from_array(self, image):
        image = image / 255.
        feed_dict = {self.input_tensor: np.expand_dims(image, axis=0)}
        boxes, scores, labels = self.sess.run(self.output_tensors, feed_dict=feed_dict)
        self.last_boxes = boxes
        self.last_scores = scores
        self.last_labels = labels
        h, w, _ = image.shape
        thick = int((h + w) // 300)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        drawer = image.copy()
        h, w, _ = drawer.shape
        for i in range(len(boxes)):
            cv2.putText(drawer, str(self.label_name[labels[i]]) + ' ' + str(scores[i]),
                        (int(boxes[i][0]), int(boxes[i][1]) - 12), 0, 1e-3 * h, self.colors[labels[i]], thick // 3)
            cv2.rectangle(drawer, (int(boxes[i][0]), int(boxes[i][1])),
                          (int(boxes[i][2]), int(boxes[i][3])), self.colors[labels[i]], thick)
        cv2.imshow("result", drawer)
        cv2.waitKey(1)
        return boxes, scores, labels

    def predict_from_path(self, image_path):
        image = self.read_image(image_path)
        return self.predict_from_array(image)

    def save_result(self, path, PIL_image=None, boxes=None, scores=None, labels=None):
        PIL_image = PIL_image or self.last_PIL_image
        boxes = boxes or self.last_boxes
        scores = scores or self.last_scores
        labels = labels or self.last_labels
        PIL_image_result = utils.draw_boxes(PIL_image, boxes, scores, labels, self.classes, [self.IMAGE_H, self.IMAGE_W], show=False)
        PIL_image_result.save(path)





    def visualize_result(self, PIL_image=None, boxes=None, scores=None, labels=None):
        PIL_image = PIL_image or self.last_PIL_image
        boxes = boxes or self.last_boxes
        scores = scores or self.last_scores
        labels = labels or self.last_labels
        PIL_image_result = utils.draw_boxes(PIL_image, boxes, scores, labels, self.classes, [self.IMAGE_H, self.IMAGE_W])
        np_image = np.array(PIL_image_result, dtype=np.int32)
        # plt.figure(figsize=(10, 10))
        plt.imshow(np_image)
        plt.show()


# if __name__ == '__main__':
#     yolo = YoloTest()
#     boxes, scores, labels = yolo.predict_from_path("./data/demo_data/car1.jpg")
#
#     print(boxes, scores, labels)
#     yolo.visualize_result()