import numpy as np
import tensorflow as tf
from PIL import Image
from core import utils
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True


IMAGE_H, IMAGE_W = 416, 416
classes = utils.read_coco_names('./data/raccoon.names')
num_classes = len(classes)
image_path = "data/hide_specific_target_dataset/000000336584.jpg"  # 181,
img = Image.open(image_path)
img_resized = np.array(img.resize(size=(IMAGE_W, IMAGE_H)), dtype=np.float32)
img_resized = img_resized / 255.
cpu_nms_graph = tf.Graph()

input_tensor, output_tensors = utils.read_pb_return_tensors(cpu_nms_graph, "./checkpoint/yolov3_cpu_nms.pb",
                                           ["Placeholder:0", "concat_9:0", "mul_6:0"])
with tf.Session(graph=cpu_nms_graph) as sess:
    boxes, scores = sess.run(output_tensors, feed_dict={input_tensor: np.expand_dims(img_resized, axis=0)})
    boxes, scores, labels = utils.cpu_nms(boxes, scores, num_classes, score_thresh=0.3, iou_thresh=0.5)
    image = utils.draw_bdeoxes(img, boxes, scores, labels, classes, [IMAGE_H, IMAGE_W], show=True)
