import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gtsdb import string_int_label_map_pb2
from gtsdb import visualization_utils as vu
import logging
from google.protobuf import text_format
from PIL import Image
from core import utils

class YoloTest:
    def __init__(self):
        self.IMAGE_H = 800
        self.IMAGE_W = 1360
        # self.classes = utils.read_coco_names('./data/coco.names')
        self.label_map = self.load_labelmap("./gtsdb/gtsdb3_label_map.pbtxt")
        self.categories = self.convert_label_map_to_categories(self.label_map, max_num_classes=3,
                                                          use_display_name=True)
        self.category_index = self.create_category_index(self.categories)
        self.classes = self.get_classes(self.category_index)
        self.num_classes = len(self.category_index)
        self.gpu_nms_graph = tf.Graph()
        self.model_path = "./gtsdb/frozen_inference_graph.pb"
        # self.input_tensor, self.output_tensors = utils.read_pb_return_tensors(
        #     self.gpu_nms_graph,
        #     "./checkpoint/yolov3_gpu_nms.pb",
        #     ["Placeholder:0", "concat_10:0", "concat_11:0", "concat_12:0"])
        self.input_tensor, self.output_tensors = utils.read_pb_return_tensors(
            self.gpu_nms_graph,
            "./gtsdb/frozen_inference_graph.pb",
            ["image_tensor:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"])
        self.sess = tf.Session(graph=self.gpu_nms_graph)
        self.last_PIL_image = None  # 原图
        self.last_boxes = None
        self.last_scores = None
        self.last_labels = None
        self.last_nd = None


    def get_classes(self, category_index):
        cate_classes = {}
        for i in category_index:
            cate_classes[i] = category_index[i]['name']
        return cate_classes

    def create_category_index(self, categories):
        category_index = {}
        for cat in categories:
            category_index[cat['id']] = cat
        return category_index

    def create_categories_from_labelmap(self, label_map_path, use_display_name=True):
        label_map = self.load_labelmap(label_map_path)
        max_num_classes = max(item.id for item in label_map.item)
        return self.convert_label_map_to_categories(label_map, max_num_classes,
                                               use_display_name)

    def create_category_index_from_labelmap(self, label_map_path, use_display_name=True):
        categories = self.create_categories_from_labelmap(label_map_path, use_display_name)
        return self.create_category_index(categories)

    def convert_label_map_to_categories(self, label_map,
                                        max_num_classes,
                                        use_display_name=True):
        categories = []
        list_of_ids_already_added = []
        if not label_map:
            label_id_offset = 1
            for class_id in range(max_num_classes):
                categories.append({
                    'id': class_id + label_id_offset,
                    'name': 'category_{}'.format(class_id + label_id_offset)
                })
            return categories
        for item in label_map.item:
            if not 0 < item.id <= max_num_classes:
                logging.info(
                    'Ignore item %d since it falls outside of requested '
                    'label range.', item.id)
                continue
            if use_display_name and item.HasField('display_name'):
                name = item.display_name
            else:
                name = item.name
            if item.id not in list_of_ids_already_added:
                list_of_ids_already_added.append(item.id)
                categories.append({'id': item.id, 'name': name})
        return categories

    def load_labelmap(self, path):
        with tf.gfile.GFile(path, 'r') as fid:
            label_map_string = fid.read()
            label_map = string_int_label_map_pb2.StringIntLabelMap()
            try:
                text_format.Merge(label_map_string, label_map)
            except text_format.ParseError:
                label_map.ParseFromString(label_map_string)
        self._validate_label_map(label_map)
        return label_map

    def _validate_label_map(self, label_map):
        for item in label_map.item:
            if item.id < 0:
                raise ValueError('Label map ids should be >= 0.')
            if (item.id == 0 and item.name != 'background' and
                    item.display_name != 'background'):
                raise ValueError('Label map id 0 is reserved for the background label')

    def read_image(self, image_path):
        img = Image.open(image_path)
        self.last_PIL_image = img.resize(size=(self.IMAGE_W, self.IMAGE_H))
        img_resized = np.array(self.last_PIL_image, dtype=np.float32)
        img_resized = img_resized / 255.
        return img_resized

    def predict_from_array(self, image):
        feed_dict = {self.input_tensor: np.expand_dims(image, axis=0)}
        (boxes, scores, labels, num_detections) = self.sess.run(self.output_tensors, feed_dict=feed_dict)
        self.last_boxes = boxes
        self.last_scores = scores
        self.last_labels = labels
        self.last_nd = num_detections
        return boxes, scores, labels

    def predict_from_path(self, image_path):
        image = self.read_image(image_path)
        return self.predict_from_array(image)

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

    def save_result(self, path, PIL_image=None, boxes=None, scores=None, labels=None, num_detections=None ):
        PIL_image = PIL_image or self.last_PIL_image
        boxes = boxes or self.last_boxes
        scores = scores or self.last_scores
        labels = labels or self.last_labels
        num_detections = num_detections or self.last_nd
        plt.imshow(PIL_image)
        plt.savefig("./data/HideAllTarget_result/test/pil.jpg")
        # PIL_image_result = utils.draw_boxes(PIL_image, np.squeeze(boxes), \
        #                                                np.squeeze(scores), \
        #                                                np.squeeze(labels).astype(np.int32), \
        #                                                self.classes,
        #                                     [self.IMAGE_H, self.IMAGE_W], show=False)
        # PIL_image_result.save(path)
        vu.visualize_boxes_and_labels_on_image_array(
            PIL_image,
            np.squeeze(boxes),
            np.squeeze(labels).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=5)
        # print(np.squeeze(boxes))
        # print(np.squeeze(labels).astype(np.int32))
        # print(np.squeeze(scores))
        plt.figure(figsize=(20, 20))
        plt.axis('off')
        plt.imshow(PIL_image)
        plt.savefig(path)