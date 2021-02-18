import cv2


class Detector:
    def __init__(
        self,
        path_to_weights: str,
        path_to_config: str,
        gpu: bool,
        classes_name,
    ) -> None:
        """
      gpu: True if using gpu to inference
      classes_name: txt file
      """
        self.weights = path_to_weights
        self.config = path_to_config
        self.use_gpu = gpu
        with open(classes_name, "r") as f:
            class_names = [cname.strip() for cname in f.readlines()]
        self.classes_name = class_names

    def init_yolo(self):
        try:
            net = cv2.dnn.readNet(self.weights, self.config)
            if self.use_gpu:
                net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            self.model = cv2.dnn_DetectionModel(net)
            self.model.setInputParams(size=(416, 416), scale=1 / 255)
            print("Loading model successfully")
        except:
            print("Error when load model from file")
            exit()

    def detect(self, frame, confidence_threshold, nms_threshold):
        classes, score, bb_list = self.model.detect(
            frame, confidence_threshold, nms_threshold
        )
        return classes, score, bb_list
