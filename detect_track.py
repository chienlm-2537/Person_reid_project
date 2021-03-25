import random as rd
import time

import cv2

from detection.opencv_dnn.detector import Detector
from tracking.deep_sort import generate_detections as gdet
from tracking.deep_sort import nn_matching, preprocessing
from tracking.deep_sort.detection import Detection
from tracking.deep_sort.tracker import Tracker
from utils import check_position, draw, mouse_event
from face_recog.face_detection.face_detector import get_detector, face_detect
import imutils
import numpy as np

COLOR = [tuple([rd.randint(0, 255)]*3), tuple([rd.randint(0, 255)]*3), tuple([rd.randint(0, 255)]*3)]
COLOR = [(255, 255, 0), (204, 0, 153), (26, 209, 255), (71, 107, 107)]
print(COLOR)
cap = cv2.VideoCapture("videos/01.mp4")

weights = "models/yolo/weights/yolov4_tiny.weights"
config = "models/yolo/configs/yolov4_tiny.cfg"
classes = "models/yolo/classes.txt"

detector = Detector(weights, config, gpu=False, classes_name=classes)
detector.init_yolo()
print("===============================================================")
MAX_COSINE_DISTANCE = 0.3
nn_budget = None

model_filename = "models/deepsort_model/mars-small128.pb"
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, nn_budget)
tracker = Tracker(metric)

# Define face detection model
face_detector = get_detector(hog=True)



cv2.namedWindow("Test")
cv2.setMouseCallback("Test", mouse_event)



points = [(264, 295), (405, 286), (430, 359), (254, 373)]

while cap.isOpened():
    _, frame = cap.read()
    if not _:
        break
    frame = cv2.resize(frame, (640, 480))
    start = time.time()
    classes, scores, bb_list = detector.detect(frame=frame, confidence_threshold=0.4, nms_threshold=0.4)
    s1 = time.time()
    # print("Detection time: {}".format(s1 - start))

    # after detection we will check position of object
    bb_check = dict()
    check_insides = []
    for box in bb_list:
      check, center_point = check_position(box, points)
      check_insides.append(check)

    features = encoder(frame, boxes=bb_list)
    detections = [ Detection(bbox, confidence, cls, feature, check_inside)
                for bbox, confidence, cls, feature, check_inside in zip(
            bb_list, scores, classes, features, check_insides)]
    tracker.predict()
    tracker.update(detections)
    s2 = time.time()
    # print("Tracking time: {}".format(s2-s1))
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                      COLOR[track.track_id%3],2)
        obj_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        scale_factor = 450 / obj_img.shape[1]
        obj_img = imutils.resize(obj_img, width=450)
        s3 = time.time()
        face_boxes = face_detect(obj_img, face_detector)
        # print("Face detection time: {}".format(time.time() - s3))
        for box in face_boxes:
          box = np.array(box) * (1/scale_factor)
          x_min, x_max, y_min, y_max = box
          x_min_real = int(x_min + bbox[0])
          y_min_real = int(y_min + bbox[1])
          x_max_real = int(x_max + bbox[0])
          y_max_real = int(y_max + bbox[1])
          # cv2.rectangle(obj_img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 3)
          cv2.rectangle(frame, (x_min_real, y_min_real), (x_max_real, y_max_real), (0, 255, 0), 2)
        # cv2.imshow("Obj ID {}".format(str(track.track_id)), obj_img)

        # cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 1, (0, 0, 255), 2)


    # for det in detections:
    #     bbox = det.to_tlbr()
    #     score = "%.2f" % round(det.confidence * 100, 2) + "%"
    #     cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),(0, 0, 255),2)
    end = time.time()


    fps_label = "FPS: %.2f" % (1 / (end - start))
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    frame = draw(frame, points)
    cv2.imshow("Test", frame)

    if cv2.waitKey(1) & 0xFF == 32:
        break
    if cv2.waitKey(1) == ord("p"):
      print("Pause")
      cv2.waitKey(0)
cv2.destroyAllWindows()
