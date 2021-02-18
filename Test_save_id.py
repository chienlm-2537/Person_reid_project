import os
import time

import cv2

from detection.opencv_dnn.detector import Detector
from tracking.deep_sort import generate_detections as gdet
from tracking.deep_sort import nn_matching, preprocessing
from tracking.deep_sort.detection import Detection
from tracking.deep_sort.tracker import Tracker


def crop_and_save(ID: int, frame, bbox, frame_num):
    if not os.path.exists("past_image/" + str(ID)):
        os.makedirs("past_image/" + str(ID))
    # int(bbox[0])
    try:
        crop_image = frame[
            int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])
        ]
        cv2.imwrite(
            "past_image/" + str(ID) + "/" + str(frame_num) + ".png", crop_image
        )
    except:
        print("[INFO] Line 20 Error when save image")


cap = cv2.VideoCapture("videos/4.mp4")

w = int(cap.get(3))
h = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("outputTestV3.avi", fourcc, 30, (w, h))

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
metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", MAX_COSINE_DISTANCE, nn_budget
)
tracker = Tracker(metric)
frame_num = 0
while cap.isOpened():
    _, frame = cap.read()

    if not _:
        break
    frame1 = frame.copy()
    frame_num += 1
    start = time.time()
    classes, scores, bb_list = detector.detect(
        frame=frame, confidence_threshold=0.4, nms_threshold=0.4
    )
    features = encoder(frame, boxes=bb_list)
    detections = [
        Detection(bbox, confidence, cls, feature)
        for bbox, confidence, cls, feature in zip(
            bb_list, scores, classes, features
        )
    ]
    tracker.predict()
    tracker.update(detections)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        crop_and_save(track.track_id, frame1, bbox, frame_num)
        cv2.rectangle(
            frame,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (255, 255, 255),
            2,
        )
        #   print((int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])))
        cv2.putText(
            frame,
            str(track.track_id),
            (
                int(bbox[0]) + int((int(bbox[2]) - int(bbox[0])) / 3),
                int(bbox[1]) + int((int(bbox[3]) - int(bbox[1])) / 2),
            ),
            0,
            2,
            (0, 255, 0),
            3,
        )
    for det in detections:
        bbox = det.to_tlbr()
        score = "%.2f" % round(det.confidence * 100, 2) + "%"
        cv2.rectangle(
            frame,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 0, 255),
            2,
        )
    #   if len(classes) > 0:
    #       cls = det.cls
    #       cv2.putText(frame, str(cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
    #                   0.5, (0, 255, 0), 2)
    end = time.time()
    fps_label = "FPS: %.2f" % (1 / (end - start))
    cv2.putText(
        frame, fps_label, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3
    )
    out.write(frame)
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("deepsort", frame)

    if cv2.waitKey(1) & 0xFF == 32:
        break
cv2.destroyAllWindows()
out.release()
