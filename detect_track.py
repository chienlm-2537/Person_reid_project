from tracking.deep_sort import preprocessing
from tracking.deep_sort import nn_matching
from tracking.deep_sort.detection import Detection
from tracking.deep_sort.tracker import Tracker
from tracking.deep_sort import generate_detections as gdet

from detection.opencv_dnn.detector import Detector
import cv2
import time

cap = cv2.VideoCapture("videos/1.mp4")

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

while cap.isOpened():
    _, frame = cap.read()
    if not _:
        break
    start = time.time()
    classes, scores, bb_list = detector.detect(frame=frame, confidence_threshold=0.4, nms_threshold=0.4)
    features = encoder(frame, boxes=bb_list)
    detections = [ Detection(bbox, confidence, cls, feature)
                for bbox, confidence, cls, feature in zip(
            bb_list, scores, classes, features)]
    tracker.predict()
    tracker.update(detections)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        cv2.rectangle(
            frame,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "ID: " + str(track.track_id),
            (int(bbox[0]), int(bbox[1])),
            0,
            1,
            (0, 0, 255),
            2,
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
        if len(classes) > 0:
            cls = det.cls
            cv2.putText(
                frame,
                str(cls) + " " + score,
                (int(bbox[0]), int(bbox[3])),
                0,
                0.5,
                (0, 255, 0),
                2,
            )
    end = time.time()
    fps_label = "FPS: %.2f" % (1 / (end - start))
    cv2.putText(
        frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
    )
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("deepsort", frame)

    if cv2.waitKey(1) & 0xFF == 32:
        break
cv2.destroyAllWindows()
