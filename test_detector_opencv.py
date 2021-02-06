from detection.opencv_dnn.detector import Detector
import cv2

cap = cv2.VideoCapture("videos/1.mp4")

weights = 'models/yolo/weights/yolov4_tiny.weights' 
config = 'models/yolo/configs/yolov4_tiny.cfg'


detector = Detector(weights, config, gpu=False, classes_name=None)
detector.init_yolo()



while cap.isOpened():
  _, frame = cap.read()
  if not _:
    break
  classes, scores, bb_list = detector.detect(frame=frame, confidence_threshold=0.4, nms_threshold=0.4)
  for (classid, score, box) in zip(classes, scores, bb_list):
      color = (0, 255, 0)
      label = str(classid[0])
      cv2.rectangle(frame, box, color, 2)
      cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
  frame = cv2.resize(frame, (640, 480))
  cv2.imshow("Test", frame)
  k = cv2.waitKey(5)
  if k == 32:
    break