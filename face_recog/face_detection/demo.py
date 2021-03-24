from face_detector import get_detector, detect
import cv2
import time
import imutils
# image  = cv2.imread("/home/le.minh.chien/Downloads/Market/pytorch/gallery/1502/835.png")
# image = imutils.resize(image, width=300)
using_hog = True

detector = get_detector(using_hog)

# start = time.time()
# bbox = detect(image, detector, using_hog)
# print("Processing time for face detection {}".format(time.time() - start))


# for box in bbox:
#     x_min, x_max, y_min, y_max = box
#     cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
# cv2.imshow("Yoona", image)
# cv2.waitKey(0)


cap = cv2.VideoCapture("/home/yoona/Desktop/test/outpy1616144005.0.mp4")
while cap.isOpened():
  _, frame = cap.read()
  if not _:
    break

  print("Test")
  bbox = detect(frame, detector, using_hog)

  for box in bbox:
      x_min, x_max, y_min, y_max = box
      cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

  # time.sleep(0.01)
  frame = cv2.resize(frame, (640, 480))
  cv2.imshow("Testtttttttttt", frame)
  k = cv2.waitKey(5)
  if k == 32:
    break
