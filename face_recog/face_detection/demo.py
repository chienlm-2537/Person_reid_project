from face_detector import get_detector, detect
import cv2
import time
import imutils
image  = cv2.imread("/home/le.minh.chien/Downloads/Market/pytorch/gallery/1502/835.png")
image = imutils.resize(image, width=300)
using_hog = False

detector = get_detector(using_hog)

start = time.time()
bbox = detect(image, detector, using_hog)
print("Processing time for face detection {}".format(time.time() - start))


for box in bbox:
    x_min, x_max, y_min, y_max = box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
cv2.imshow("Yoona", image)
cv2.waitKey(0)