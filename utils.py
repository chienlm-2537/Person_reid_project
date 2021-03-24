import cv2
import numpy as np



def mouse_event(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONDBLCLK:
    print(x, y)


def draw(frame, points):
    cv2.polylines(frame, np.array([points]), 1, (255, 0, 0), 1)
    if len(points) >1:
      b, g, r = cv2.split(frame)
      cv2.fillConvexPoly(b, np.array(points, 'int32'), (0, 255, 0))
      cv2.fillConvexPoly(r, np.array(points, 'int32'), (0, 255, 0))           
      frame = cv2.merge([b, g, r])
    return frame


def check_position(obj_coordinate, points):
  """
  - Input: 
          + obj_coordinate (x_min, y_min, weight, height)
          + points: the coordinate of prohibited area ( top-left, top-right, bottom-right, bottom-left)
  - Ouput:
          + check = 0, 1 if object inside the area
          + check = -1 if object not inside the area 
  
  """
  x_min, y_min, weight, height = obj_coordinate
  x_center = int(x_min + weight/2)
  y_center = int(y_min + height)
  # print("[INFO] line 34 utils center point coor: ({}, {})".format( x_center, y_center))
  check = cv2.pointPolygonTest(np.array([points]), (x_center, y_center), False)
  return check, (x_center, y_center)