import dlib

# detector = dlib.dlib.get_frontal_face_detector()

def get_detector(hog=True):
    if hog:
        return dlib.get_frontal_face_detector()
    else:
        return dlib.cnn_face_detection_model_v1("../../models/face_detection/mmod_dlib.dat")





def face_detect(image, detector, hog= True):
    faceRects = detector(image, 0)
    bbox = []
    if hog:
        for face in faceRects:
            x_min = face.left()
            y_min = face.top()
            x_max = face.right()
            y_max = face.bottom()
            bbox.append((x_min, x_max, y_min, y_max))
    else:
        for face in faceRects:
            x_min = face.rect.left()
            y_min = face.rect.top()
            x_max = face.rect.right()
            y_max = face.rect.bottom()
            bbox.append((x_min, x_max, y_min, y_max))       
    return bbox