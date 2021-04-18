
import cv2
#ssdsd
import numpy as np
import time


#uuui

cap =cv2.VideoCapture('rtsp://admin:Edabk4321@192.168.0.98:554/Streaming/channels/1/')
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)

# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed") 

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
print("Resolution ({}, {})".format(frame_width, frame_height))
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy'+str(round(time.time(), 0))+'.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

while(True):
  ret, frame = cap.read()

  if ret == True: 
    
    # Write the frame into the file 'output.avi'
    out.write(frame)

    # Display the resulting frame   
    frame = cv2.resize(frame, (640, 480)) 
    cv2.imshow('frame',frame)

    # Press Q on keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # Break the loop
  else:
    break  

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows() 
