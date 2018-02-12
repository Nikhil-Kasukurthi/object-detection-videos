import math
import cv2

cap = cv2.VideoCapture('240p1.mp4')
framerate = cap.get(5)
while(cap.isOpened()):
    # Capture frame-by-frame
    frame_id = cap.get(1)
    ret, frame = cap.read()
    
    
    # Display the resulting frame
    if(frame_id % math.floor(framerate) == 0):
    	cv2.imshow('frame',frame)
    	print(frame_id)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()