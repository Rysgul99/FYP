import torch
from cv_camera import Camera
import cv2
import cvzone
import time
import os
import usb_arm

arm = usb_arm.Arm()

def right_side():

    arm.move(usb_arm.ShoulderDown | usb_arm.ElbowDown | usb_arm.WristDown, 1)
    arm.move(usb_arm.ShoulderDown | usb_arm.ElbowDown, 1.5)
    arm.move(usb_arm.GripsClose, 1.7)
    arm.move(usb_arm.ShoulderUp | usb_arm.ElbowUp | usb_arm.WristUp, 1.5)
    arm.move(usb_arm.ShoulderUp | usb_arm.ElbowUp, 1.5)
    arm.move(usb_arm.BaseClockWise, 6)
    arm.move(usb_arm.GripsOpen, 1.5)
    arm.move(usb_arm.BaseCtrClockWise, 6)
    
def left_side():
    arm.move(usb_arm.ShoulderDown | usb_arm.ElbowDown | usb_arm.WristDown, 1)
    arm.move(usb_arm.ShoulderDown | usb_arm.ElbowDown, 1.5)
    arm.move(usb_arm.GripsClose, 1.7)
    arm.move(usb_arm.ShoulderUp | usb_arm.ElbowUp | usb_arm.WristUp, 1.5)
    arm.move(usb_arm.ShoulderUp | usb_arm.ElbowUp, 1.5)
    arm.move(usb_arm.BaseCtrClockWise, 3)
    arm.move(usb_arm.GripsOpen, 1.5)
    arm.move(usb_arm.BaseClockWise, 3)

def backward():
    arm.move(usb_arm.ShouderUp, 0.7)
    arm.move(usb_arm.ShoulderDown | usb_arm.ElbowDown | usb_arm.WristDown, 1)
    arm.move(usb_arm.ShoulderDown | usb_arm.ElbowDown, 1.5)
    arm.move(usb_arm.GripsClose, 1.7)
    arm.move(usb_arm.ShoulderUp | usb_arm.ElbowUp | usb_arm.WristUp, 1.5)
    arm.move(usb_arm.ShoulderUp | usb_arm.ElbowUp, 1.5)
    arm.move(usb_arm.BaseCtrClockWise, 7)
    arm.move(usb_arm.GripsOpen, 1.5)
    arm.move(usb_arm.BaseClockWise, 7)

cam = Camera(2)
model = torch.hub.load(os.getcwd(), 'custom', source='local', path = 'last.model.pt', force_reload = True)

while True:
    frame = cam.get_frame()
    img = frame

    # Inference
    results = model(img)

    # Display image with bounding boxes
    cv2.imshow('image', results.render()[0])

    # Filter detections with confidence > 70%
    high_conf_detections = [det for det in results.xyxy[0] if det[4] > 0.70]

    # Check if 'Plastic' was detected with high confidence
    if any(results.names[int(det[5])] == 'Plastic' for det in high_conf_detections):
        right_side()
        print('Plastic found and sorted.')
        continue

    # Check if 'Cardboard' was detected with high confidence
    if any(results.names[int(det[5])] == 'Cardboard' for det in high_conf_detections):
        left_side()
        print('Cardboard found and sorted.')
        continue

    # Check if 'Paper' was detected with high confidence
    if any(results.names[int(det[5])] == 'Paper' for det in high_conf_detections):
        backward()
        print('Paper found and sorted.')
        continue

    # Check if 'q' key is pressed to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
           