import cv2
from ultralytics import YOLO
import torch

#torch.cuda.set_device(0)

#model = YOLO('runs/detect/train/weights/best.pt')
#model.to('cuda')
model = YOLO('yolov8x.pt')

video_path='test_video.avi'
cap = cv2.VideoCapture(video_path)

ret = True

while ret:
    ret, frame = cap.read()

    #result = model.predict(frame,verbose = True,conf=0.5,iou=0.4,vid_stride=True, retina_masks=False)
    result = model.predict(frame, iou=0.4, conf=0.5, imgsz=128, verbose=False)
    frame_ = result[0].plot()


    cv2.imshow('frame',frame_)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()