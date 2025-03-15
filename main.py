from ultralytics import YOLO
import torch
print(torch.cuda.is_available())

def train_my():
    model = YOLO("best.pt")



    model.train(data='datasets/wood2/data.yaml',epochs=23,device=0,batch=8,workers=8,name="new",plots=True)
    metrics=model.val()
    path=model.export(format="onnx")


def pre():
    model = YOLO('best.pt')
   # results = model.predict("02.jpg", save=True, imgsz=320, conf=0.5, show=True, show_boxes=True,
    #                        visualize=True, agnostic_nms=True)  # return a list of Results objects
    results=model("gg.jpg")
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="result2.jpg")  # save to disk
if __name__ == '__main__':
    pre()