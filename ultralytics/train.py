from ultralytics import YOLO
# Load a model
model = YOLO("yolov9s.yaml")
model.load("yolov9s.pt")  # build from YAML and transfer weights

#Train the model
results = model.train(data="coco.yaml", batch=32, epochs=5000, imgsz=640)