from ultralytics import YOLO
MODEL = 'yolov8n.pt'
model = YOLO(MODEL)
model.train(data='coco8.yaml', epochs=50, imgsz=640)
# model(SOURCE)

