from ultralytics import YOLO

model = YOLO("yolov8s.pt")
results = model.train(
    data="data.yaml",
    epochs=100,
    patience=20,
    imgsz=1024,
    batch=4,

    freeze=10,
    mosaic=0.0,
    mixup=0.0,
    degrees=10,
    scale=0.5,
    fliplr=0.0,
    flipud=0.0,

    project="runs",
    name="train",
    exist_ok=True
)