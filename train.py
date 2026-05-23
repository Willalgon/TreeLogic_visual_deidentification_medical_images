from ultralytics import YOLO

# 1. Usamos la versión 's' (small)
model = YOLO("yolov8s.pt")

# 2. Configuración optimizada
results = model.train(
    data="data.yaml",
    epochs=100,
    patience=20,
    imgsz=1024,
    batch=4,

    # Transfer Learning y aumentos
    freeze=10,
    mosaic=0.0,
    mixup=0.0,
    degrees=10,
    scale=0.5,
    flips=0.0,

    # RUTAS: Forzamos a que guarde en runs/train (tu carpeta)
    project="runs",
    name="train",
    exist_ok=True  # Sobrescribe la carpeta actual en vez de crear "train2"
)