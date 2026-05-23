from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2, os, random

# ── 1. Cargar el modelo desde tu ruta exacta ────────────
model = YOLO("runs/train/weights/best.pt")

# ── 2. Métricas sobre validación ────────────────────────
metrics = model.val(data="data.yaml")
print(f"mAP@50:     {metrics.box.map50:.3f}")
print(f"mAP@50-95:  {metrics.box.map:.3f}")
print(f"Precision:  {metrics.box.mp:.3f}")
print(f"Recall:     {metrics.box.mr:.3f}")

# ── 3. Visualizar predicciones en imágenes de val ───────
val_dir = "images/val"
images  = random.sample(os.listdir(val_dir), min(6, len(os.listdir(val_dir))))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

COLORS = {0:(255,80,80), 1:(80,200,80), 2:(80,80,255),
          3:(255,165,0), 4:(180,0,255)}
NAMES  = {0:"name", 1:"id", 2:"age", 3:"date", 4:"time"}

for i, fname in enumerate(images):
    path = os.path.join(val_dir, fname)
    img  = cv2.imread(path)
    results = model(path, verbose=False)[0]

    for box in results.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cls  = int(box.cls[0])
        conf = float(box.conf[0])
        color = COLORS.get(cls, (255,255,255))
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(img, f"{NAMES[cls]} {conf:.2f}",
                    (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[i].set_title(fname[:30], fontsize=8)
    axes[i].axis("off")

plt.suptitle("Predicciones en Validación", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("val_predictions.png", dpi=150)
plt.show()
print("Guardado: val_predictions.png")