from ultralytics import YOLO
import cv2
import os

# ── 1. Cargar el modelo desde tu ruta exacta ────────────
model = YOLO("runs/train/weights/best.pt")


def anonymize_image(image_path, output_path="radiografia_anonima.png", method="pixelate"):
    if not os.path.exists(image_path):
        print(f"Error: No se encuentra {image_path}")
        return

    # ── 2. Leer imagen y predecir ───────────────────────
    img = cv2.imread(image_path)
    results = model(img, verbose=False)[0]

    # ── 3. Aplicar censura caja a caja ──────────────────
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = img[y1:y2, x1:x2]

        if method == "blur":
            img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 30)
        elif method == "black":
            img[y1:y2, x1:x2] = 0
        elif method == "pixelate":
            small = cv2.resize(roi, (10, 10), interpolation=cv2.INTER_LINEAR)
            img[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)

    # ── 4. Guardar resultado ────────────────────────────
    cv2.imwrite(output_path, img)
    print(f"Imagen anonimizada guardada en: {output_path}")


if __name__ == "__main__":
    # Sustituye este nombre por una imagen real que tengas en tu carpeta val
    imagen_prueba = "images/val/ed9c0dfc-ea25b576-0f8cc069-df4cdf14-0cd60eb7.png"
    anonymize_image(imagen_prueba)