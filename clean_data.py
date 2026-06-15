import os

# Ruta base: la carpeta donde está este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def renombrar_imagenes(split):
    img_dir = os.path.join(BASE_DIR, 'images', split)
    renamed = 0
    for f in os.listdir(img_dir):
        if '_annotated' in f:
            ext = os.path.splitext(f)[1]
            stem = os.path.splitext(f)[0].replace('_annotated', '')
            os.rename(
                os.path.join(img_dir, f),
                os.path.join(img_dir, stem + ext)
            )
            renamed += 1
    print(f"✅ images/{split}: {renamed} archivos renombrados")

def verificar_coincidencias(split):
    img_dir = os.path.join(BASE_DIR, 'images', split)
    lbl_dir = os.path.join(BASE_DIR, 'labels', split)
    imgs = {os.path.splitext(f)[0] for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))}
    lbls = {os.path.splitext(f)[0] for f in os.listdir(lbl_dir)
            if f.endswith('.txt')}
    match = len(imgs & lbls)
    total = len(imgs)
    simbolo = '✅' if match == total else '❌'
    print(f"{simbolo} {split}: {match}/{total} coincidencias imagen-label")

if __name__ == '__main__':
    for split in ['train', 'val']:
        renombrar_imagenes(split)
        verificar_coincidencias(split)