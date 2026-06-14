**Dataset de Desidentificación de Radiografías**

**Descripción**
Este dataset ha sido creado para tareas de desidentificación automática de imágenes médicas, concretamente radiografías en formato PNG.
El objetivo principal es detectar información sensible incrustada en las imágenes mediante técnicas de detección de objetos utilizando el formato de anotación YOLO.
Las anotaciones contienen bounding boxes alrededor de texto sensible presente en las radiografías, permitiendo posteriormente aplicar técnicas de anonimización u OCR seguro.

**Tipo de imágenes**
* Radiografías médicas
* Formato de imagen: .png

**Formato de anotaciones**
Las anotaciones siguen el formato estándar YOLO:
<class\_id> <x\_center> <y\_center> <width> <height>
Donde:
* **class\_id:** identificador numérico de la clase
* **x\_center:** coordenada X del centro de la caja (normalizada entre 0 y 1)
* **y\_center:** coordenada Y del centro de la caja (normalizada entre 0 y 1)
* **width:** ancho de la caja (normalizado entre 0 y 1)
* **height:** alto de la caja (normalizado entre 0 y 1)
Ejemplo:
0 0.512 0.183 0.245 0.052

**Clases**

ID	Clase
0	name
1	id
2	age
3	date
4	time

**Estructura del dataset**
.
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├─── Readme
└── data.yaml
images/: contiene las radiografías en formato PNG.
labels/: contiene las anotaciones YOLO (.txt) correspondientes a cada imagen.
Cada imagen tiene un fichero .txt asociado con el mismo nombre.
Ejemplo:
images/train/ed9c0dfc-ea25b576-0f8cc069-df4cdf14-0cd60eb7.png
labels/train/ed9c0dfc-ea25b576-0f8cc069-df4cdf14-0cd60eb7.txt

**División del dataset**
El dataset contiene aproximadamente:
* 400 imágenes en total
* 80% para entrenamiento (train)
* 20% para validación (val)
Distribución:
* Train: \~320 imágenes
* Validation: \~80 imágenes


**Uso previsto**
Este dataset está diseñado para:
* Desidentificación automática de radiografías
* Detección de información sensible en imágenes médicas
* Entrenamiento de modelos
* Investigación en privacidad y anonimización médica


**Consideraciones**
* Las coordenadas están normalizadas siguiendo el estándar YOLO.
* Las imágenes contienen información sensible delimitada mediante bounding boxes.
* El dataset está orientado a tareas de detección de objetos y anonimización automática.
* El conjunto de datos esta formado por imágenes reales y datos sintéticos, permitiendo conservar la privacidad y anonimización de los pacientes.

## Aplicación Web

Incluimos una interfaz gráfica interactiva que permite cargar radiografías, pasarlas por YOLO entrenado y descargar la imagen procesada.

### Características
* **Carga en caché:** El modelo YOLO se aloja en memoria mediante `@st.cache_resource`.
* **Formatos soportados:** `.png`, `.jpg` y `.jpeg`.
* **Tres técnicas posibles:**
  1. **Pixelado:** Reduce la resolución de la región y la escala de nuevo usando interpolación de vecinos cercanos.
  2. **Borroso:** Aplica un filtro Gaussiano destructivo para emborronar la zona.
  3. **Banda Negra:** Sobrescribe las coordenadas detectadas con un bloque opaco.
* **Descarga:** Permite exportar el resultado en formato PNG.

### Librerías

Dependencias necesarias:

```bash
pip install streamlit ultralytics opencv-python numpy Pillow

