# Old Book Illustration Generator

**Autor:** Raúl Sánchez Serrano — rulox.github@gmail.com

- **Repositorio GitHub:** https://github.com/RaulRX/vintagebooks-illustrator-generator
- **Repositorio Hugging Face (modelo):** https://huggingface.co/RuloxSS/vintage-bookshop-practice-rss-unet

## Descripción

Proyecto de fine-tuning de **Stable Diffusion v1.4** (`CompVis/stable-diffusion-v1-4`)
sobre el dataset de ilustraciones de libros antiguos
[`gigant/oldbookillustrations`](https://huggingface.co/datasets/gigant/oldbookillustrations)
(columnas `1600px` como imagen y `info_alt` como descripción). Se congelan el VAE y el
text encoder, y se entrena únicamente la **UNet**. La UNet resultante se sube a
Hugging Face y se reutiliza después para generar imágenes a partir de un prompt.

## Estructura del proyecto

```
src/
  main.py                Flujo end-to-end: genera con el modelo base, fine-tunea, regenera y sube la UNet
  loader.py              Carga del dataset y de los componentes del modelo (cache local o Hub)
  trainer.py             Bucle de fine-tuning de la UNet (forward diffusion + MSE sobre el ruido)
  image_transformer.py   Preprocesado de imágenes: crop/resize/normalización a tensores
  repository.py          Cache local del modelo base y subida de la UNet a Hugging Face
  test_finetunned.py     Descarga la UNet fine-tuneada y genera una imagen desde un prompt
oldbookillustration_generator.ipynb   Notebook con el fine-tuning y guardado en Hugging Face
requirements.txt        Dependencias del proyecto
Makefile                Targets: run, test-finetuned, kernel, notebook, freeze, clean
env-template            Plantilla de variables de entorno (.env)
```

## Usos del proyecto

- **Fine-tuning completo + comparación + subida a Hugging Face:**
  `make run` (equivalente a `python src/main.py`). Pide un prompt, genera una imagen
  con el modelo base, entrena la UNet, genera de nuevo con el mismo prompt usando la
  UNet fine-tuneada y sube el resultado a Hugging Face.
- **Generar imágenes con el modelo ya fine-tuneado:**
  `make test-finetuned` (equivalente a `python src/test_finetunned.py`). Descarga la
  UNet desde Hugging Face y genera una imagen a partir de un prompt.
- **Notebook:** `make notebook`. Ejecuta únicamente el fine-tuning y el guardado del
  modelo en Hugging Face.

Los targets anteriores del `Makefile` son un atajo que crea el entorno virtual,
instala las dependencias y lanza el script correspondiente. Ambos scripts también se
pueden ejecutar directamente, sin pasar por el `Makefile`, siempre que el entorno ya
tenga las dependencias de `requirements.txt` instaladas y exista un `.env` (a partir de
`env-template`) en la raíz del proyecto:

- **`src/main.py`** (equivalente a `make run`): ejecuta el flujo completo. Pide por
  consola un prompt, genera una imagen con el modelo base, entrena la UNet, regenera la
  imagen con el mismo prompt usando la UNet fine-tuneada y sube el resultado a Hugging
  Face.

  ```bash
  library-ve/Scripts/python src/main.py
  ```

- **`src/test_finetunned.py`** (equivalente a `make test-finetuned`): pide por consola
  un prompt, descarga la UNet fine-tuneada desde Hugging Face y genera una imagen a
  partir de ese prompt.

  ```bash
  library-ve/Scripts/python src/test_finetunned.py
  ```

### Variables de entorno (`env-template`)

| Variable | Descripción |
|---|---|
| `BASE_MODEL_NAME` | Modelo base de Stable Diffusion |
| `FTUNNING_MODEL_NAME` | Nombre del modelo fine-tuneado |
| `LEARNING_RATE` | Learning rate del optimizador |
| `DATASET_NAME` | Dataset de Hugging Face usado para entrenar |
| `DATASE_COLUMNS` | Columnas del dataset: imagen, descripción |
| `MAX_TRAIN_SAMPLES` | Número de imágenes de entrenamiento |
| `IMAGE_RESOLUTION` | Resolución de las imágenes de entrenamiento |
| `IMAGE_CROP_METHOD` | Método de recorte (`center` o `resize_crop`) |
| `TRAIN_EPOCHS` | Número de epochs |
| `TRAIN_BATCH_SIZE` | Tamaño de batch |
| `HGGF_USERNAME` | Usuario de Hugging Face |
| `HGGF_TOKEN` | Token de Hugging Face (sensible, no versionar) |

## Explicación técnica

Se cargan por separado los componentes de Stable Diffusion v1.4: tokenizer, scheduler
(`DDPMScheduler`), text encoder (CLIP), VAE y UNet. El VAE y el text encoder se
congelan (`requires_grad=False`); solo la UNet se entrena.

En cada paso de entrenamiento: el VAE codifica la imagen a su espacio latente, se le
añade ruido aleatorio en un timestep también aleatorio (forward diffusion), y la UNet
predice ese ruido a partir del latente ruidoso, el timestep y el embedding de texto del
prompt. El error se calcula como MSE entre el ruido real y el predicho, y se optimiza
con `AdamW` usando `accelerate`.

Como las imágenes del dataset no son cuadradas, se aplica un crop (`center` o
`resize_crop`) seguido de un resize a una resolución cuadrada antes de convertirlas a
tensores normalizados.

## Limitaciones y ajuste de parámetros

El entrenamiento se ejecuta en local sobre un equipo con **CPU Intel i5-6500HQ (4
núcleos), GPU de 2GB y 8GB de RAM**. Esta GPU no tiene memoria suficiente para
entrenar la UNet de Stable Diffusion, por lo que el fine-tuning se fuerza a CPU
(`torch.set_num_threads(4)`).

Con 8GB de RAM, cargar y entrenar la UNet completa es muy ajustado: si la configuración
es demasiado exigente, Windows puede matar el proceso de forma abrupta con el código de
salida `-1073741810`, sin traceback de Python, debido al agotamiento de memoria. Por la
misma razón se mantiene `num_workers=0` en el `DataLoader`, ya que en Windows el uso de
varios workers en multiprocessing puede provocar el mismo tipo de fallo.

**Valores usados** para permitir una ejecución local estable (no orientados a calidad
final del modelo, sino a validar el pipeline):

- `MAX_TRAIN_SAMPLES=20`
- `TRAIN_BATCH_SIZE=1`
- `IMAGE_RESOLUTION=64`
- `IMAGE_CROP_METHOD=resize_crop`
- `TRAIN_EPOCHS=2`
- `LEARNING_RATE=1e-5`

**Futuras mejoras:** ejecutar el entrenamiento en una infraestructura con más recursos
(VPS, Google Cloud u otra plataforma con GPU dedicada), lo que permitiría aumentar el
número de imágenes de entrenamiento, subir la resolución hasta el límite del modelo
(512×512) y usar batch sizes y epochs mayores.

## Líneas de experimentación pendientes

- Comparar métodos de crop: `center` vs `resize_crop`.
- Comparar imágenes en color, en escala de grises y sin modificar (originales del
  dataset).
- Aumentar la resolución de entrenamiento hasta el límite del modelo (512×512).
- Probar distintos valores de `TRAIN_BATCH_SIZE` y `MAX_TRAIN_SAMPLES`.

## Material adjunto

En la carpeta `images/` se incluyen ejemplos de imágenes generadas:
- `image_before_finetunning.PNG`: imagen generada con el modelo base antes del
  fine-tuning, con el prompt *"a farmer taking care of a sheep"*.
- `image_after_finetunning.PNG`: imagen generada con el mismo prompt
  (*"a farmer taking care of a sheep"*) tras el fine-tuning.
- `image_finetunned.PNG`: imagen generada con el modelo fine-tuneado descargado desde
  Hugging Face, con el prompt *"A pirate digging out a treasure"*.
