# oldbookillustration-image-generator

prompt used: An illustration about a farmer taking care of a sheep

prompt with finetunned model: a pirate digging up a treasure

- Explicar las configuraciones adicionales debido a las limitaciones de memoria, CPU y GPU del equipo. Añadir como futuras mejoras el usar VPS, Google Cloud u otra plataforma que nos permita obtener una infraestructura mejor. Además, indicar los valores de batch size, epochs, numero de imagenes de entrenamiento y tamaño de las imagenes (valores dados debido a las limitaciones del equipo) 

- Indicar que hayd dos ficheros principales: uno para ejecutar el fine tunning del modelo original y guardarlo en el repo de hugging face. Además, se mostrará una imagen antes de fine tunnear y después para verificar la diferencia. Otro script para descargar el modelo fine tuneado, ejecutar el modelo y generar imagenes a partir de un prompt

- Explicar la estructura del projecto, asi como presentar una breve descripción del mismo asi como presentar al desarrollador con su nombre, cuenta de correo y repositorio tanto de github como de hugging faces

- INdicar que se debe investigar sobre las siguientes diferencias:
    * usar otros tipos de recorte (center). Se usa resize_crop
    * Comparativa entre convertir imagenes con colores, grises, dejarlas originales como el dataset.
    * USar un resize mayor hasta el limite que usa el modelo, que es 512x512
    * Usar diferentes valores en batchsize, max_training images 

- Indicar que se adjuntan imagenes:
    - Imagenes antes y después de ejecutar el fine tunning
    - Imagen generada con el modelo finetuneado descargado de hugging face

- Además, se añade notebook con la ejecución unicamente del fine tunning y del guardado del modelo fine tuneado en el repositorio del hugging face