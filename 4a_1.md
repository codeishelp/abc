# Explicación del Cuaderno: Evaluación de Clasificación de Frutas

Este cuaderno te guía a través del proceso de entrenar un modelo de aprendizaje profundo para clasificar frutas como frescas o podridas. El objetivo es alcanzar una precisión de validación de al menos el 92%.

## 1. Introducción y Configuración Inicial

El cuaderno comienza con una introducción al problema y la importación de las bibliotecas necesarias.

*   **Celda de Markdown (Introducción):**
    *   Describe el objetivo: entrenar un modelo para reconocer frutas frescas y podridas con una precisión de validación del 92%.
    *   Sugiere usar una combinación de aprendizaje por transferencia, aumento de datos y ajuste fino.

*   **Celda de Código (Importaciones y Configuración del Dispositivo):**
    ```python
    # filepath: d:\UNJBG\NOVEMO SEMESTRE\SISTEMAS EXPERTOS\lab-sistemas-expertos\moedlos\07_assessment.ipynb
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms.v2 as transforms
    import torchvision.io as tv_io

    import glob
    from PIL import Image

    import utils

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.is_available()
    ```
    *   **Explicación:**
        *   `torch`, `torch.nn`, `torch.optim`, `torch.utils.data`: Módulos fundamentales de PyTorch para construir redes neuronales, optimizadores y cargar datos.
        *   `torchvision.transforms.v2`, `torchvision.io`: Para transformaciones de imágenes y lectura de imágenes.
        *   `glob`: Para encontrar archivos que coincidan con un patrón (útil para cargar imágenes).
        *   `PIL.Image`: Aunque importado, no se usa directamente en este fragmento, pero es común para manipulación de imágenes.
        *   `utils`: Un archivo local (`utils.py`) que probablemente contiene funciones auxiliares como `train` y `validate`.
        *   `device`: Configura el dispositivo para el entrenamiento. Usará una GPU CUDA si está disponible (`"cuda"`), de lo contrario usará la CPU (`"cpu"`).
    *   **Posibles Modificaciones:**
        *   Si necesitas bibliotecas adicionales, agrégalas aquí.
        *   Si el archivo `utils.py` tiene un nombre diferente o está en otra ubicación, deberás ajustar la importación.

## 2. El Conjunto de Datos (Dataset)

Esta sección describe el conjunto de datos utilizado para el entrenamiento.

*   **Celda de Markdown (Descripción del Dataset):**
    *   El dataset proviene de Kaggle y contiene imágenes de frutas frescas y podridas.
    *   Ubicación: `data/fruits`.
    *   6 categorías: manzanas frescas, naranjas frescas, plátanos frescos, manzanas podridas, naranjas podridas, plátanos podridos.
    *   Esto implica que la capa de salida del modelo necesitará 6 neuronas.
    *   Se debe usar `categorical_crossentropy` (o `CrossEntropyLoss` en PyTorch) como función de pérdida debido a las múltiples categorías.

## 3. Cargar el Modelo Base de ImageNet

Se utiliza aprendizaje por transferencia, comenzando con un modelo preentrenado en ImageNet.

*   **Celda de Markdown (Instrucciones para Cargar Modelo):**
    *   Se recomienda usar un modelo preentrenado en ImageNet (VGG16 en este caso).
    *   Las imágenes son a color (3 canales: rojo, verde, azul).
    *   Se referencia el cuaderno `05b_presidential_doggy_door.ipynb` como guía.

*   **Celda de Código (Carga de VGG16):**
    ```python
    # filepath: d:\UNJBG\NOVEMO SEMESTRE\SISTEMAS EXPERTOS\lab-sistemas-expertos\moedlos\07_assessment.ipynb
    from torchvision.models import vgg16
    from torchvision.models import VGG16_Weights

    weights = VGG16_Weights.IMAGENET1K_V1
    vgg_model = vgg16(weights=weights)
    ```
    *   **Explicación:**
        *   `from torchvision.models import vgg16, VGG16_Weights`: Importa el modelo VGG16 y sus pesos preentrenados.
        *   `weights = VGG16_Weights.IMAGENET1K_V1`: Especifica que se usarán los pesos preentrenados en el dataset ImageNet (versión 1).
        *   `vgg_model = vgg16(weights=weights)`: Crea una instancia del modelo VGG16 con los pesos cargados.
    *   **Posibles Modificaciones:**
        *   Puedes elegir otro modelo preentrenado de `torchvision.models` (ej. `resnet50`, `efficientnet_b0`) y sus pesos correspondientes. Si cambias el modelo, la forma en que accedes y modificas las capas (ver más abajo) podría necesitar ajustes.

## 4. Congelar el Modelo Base

Para preservar el conocimiento aprendido de ImageNet, las capas del modelo base se congelan inicialmente.

*   **Celda de Markdown (Instrucciones para Congelar):**
    *   Se congela el modelo base para evitar que los pesos aprendidos de ImageNet se destruyan durante el entrenamiento inicial con el nuevo dataset.
    *   Nuevamente, se referencia el cuaderno `05b`.

*   **Celda de Código (Congelación de Capas):**
    ```python
    # filepath: d:\UNJBG\NOVEMO SEMESTRE\SISTEMAS EXPERTOS\lab-sistemas-expertos\moedlos\07_assessment.ipynb
    # Freeze base model
    vgg_model.requires_grad_(False)
    next(iter(vgg_model.parameters())).requires_grad
    ```
    *   **Explicación:**
        *   `vgg_model.requires_grad_(False)`: Itera sobre todos los parámetros (pesos y sesgos) del `vgg_model` y establece su atributo `requires_grad` a `False`. Esto significa que no se calcularán gradientes para estos parámetros durante la retropropagación, y por lo tanto, no se actualizarán.
        *   `next(iter(vgg_model.parameters())).requires_grad`: Es una forma de verificar si el primer parámetro del modelo realmente tiene `requires_grad` como `False`.
    *   **Posibles Modificaciones:**
        *   Si decides no congelar el modelo base desde el principio (lo cual no es recomendable para el aprendizaje por transferencia inicial), puedes omitir esta celda.
        *   Más adelante, para el "fine-tuning", se descongelarán algunas o todas estas capas.

## 5. Añadir Capas al Modelo

Se añaden nuevas capas al modelo base para adaptarlo a la tarea específica de clasificación de frutas.

*   **Celda de Markdown (Instrucciones para Añadir Capas):**
    *   Se deben añadir capas personalizadas al modelo preentrenado.
    *   La última capa densa (o lineal) debe tener el número correcto de neuronas (6 para las 6 categorías de frutas).
    *   Se explica que se pueden seleccionar partes específicas del clasificador de VGG16.

*   **Celda de Código (Selección de Capas de VGG y Definición del Nuevo Modelo):**
    ```python
    # filepath: d:\UNJBG\NOVEMO SEMESTRE\SISTEMAS EXPERTOS\lab-sistemas-expertos\moedlos\07_assessment.ipynb
    vgg_model.classifier[0:3]
    ```
    *   **Explicación:** Esta celda simplemente muestra las primeras tres capas del clasificador original de VGG16. VGG16 tiene una sección de `features` (capas convolucionales) y una sección de `classifier` (capas densas).

    ```python
    # filepath: d:\UNJBG\NOVEMO SEMESTRE\SISTEMAS EXPERTOS\lab-sistemas-expertos\moedlos\07_assessment.ipynb
    N_CLASSES = 6

    my_model = nn.Sequential(
        vgg_model.features,
        vgg_model.avgpool,
        nn.Flatten(),
        vgg_model.classifier[0:3],
        nn.Linear(4096, 500),
        nn.ReLU(),
        nn.Linear(500, N_CLASSES)
    )
    my_model
    ```
    *   **Explicación:**
        *   `N_CLASSES = 6`: Define el número de clases de salida.
        *   `my_model = nn.Sequential(...)`: Crea un nuevo modelo secuencial.
            *   `vgg_model.features`: Se utilizan todas las capas convolucionales (extractoras de características) del VGG16 preentrenado.
            *   `vgg_model.avgpool`: La capa de average pooling de VGG16.
            *   `nn.Flatten()`: Aplana la salida para que pueda ser procesada por capas lineales (densas).
            *   `vgg_model.classifier[0:3]`: Se reutilizan las primeras tres capas del clasificador original de VGG16. Estas son capas lineales y ReLU. La tercera capa (`vgg_model.classifier[2]`) es una `ReLU`. La capa `vgg_model.classifier[0]` es `Linear(in_features=25088, out_features=4096, bias=True)`.
            *   `nn.Linear(4096, 500)`: Una nueva capa lineal que toma las 4096 características de la capa anterior de VGG y las reduce a 500.
            *   `nn.ReLU()`: Una función de activación ReLU.
            *   `nn.Linear(500, N_CLASSES)`: La capa de salida final, con 500 neuronas de entrada y `N_CLASSES` (6) neuronas de salida, una para cada categoría de fruta.
    *   **Posibles Modificaciones:**
        *   `N_CLASSES`: Debe coincidir con el número de categorías en tu dataset.
        *   La arquitectura de las capas añadidas (`nn.Linear(4096, 500)`, `nn.ReLU()`, etc.) es experimental. Puedes cambiar el número de capas, el número de neuronas en cada capa, o las funciones de activación. Por ejemplo, podrías añadir más capas lineales, capas de Dropout para regularización, o cambiar el número de neuronas (ej. `nn.Linear(4096, 1024)`).
        *   Si usaste un modelo base diferente a VGG16, la forma de acceder a sus `features` o `classifier` y el tamaño de las características de salida de la parte convolucional (que aquí es 4096 después de `vgg_model.classifier[0:3]`) cambiarán. Tendrás que ajustar `in_features` de la primera capa `nn.Linear` que añadas.

## 6. Compilar el Modelo

Se define la función de pérdida y el optimizador para el entrenamiento.

*   **Celda de Markdown (Instrucciones para Compilar):**
    *   Se pregunta qué función de pérdida usar para 6 clases (la respuesta es entropía cruzada categórica).

*   **Celda de Código (Definición de Pérdida y Optimizador):**
    ```python
    # filepath: d:\UNJBG\NOVEMO SEMESTRE\SISTEMAS EXPERTOS\lab-sistemas-expertos\moedlos\07_assessment.ipynb
    loss_function = nn.CrossEntropyLoss()
    optimizer = Adam(my_model.parameters())
    my_model = torch.compile(my_model.to(device))
    ```
    *   **Explicación:**
        *   `loss_function = nn.CrossEntropyLoss()`: Define la función de pérdida. `CrossEntropyLoss` es adecuada para problemas de clasificación multiclase. Combina `LogSoftmax` y `NLLLoss` en una sola clase.
        *   `optimizer = Adam(my_model.parameters())`: Define el optimizador Adam. Se le pasan los parámetros de `my_model` que deben ser actualizados. Como las capas base de `vgg_model` están congeladas, solo los parámetros de las nuevas capas añadidas (y las capas no congeladas del clasificador de VGG) tendrán `requires_grad=True` y serán optimizados.
        *   `my_model = torch.compile(my_model.to(device))`:
            *   `.to(device)`: Mueve el modelo al dispositivo configurado (GPU o CPU).
            *   `torch.compile()`: (Introducido en PyTorch 2.0) Intenta optimizar el modelo para una ejecución más rápida. Puede que no esté disponible o no ofrezca ventajas en todas las configuraciones.
    *   **Posibles Modificaciones:**
        *   `loss_function`: Podrías usar otras funciones de pérdida si la tarea lo requiere, pero `CrossEntropyLoss` es estándar para esto.
        *   `optimizer`: Puedes probar otros optimizadores como `SGD`, `RMSprop`, etc., o ajustar los hiperparámetros de Adam (ej. `lr=0.001`, `betas=(0.9, 0.999)`). La tasa de aprendizaje (`lr`) es un hiperparámetro crucial.
        *   Si `torch.compile` causa problemas o no está disponible, puedes simplemente usar `my_model = my_model.to(device)`.

## 7. Transformaciones de Datos

Se definen las transformaciones para preprocesar las imágenes de entrada y para el aumento de datos.

*   **Celda de Markdown (Transformaciones de Preprocesamiento):**
    *   Se usarán las transformaciones incluidas con los pesos de VGG16 para el preprocesamiento.

*   **Celda de Código (Transformaciones de Preprocesamiento de VGG16):**
    ```python
    # filepath: d:\UNJBG\NOVEMO SEMESTRE\SISTEMAS EXPERTOS\lab-sistemas-expertos\moedlos\07_assessment.ipynb
    pre_trans = weights.transforms()
    ```
    *   **Explicación:**
        *   `pre_trans = weights.transforms()`: Obtiene la secuencia de transformaciones recomendadas para el modelo VGG16 con los pesos `IMAGENET1K_V1`. Esto típicamente incluye redimensionar la imagen a 224x224, convertirla a tensor y normalizarla con la media y desviación estándar de ImageNet.
    *   **Posibles Modificaciones:**
        *   Si usas un modelo base diferente, deberías usar las transformaciones recomendadas para ese modelo.

*   **Celda de Markdown (Aumento de Datos):**
    *   Se anima a añadir transformaciones aleatorias para el aumento de datos.
    *   Se referencian los cuadernos `04a` y `05b` y la documentación de `TorchVision Transforms`.
    *   Se advierte no hacer el aumento de datos demasiado extremo.

*   **Celda de Código (Transformaciones Aleatorias para Aumento):**
    ```python
    # filepath: d:\UNJBG\NOVEMO SEMESTRE\SISTEMAS EXPERTOS\lab-sistemas-expertos\moedlos\07_assessment.ipynb
    IMG_WIDTH, IMG_HEIGHT = (224, 224)

    random_trans = transforms.Compose([
        transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), antialias=True),
        transforms.RandomHorizontalFlip(p=0.5)
    ])
    ```
    *   **Explicación:**
        *   `IMG_WIDTH, IMG_HEIGHT = (224, 224)`: Define el tamaño de imagen esperado por VGG16.
        *   `random_trans = transforms.Compose([...])`: Crea una secuencia de transformaciones de aumento de datos.
            *   `transforms.RandomResizedCrop((IMG_WIDTH, IMG_HEIGHT), antialias=True)`: Recorta una porción aleatoria de la imagen y la redimensiona al tamaño especificado.
            *   `transforms.RandomHorizontalFlip(p=0.5)`: Invierte horizontalmente la imagen con una probabilidad del 50%.
    *   **Posibles Modificaciones:**
        *   Puedes añadir más transformaciones de `torchvision.transforms.v2` como `RandomRotation`, `ColorJitter`, etc.
        *   Los parámetros de cada transformación (ej. `degrees` para `RandomRotation`, `probability` `p` para `RandomHorizontalFlip`) pueden ajustarse.
        *   Es importante que estas transformaciones aleatorias solo se apliquen al conjunto de entrenamiento. El conjunto de validación/prueba debe usar solo las transformaciones de preprocesamiento deterministas. (Nota: En el código de entrenamiento `utils.train`, `random_trans` se aplica, lo que es correcto para el aumento en el entrenamiento).

## 8. Cargar el Conjunto de Datos

Se define una clase `Dataset` personalizada y se crean `DataLoader`s para los conjuntos de entrenamiento y validación.

*   **Celda de Markdown (Instrucciones para Cargar Datos):**
    *   Se indica que es momento de cargar los datos de entrenamiento y validación.

*   **Celda de Código (Clase `MyDataset`):**
    ```python
    # filepath: d:\UNJBG\NOVEMO SEMESTRE\SISTEMAS EXPERTOS\lab-sistemas-expertos\moedlos\07_assessment.ipynb
    DATA_LABELS = ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges"] 
        
    class MyDataset(Dataset):
        def __init__(self, data_dir):
            self.imgs = []
            self.labels = []
            
            for l_idx, label in enumerate(DATA_LABELS):
                data_paths = glob.glob(data_dir + label + '/*.png', recursive=True) # Asume subcarpetas con nombres de etiquetas
                for path in data_paths:
                    img = tv_io.read_image(path, tv_io.ImageReadMode.RGB)
                    self.imgs.append(pre_trans(img).to(device)) # Aplica pre_trans aquí
                    self.labels.append(torch.tensor(l_idx).to(device))


        def __getitem__(self, idx):
            img = self.imgs[idx]
            label = self.labels[idx]
            # Las transformaciones aleatorias (random_trans) se aplican en el bucle de entrenamiento, no aquí.
            return img, label

        def __len__(self):
            return len(self.imgs)
    ```
    *   **Explicación:**
        *   `DATA_LABELS`: Una lista de los nombres de las carpetas (y etiquetas) para cada clase. El orden es importante ya que `enumerate` asignará índices (0, 1, 2...) basados en este orden.
        *   `MyDataset(Dataset)`: Define una clase que hereda de `torch.utils.data.Dataset`.
        *   `__init__(self, data_dir)`:
            *   Inicializa listas `self.imgs` y `self.labels`.
            *   Itera sobre `DATA_LABELS`. Para cada etiqueta, usa `glob.glob` para encontrar todas las imágenes `.png` en la subcarpeta correspondiente (ej. `data_dir + "freshapples" + '/*.png'`).
            *   `tv_io.read_image(path, tv_io.ImageReadMode.RGB)`: Lee cada imagen como un tensor RGB.
            *   `pre_trans(img).to(device)`: Aplica las transformaciones de preprocesamiento (`pre_trans`) a la imagen y la mueve al `device`. **Importante:** Almacenar todas las imágenes preprocesadas en memoria (`self.imgs`) puede ser problemático para datasets muy grandes. Una alternativa sería cargar y transformar imágenes bajo demanda en `__getitem__`.
            *   `self.labels.append(torch.tensor(l_idx).to(device))`: Almacena el índice de la etiqueta como un tensor en el `device`.
        *   `__getitem__(self, idx)`: Devuelve la imagen y la etiqueta en el índice `idx`.
        *   `__len__(self)`: Devuelve el número total de muestras en el dataset.
    *   **Posibles Modificaciones:**
        *   `DATA_LABELS`: Debe coincidir exactamente con los nombres de las subcarpetas de tu dataset y el orden deseado para las etiquetas.
        *   La estructura de carpetas (`data_dir + label + '/*.png'`) debe coincidir con cómo están organizados tus datos. Si usas otros formatos de imagen (ej. `.jpg`), cambia `/*.png` a `/*.jpg`.
        *   Si el dataset es muy grande, considera cargar y transformar imágenes en `__getitem__` en lugar de `__init__` para ahorrar memoria. En ese caso, `pre_trans` se aplicaría en `__getitem__`.
        *   Las transformaciones aleatorias (`random_trans`) no se aplican aquí, sino en el bucle de entrenamiento (pasadas a `utils.train`), lo cual es correcto.

*   **Celda de Markdown (Instrucciones para DataLoader):**
    *   Se pide seleccionar el tamaño del lote (`n`) y si se deben barajar (`shuffle`) los datos.

*   **Celda de Código (Creación de DataLoaders):**
    ```python
    # filepath: d:\UNJBG\NOVEMO SEMESTRE\SISTEMAS EXPERTOS\lab-sistemas-expertos\moedlos\07_assessment.ipynb
    n = 16 # Tamaño del lote (batch size)

    train_path = "data/fruits/train/"
    train_data = MyDataset(train_path)
    train_loader = DataLoader(train_data, batch_size=n, shuffle=True)
    train_N = len(train_loader.dataset)

    valid_path = "data/fruits/valid/"
    valid_data = MyDataset(valid_path)
    valid_loader = DataLoader(valid_data, batch_size=n, shuffle=False)
    valid_N = len(valid_loader.dataset)
    ```
    *   **Explicación:**
        *   `n = 16`: Define el tamaño del lote (batch size).
        *   `train_path`, `valid_path`: Rutas a los directorios de datos de entrenamiento y validación.
        *   `train_data = MyDataset(train_path)`: Crea una instancia de `MyDataset` para los datos de entrenamiento.
        *   `train_loader = DataLoader(train_data, batch_size=n, shuffle=True)`: Crea un `DataLoader` para el entrenamiento.
            *   `batch_size=n`: Especifica cuántas muestras cargar por lote.
            *   `shuffle=True`: Baraja los datos de entrenamiento en cada época, lo cual es bueno para el entrenamiento.
        *   `train_N`: Almacena el número total de muestras de entrenamiento.
        *   Se repite un proceso similar para los datos de validación (`valid_data`, `valid_loader`), pero con `shuffle=False` porque el orden no importa para la validación y asegura consistencia.
    *   **Posibles Modificaciones:**
        *   `n` (batch size): Puedes experimentar con diferentes tamaños de lote. Un tamaño de lote más grande puede acelerar el entrenamiento pero requiere más memoria y a veces puede llevar a una peor generalización. Tamaños comunes son 16, 32, 64, 128.
        *   `train_path`, `valid_path`: Asegúrate de que estas rutas apunten a las ubicaciones correctas de tus datos.

## 9. Entrenar el Modelo (Fase Inicial)

Se entrena el modelo con las capas base congeladas.

*   **Celda de Markdown (Instrucciones de Entrenamiento):**
    *   Las funciones `train` y `validate` están en `utils.py`.
    *   Se sugiere reejecutar la celda o cambiar el número de `epochs` si es necesario.

*   **Celda de Código (Bucle de Entrenamiento Inicial):**
    ```python
    # filepath: d:\UNJBG\NOVEMO SEMESTRE\SISTEMAS EXPERTOS\lab-sistemas-expertos\moedlos\07_assessment.ipynb
    epochs = 10

    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch))
        utils.train(my_model, train_loader, train_N, random_trans, optimizer, loss_function)
        utils.validate(my_model, valid_loader, valid_N, loss_function)
    ```
    *   **Explicación:**
        *   `epochs = 10`: Define el número de veces que se iterará sobre todo el conjunto de datos de entrenamiento.
        *   El bucle `for epoch in range(epochs)`: Itera a través de las épocas.
        *   `utils.train(...)`: Llama a la función `train` (definida en `utils.py`). Esta función probablemente contiene el bucle de entrenamiento para una época, incluyendo:
            *   Iterar sobre `train_loader`.
            *   Aplicar `random_trans` a las imágenes de entrenamiento (si no se hizo ya en `MyDataset` o si se pasa como argumento para aplicarse en el momento).
            *   Pasar los datos al modelo.
            *   Calcular la pérdida.
            *   Realizar la retropropagación (`loss.backward()`).
            *   Actualizar los pesos (`optimizer.step()`).
        *   `utils.validate(...)`: Llama a la función `validate` (definida en `utils.py`). Esta función evalúa el modelo en el conjunto de validación, calculando la pérdida y la precisión de validación.
    *   **Posibles Modificaciones:**
        *   `epochs`: Puedes aumentar o disminuir el número de épocas. Entrenar por muy pocas épocas puede llevar a un subajuste (underfitting), mientras que demasiadas épocas pueden llevar a un sobreajuste (overfitting). Monitorea la pérdida y precisión de validación para decidir.
        *   Si las funciones `train` o `validate` en `utils.py` requieren diferentes argumentos o tienen un comportamiento diferente, necesitarás ajustar la llamada o el propio archivo `utils.py`.

## 10. Descongelar Modelo para Ajuste Fino (Fine-Tuning)

Después del entrenamiento inicial, se pueden descongelar algunas o todas las capas del modelo base para un ajuste fino con una tasa de aprendizaje más baja.

*   **Celda de Markdown (Instrucciones para Ajuste Fino):**
    *   Este paso es opcional si ya se alcanzó el 92% de precisión.
    *   Se sugiere descongelar el modelo y usar una tasa de aprendizaje muy baja.

*   **Celda de Código (Descongelar y Nuevo Optimizador):**
    ```python
    # filepath: d:\UNJBG\NOVEMO SEMESTRE\SISTEMAS EXPERTOS\lab-sistemas-expertos\moedlos\07_assessment.ipynb
    # Unfreeze the base model
    vgg_model.requires_grad_(True)
    optimizer = Adam(my_model.parameters(), lr=.0001) # Tasa de aprendizaje más baja
    ```
    *   **Explicación:**
        *   `vgg_model.requires_grad_(True)`: Establece `requires_grad` a `True` para todos los parámetros del `vgg_model` original (que es parte de `my_model`). Ahora, los gradientes se calcularán para estas capas y se actualizarán durante el entrenamiento.
        *   `optimizer = Adam(my_model.parameters(), lr=0.0001)`: Se crea un nuevo optimizador Adam. Es importante pasar `my_model.parameters()` para que el optimizador conozca todos los parámetros (incluidos los recién descongelados). Se usa una tasa de aprendizaje (`lr`) mucho más baja (0.0001) para el ajuste fino. Esto evita cambios drásticos en los pesos preentrenados que ya son buenos.
    *   **Posibles Modificaciones:**
        *   Podrías optar por descongelar solo una parte del modelo base (ej. las últimas capas convolucionales) en lugar de todo. Esto requeriría iterar selectivamente sobre los parámetros de `vgg_model.features`.
        *   La tasa de aprendizaje (`lr`) para el ajuste fino es crucial. `0.0001` es un punto de partida común, pero podrías necesitar ajustarla (ej. `1e-5`, `5e-5`).

*   **Celda de Código (Bucle de Entrenamiento para Ajuste Fino):**
    ```python
    # filepath: d:\UNJBG\NOVEMO SEMESTRE\SISTEMAS EXPERTOS\lab-sistemas-expertos\moedlos\07_assessment.ipynb
    epochs = 10

    for epoch in range(epochs):
        print('Epoch: {}'.format(epoch))
        utils.train(my_model, train_loader, train_N, random_trans, optimizer, loss_function)
        utils.validate(my_model, valid_loader, valid_N, loss_function)
    ```
    *   **Explicación:**
        *   Similar al bucle de entrenamiento anterior, pero ahora se usa el optimizador con la tasa de aprendizaje más baja y los parámetros del modelo base descongelados.
    *   **Posibles Modificaciones:**
        *   `epochs`: El número de épocas para el ajuste fino también puede variar. A menudo se necesitan menos épocas para el ajuste fino que para el entrenamiento inicial de las capas superiores.

## 11. Evaluar el Modelo

Se verifica la precisión final del modelo en el conjunto de validación.

*   **Celda de Markdown (Instrucciones de Evaluación):**
    *   Se espera una precisión de validación del 92% o más.
    *   Si no se alcanza, se sugiere reentrenar por más épocas o ajustar el aumento de datos.
    *   La función `evaluate` (probablemente `utils.validate`) devuelve la pérdida y la precisión.

*   **Celda de Código (Evaluación Final):**
    ```python
    # filepath: d:\UNJBG\NOVEMO SEMESTRE\SISTEMAS EXPERTOS\lab-sistemas-expertos\moedlos\07_assessment.ipynb
    utils.validate(my_model, valid_loader, valid_N, loss_function)
    ```
    *   **Explicación:**
        *   Llama a `utils.validate` una última vez para obtener las métricas finales en el conjunto de validación. El resultado impreso por esta función te dirá si alcanzaste el objetivo.

## 12. Ejecutar la Evaluación (Assessment)

Se utiliza un script proporcionado para evaluar formalmente el modelo.

*   **Celda de Markdown (Instrucciones para `run_assessment`):**
    *   Se indica que se deben ejecutar las siguientes dos celdas.
    *   Advierte que `run_assessment` asume que el modelo se llama `my_model`.

*   **Celda de Código (Importar `run_assessment`):**
    ```python
    # filepath: d:\UNJBG\NOVEMO SEMESTRE\SISTEMAS EXPERTOS\lab-sistemas-expertos\moedlos\07_assessment.ipynb
    from run_assessment import run_assessment
    ```
    *   **Explicación:** Importa la función `run_assessment` desde un script `run_assessment.py`.

*   **Celda de Código (Ejecutar `run_assessment`):**
    ```python
    # filepath: d:\UNJBG\NOVEMO SEMESTRE\SISTEMAS EXPERTOS\lab-sistemas-expertos\moedlos\07_assessment.ipynb
    run_assessment(my_model)
    ```
    *   **Explicación:**
        *   Llama a la función `run_assessment` pasándole el modelo entrenado `my_model`. Esta función probablemente carga un conjunto de datos de prueba oculto, evalúa el modelo y determina si se ha superado la evaluación.
    *   **Posibles Modificaciones:**
        *   Si has renombrado tu modelo (ej. `final_model` en lugar de `my_model`), debes cambiar el argumento aquí: `run_assessment(final_model)`.

## 13. Generar un Certificado

Información sobre cómo obtener un certificado si se supera la evaluación.

*   **Celda de Markdown (Instrucciones para el Certificado):**
    *   Si se pasa la evaluación, se debe regresar a la página del curso y hacer clic en "ASSESS TASK" para generar el certificado.

Este desglose debería ayudarte a entender cada parte del cuaderno y saber dónde y por qué podrías necesitar hacer modificaciones para experimentar o adaptarlo a diferentes problemas.
