# 🧠 Práctica 2: Autoencoders - Denoising y Super-Resolución

Implementación de autoencoders convolucionales para dos tareas principales: eliminación de ruido gaussiano y super-resolución de imágenes usando el dataset MNIST.

---

## 📋 Descripción

Este proyecto implementa y compara diferentes arquitecturas de autoencoders para:

1. **Denoising (Eliminación de ruido)**: Reconstrucción de imágenes corruptas con ruido gaussiano
2. **Super-Resolución**: Aumento de resolución desde imágenes de baja calidad (7×7 y 14×14) a 28×28 píxeles

### 🎯 Objetivos

- Implementar autoencoders convolucionales efectivos
- Comparar el rendimiento en diferentes niveles de degradación
- Analizar los límites de reconstrucción desde información extremadamente limitada
- Demostrar la capacidad de los modelos para tareas múltiples (SR + denoising simultáneo)

---

## 🏗️ Arquitectura

### Modelo de Denoising

```
Encoder: 28×28 → 14×14 → 7×7 → 1×1 (64 canales)
Decoder: 1×1 → 7×7 → 14×14 → 28×28
```

### Modelo de Super-Resolución

#### SR 7×7 → 28×28
```
Encoder: 7×7 (1→64→128→256 canales)
Decoder: 7×7 → 14×14 → 28×28 (ConvTranspose2d + Conv2d)
```

#### SR 14×14 → 28×28
```
Encoder: 14×14 (1→32→64 canales)
Decoder: 14×14 → 28×28 (ConvTranspose2d + Conv2d)
```

---

## 📊 Modelos Entrenados

| Modelo | Entrada | Degradación | Loss Inicial | Loss Final | Mejora | Epochs |
|--------|---------|-------------|--------------|------------|--------|--------|
| **Denoising** | 28×28 | Ruido σ=0.4 | 0.0426 | 0.0106 | 75.14% | 10 |
| **SR 7×7** | 7×7 (49 px) | Solo resolución | 0.1125 | 0.0190 | 83.10% | 15 |
| **SR 14×14 limpio** | 14×14 (196 px) | Solo resolución | 0.0077 | 0.0014 | 81.72% | 10 |
| **SR 14×14 ruidoso** | 14×14 (196 px) | Resolución + ruido σ=0.15 | 0.1130 | 0.0080 | 92.93% | 10 |

---

## 🚀 Instalación

### Requisitos

```bash
Python 3.8+
PyTorch 2.0+
torchvision
numpy
matplotlib
```

### Instalar dependencias

```bash
pip install torch torchvision numpy matplotlib
```

---

## 💻 Uso

### Entrenamiento completo

```python
# Ejecutar el notebook completo
jupyter notebook Practica2.ipynb
```

### Entrenar modelos individuales

```python
# Denoising
model_denoise = ModelFactory.create_denoising_autoencoder(device)
denoising_task = DenoisingTask(device, noise_std=0.4)
trainer_denoise.train(train_loader, num_epochs=10, ...)

# Super-resolución 7x7
model_sr7 = ModelFactory.create_super_resolution_autoencoder(device, input_size=7)
sr7_task = SuperResolutionTask(device, low_res_size=7)
trainer_sr7.train(train_loader, num_epochs=15, ...)

# Super-resolución 14x14 con ruido
model_sr14 = ModelFactory.create_super_resolution_autoencoder(device, input_size=14)
sr14_task = SuperResolutionTask(device, low_res_size=14, add_noise=True, noise_std=0.15)
trainer_sr14.train(train_loader, num_epochs=10, ...)
```

---

## 📈 Resultados

### Convergencia de Pérdidas

Los modelos muestran patrones de convergencia distintos según la dificultad de la tarea:

- **SR 14×14 sin ruido**: Convergencia casi instantánea (epoch 1-2) → Tarea trivial
- **Denoising**: Convergencia rápida (epoch 3-4)
- **SR 7×7**: Caída dramática en epoch 4-5, luego estabilización
- **SR 14×14 con ruido**: Convergencia gradual más lenta (epoch 6-7)

### Observaciones Clave

1. **SR 7×7 representa el límite práctico**: Loss final de 0.019 (el más alto), indicando que reconstruir desde 49 píxeles es extremadamente desafiante pero viable.

2. **Añadir ruido transforma tareas triviales**: El SR 14×14 pasa de trivial (loss inicial 0.008) a desafiante (0.113) con ruido gaussiano.

3. **Mejor mejora porcentual**: SR 14×14 con ruido (92.93%), demostrando capacidad de multi-tarea (super-resolución + denoising).

---

## 🎨 Visualizaciones

Cada modelo genera visualizaciones con 3 filas:

```
Fila 1: Imágenes originales (28×28, nítidas)
Fila 2: Imágenes degradadas (ruido/baja resolución)
Fila 3: Imágenes reconstruidas por el autoencoder
```

**Nota técnica**: Se utiliza `interpolation='nearest'` en matplotlib para evitar artefactos visuales en imágenes de baja resolución.

---

## 🔬 Análisis Técnico

### Por qué NO se añadió ruido al SR 7×7

La decisión de no añadir ruido gaussiano a las imágenes de 7×7 se fundamenta en:

1. **Información limitada**: 49 píxeles vs 784 objetivo (ratio 1:16)
2. **Degradación suficiente**: El modelo ya alcanza el loss final más alto (0.019)
3. **Límite de viabilidad**: Añadir ruido comprometería la convergencia sin aportar valor analítico

En contraste, el SR 14×14 (196 píxeles, ratio 1:4) tiene margen para degradación adicional, resultando en la comparación más interesante del experimento.

### Mejoras Arquitectónicas

El modelo SR 7×7 utiliza una arquitectura más profunda (64→128→256 canales) comparada con el SR 14×14 (32→64 canales) para compensar la extrema reducción de información de entrada.

---

## 🧪 Experimentos Adicionales

### Hiperparámetros Probados

- **Ruido gaussiano**: σ ∈ {0.3, 0.4, 0.5} para denoising
- **Ruido en SR**: σ=0.15 para 14×14 (óptimo para dificultad media-alta)
- **Epochs**: 10 (estándar), 15 (SR 7×7 para mejor convergencia)
- **Learning rate**: 1e-3 con weight decay 1e-5

---

## 📂 Estructura del Proyecto

```
.
├── Practica2.ipynb          # Notebook principal con todo el código
├── README.md                # Este archivo
├── data/                    # Dataset MNIST (descarga automática)
└── results/                 # Visualizaciones y gráficas generadas
    ├── denoising_results.png
    ├── sr7_results.png
    ├── sr14_clean_results.png
    ├── sr14_noisy_results.png
    └── loss_comparison.png
```

---

## 🎓 Conceptos Implementados

- **Autoencoders convolucionales**: Reducción y reconstrucción de dimensionalidad
- **Skip connections implícitas**: A través de la arquitectura simétrica
- **Multi-task learning**: Super-resolución + denoising simultáneo
- **Transfer learning concepts**: Misma arquitectura, diferentes tareas
- **Patrones de diseño**: Factory, Strategy, Service Layer

---

## 📚 Referencias

- Dataset: [MNIST - Yann LeCun](http://yann.lecun.com/exdb/mnist/)
- Framework: [PyTorch](https://pytorch.org/)
- Autoencoders: [Deep Learning Book - Ian Goodfellow](https://www.deeplearningbook.org/)

---

## 👤 Autor

**Tu Nombre**  
Universidad / Curso  
Práctica 2 - Redes Neuronales y Deep Learning

---

## 📄 Licencia

Este proyecto es material educativo para la asignatura de Deep Learning.

---

## 🙏 Agradecimientos

- Profesores y equipo docente del curso
- Comunidad de PyTorch por la documentación
- Dataset MNIST por ser el benchmark estándar

---

## 📞 Contacto

Para dudas o sugerencias:
- Email: tu.email@universidad.edu
- GitHub: [@tu-usuario](https://github.com/tu-usuario)

---

**⭐ Si este proyecto te fue útil, no olvides darle una estrella!**
