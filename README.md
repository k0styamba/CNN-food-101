# Отчет по ЛР №3

## Графики обучения нейронной сети EfficientNetB0 (validation)
### 1.Графики обучения с фиксированными темпами обучения

Красный - lr = 0.0001.

Оранжевый - lr = 0.001.

Синий - lr = 0.01.

Метрика качества
![1](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/Static/epoch_categorical_accuracy.svg)

Функция потерь
![2](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/Static/epoch_loss.svg)

### 2.Графики обучения с использованием CosineDecay

CosineDecay с начальным lr = 0.0011 - Оранжевый
```python
LearningRateScheduler(tf.keras.experimental.CosineDecay(0.0011, 500, 0.0, None))
```
CosineDecay с начальным lr = 0.00011 - Синий
```python
LearningRateScheduler(tf.keras.experimental.CosineDecay(0.00011, 500, 0.0, None))
```

Метрика качества
![3](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/CosineDecay/epoch_categorical_accuracy.svg)

Функция потерь
![4](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/CosineDecay/epoch_loss.svg)

### 3.Графики обучения с использованием CosineDecayRestarts

CosineDecayRestarts с начальным lr = 0.0011 - Голубой
```python
LearningRateScheduler(tf.keras.experimental.CosineDecay(0.0011, 500, 2.0, 1.0, 0.0, None))
```
CosineDecayRestarts с начальным lr = 0.00011 - Красный
```python
LearningRateScheduler(tf.keras.experimental.CosineDecay(0.00011, 500, 2.0, 1.0, 0.0, None))
```

Метрика качества
![5](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/CosineDecayRestarts/epoch_categorical_accuracy.svg)

Функция потерь
![6](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/CosineDecayRestarts/epoch_loss.svg)

## Анализ
Проанализировав графики фиксированных значений, был сделан вывод, что сеть, которая обучалась с параметром lr = 0.01, имеет значения точности ниже чем те, у которых темп обучения был равен 0.001 и 0.0001. В дальнейшем работы проводились именно с ними. 
