# Отчет по ЛР №3

## Графики обучения нейронной сети EfficientNetB0
### 1.Графики обучения с фиксированными темпами обучения

![11](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/Static/legend.png)

Метрика качества
![1](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/Static/epoch_categorical_accuracy%20-%20static.svg)

Функция потерь
![2](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/Static/epoch_loss%20-%20static.svg)

### 2.Графики обучения с использованием CosineDecay

![22](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/CosineDecay/CosineDecay%20-%20train.png)

train и validation - голубой:

```python
LearningRateScheduler(tf.keras.experimental.CosineDecay(0.01, 7500, 0.001, None))
```

train и validation - зеленый:

```python
LearningRateScheduler(tf.keras.experimental.CosineDecay(0.0005, 300, 0.1, None))
```

train и validation - коричневый:

```python
LearningRateScheduler(tf.keras.experimental.CosineDecay(0.5, 10000, 0.01, None))
```

train и validation - розовый:

```python
LearningRateScheduler(tf.keras.experimental.CosineDecay(0.007, 200, 0.3, None))
```

train и validation - синий:
```python
LearningRateScheduler(tf.keras.experimental.CosineDecay(0.00011, 500, 0.0, None))
```

train и validation - оранжевый:

```python
LearningRateScheduler(tf.keras.experimental.CosineDecay(0.0011, 500, 0.0, None))
```

![33](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/CosineDecay/CosineDecay%20-%20validation.png)

Метрика качества (validation)
![3](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/CosineDecay/epoch_categorical_accuracy%20CosineDecay%20-%20validation.svg)

Функция потерь (все кроме CosineDecay(0.5, 10000, 0.01, None)) (validation)
![4](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/CosineDecay/epoch_loss%20CosineDecay%20-%20validation.svg)

Функция потерь (CosineDecay(0.5, 10000, 0.01, None)) (validation)
![4](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/CosineDecay/epoch_loss%20CosineDecay%20-%20validation%20-%200.5.svg)

![33](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/CosineDecay/CosineDecay%20-%20train.png)

Функция потерь (все кроме CosineDecay(0.5, 10000, 0.01, None)) (train)
![4](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/CosineDecay/epoch_loss%20CosineDecay%20-%20train.svg)

Функция потерь (CosineDecay(0.5, 10000, 0.01, None)) (train)
![4](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/CosineDecay/epoch_loss%20CosineDecay%20-%20train%20-%200.5.svg)

### 3.Графики обучения с использованием CosineDecayRestarts

train и validation - голубой:

```python
LearningRateScheduler(tf.keras.experimental.CosineDecayRestarts(0.01, 1000, 1.0, 0.75, 0.001, None))
```

train и validation - розовый:

```python
LearningRateScheduler(tf.keras.experimental.CosineDecayRestarts(0.0005, 4000, 2.0, 1.0, 0.5, None))
```

train и validation - зеленый:

```python
LearningRateScheduler(tf.keras.experimental.CosineDecayRestarts(0.005, 1000, 1.0, 0.8, 0.1, None))
```

train и validation - оранжевый:

```python
LearningRateScheduler(tf.keras.experimental.CosineDecayRestarts(0.00011, 500, 2.0, 1.0, 0.0, None))
```

train и validation - коричневый:

```python
LearningRateScheduler(tf.keras.experimental.CosineDecayRestarts(0.0011, 500, 0.5, 1.0, 0.0, None))
```

train и validation - синий:

```python
LearningRateScheduler(tf.keras.experimental.CosineDecayRestarts(0.0011, 500, 2.0, 1.0, 0.0, None))
```

![33](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/CosineDecayRestarts/CosineDecayRestarts%20-%20validation.png)

Метрика качества (validation)
![5](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/CosineDecayRestarts/epoch_categorical_accuracy%20-%20CosineDecayRestarts%20-%20validation.svg)

Функция потерь (validation)
![6](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/CosineDecayRestarts/epoch_loss%20-%20CosineDecayRestarts%20-%20validation.svg)

![33](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/CosineDecayRestarts/CosineDecayRestarts%20-%20train.png)

Функция потерь (train)
![6](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/CosineDecayRestarts/epoch_loss%20-%20CosineDecayRestarts%20-%20train.svg)

### 4.Графики обучения с оптимальными значениями параметров

Оптимальным значением фиксированного темпа обучения оказалось lr = 0.0001 - синий(validation) - оранжевый(train)

```python
optimizer=tf.optimizers.Adam(lr=0.001)
```

Оптимальным значением параметров для темпа обучения с косинусным затуханием (CosineDecay(0.00011, 500, 0.0, None)) - голубой(validation) - коричневый(train)

```python
LearningRateScheduler(tf.keras.experimental.CosineDecay(0.00011, 500, 0.0, None))
```

Оптимальным значением параметров для темпа обучения с косинусным затуханием с рестартом (CosineDecayRestarts(0.00011, 500, 2.0, 1.0, 0.0, None)) - зеленый(validation) - розовый(train)

```python
LearningRateScheduler(tf.keras.experimental.CosineDecayRestarts(0.00011, 500, 2.0, 1.0, 0.0, None))
```

![33](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/Optimal/validation.png)

Метрика качества (validation)

![6](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/Optimal/epoch_categorical_accuracy%20-%20validation.svg)

Функция потерь (validation)

![6](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/Optimal/epoch_loss%20-%20validation.svg)

![33](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/Optimal/train.png)

Функция потерь (train)

![6](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab3/Graphs/Optimal/epoch_loss%20-%20train.svg)

## Анализ

### 1.Фиксированные значения темпа обучения
Проанализировав графики фиксированных значений, был сделан вывод, что сеть, которая обучалась с параметром lr = 0.01, имеет значения точности ниже чем те, у которых темп обучения был равен 0.001 и 0.0001(на 5% и 8% соответсвенно). В дальнейшем работы проводились именно с ними. Была совершена попытка увеличить точность нейронной сети при помощи CosineDecay(косинусное затухание темпа обучения). Начальные значения темпа обучения были выбраны немного выше, чем предыдущие оптимальные - 0.0011, 0.00011 соответственно. Было выбрано кол-во итераций равное 500, а минимальное значение темпа обучения равное 0 (500 * 0,0). К каким либо улучшениям это не привело. В дальнейшем была использована CosineDecayRestarts (косинусное затухание с рестартами). Данный метод может помочь выбраться из "узкого" локального минимума. Параметры начального темпа обучения, кол-ва начальных итераций, и минимальной темп обучения остались неизменными, параметру отвечающему за кол-во итераций на i-том периоде было присвоено значение 2,0 (на первом периоде 500, на втором 1000 итераций и тд), а параметру отвечающему за начальный темп обучения на периоде - 1,0(остается всегда неизменным). Это также к значительным улучшениям не привело. Из этого можно утверждать что, был найден стабильный локальный минимум.
