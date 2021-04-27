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

Проанализировав графики фиксированных значений, был сделан вывод, что сеть, которая обучалась с параметром lr = 0.0001, имеет значение точности выше чем те, у которых темп обучения был равен 0.01 и 0.001 - 67.5% (выше на 3.5% и 8.5% соответсвенно).

### 2.Темп обучения с косинусным затуханием
Была совершена попытка увеличить точность нейронной сети при помощи  политики CosineDecay(косинусное затухание темпа обучения). Начальные значения темпа обучения были выбраны в диапазоне от 0.5 до 0.00011. Было выбрано кол-во итераций в диапазоне от 200 до 10000, а минимальное значение темпа обучения в диапазоне от 0.0 до 0.3 от начального значения темпа обучения в каждом случае. Наиболее оптимальными параметрами для политики темпа обучения с косинусным затуханием оказались:

```python
LearningRateScheduler(tf.keras.experimental.CosineDecay(0.00011, 500, 0.0, None))
```

С применением данных параметров точность возросла до 67.6% (+0.1% от фиксированного значения).
Также стоит подметить, что из-за слишком большого начального значения темпа обучения и большого кол-ва итераций в 

```python
LearningRateScheduler(tf.keras.experimental.CosineDecay(0.5, 10000, 0.01, None))
```

значения функции потерь при обучении и валидации стало очень большим.

### 3.Темп обучения с косинусным затуханием с рестартами

В дальнейшем была использована политика темпа обучения CosineDecayRestarts (косинусное затухание темпа обучения с рестартами). Данный метод может помочь выбраться из "узкого" локального минимума. Диапазон в котором были подобраны параметры начального темпа обучения был уменьшен - от 0.00011 до 0.01, кол-во начальных итераций - от 500 до 4000, и диапазон минимального темпа обучения был изменен от 0.0 до 0.5 относительно начального значения темпа обучения в каждом случае, параметру отвечающему за кол-во итераций на i-том периоде были присвоены значения в диапазоне от 0.5 (на каждом следующем периоде в 2 раза меньше итераций) до 2.0 (на каждом следующем периоде в 2 раза больше итераций), а параметру отвечающему за начальный темп обучения на периоде - от 0.75 (на каждом следующем периоде начальный темп уменьшается в 0.75 раз) до 1,0(на каждом следующем периоде начальный темп остается неизменным). Было принято решение не брать значение выше 1.0 по причине того что мы могли выпрыгнуть из хорошего локального минимума в более плохой в результате постоянного увеличения начального темпа обучения после каждого рестарта. В результате сравнения графиков было выяснено, что политика темпа обучения CosineDecayRestarts со следующими параметрами, является самой оптимальной среди данного набора:

```python
LearningRateScheduler(tf.keras.experimental.CosineDecayRestarts(0.00011, 500, 2.0, 1.0, 0.0, None))
```

Значение точности с данными параметрами равняется 67.8(+0.2% от предыдущего и +0.3% от фиксированного).

### 4.Сравнение графиков разных политик с оптимальными параметрами

В результате сравнения графиков можно сказать, что применение особых политик к темпу обучения, в нашем случае, к особым улучшениям не привело (всего +0.3% повышение точности относительно фиксированного значения). Само максимальное значение точности составило 67.8%. То, что все графики идут довольно близко друг с другом, может говорить о том, что локальный минимум, достигаемый уже при фиксированном значении является довольно широким. Также вероятно, что подобрав другие параметры для политик можно получить максимальную точность выше, чем в данном случае.
