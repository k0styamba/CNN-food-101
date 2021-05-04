# Отчет по ЛР №4

## Графики обучения нейронной сети EfficientNetB0
### 1.Графики обучения с использованием техники аугментации данных - RandomFlip


```python
tf.keras.layers.experimental.preprocessing.RandomFlip(mode = '...' , seed=None, name=None)
```

Легенда(validation) с указанными соответствующими параметрами

![1](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomFlip/Validation.png)

Метрика качества(validation)
![1](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomFlip/epoch_categorical_accuracy_validation.svg)

Функция потерь(validation)
![2](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomFlip/epoch_loss_validation.svg)

Легенда(train) с указанными соответствующими параметрами

![1](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomFlip/Train.png)

Функция потерь(train)
![2](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomFlip/epoch_loss_train.svg)

Пример аугментированного изображения

![2](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/OptimalFlipSample.jpg)

### 2.Графики обучения с использованием техники аугментации - RandomRotation

```python
tf.keras.layers.experimental.preprocessing.RandomRotation(factor = ... , fill_mode = ... , interpolation = ... ,seed=None, name=None, fill_value=0.0)
```

Легенда(validation) с указанными соответствующими параметрами

![1](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomRotation/Validation.png)

Метрика качества(validation)
![1](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomRotation/epoch_categorical_accuracy_validation.svg)

Функция потерь(validation)
![2](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomRotation/epoch_loss_validation.svg)

Легенда(train) с указанными соответствующими параметрами

![1](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomRotation/Train.png)

Функция потерь(train)
![2](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomRotation/epoch_loss_train.svg)

Пример аугментированного изображения

![2](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/OptimalRotationSample.jpg)

### 3.Графики обучения с использованием техники аугментации - random_crop

Задаем "изначальное" разрешение изображения

```python
example['image'] = tf.image.resize(example['image'], tf.constant([..., ...]), method = 'nearest')
```

Применяем на него random_crop, который выдает изображение с параметрами которые принимает наша модель

```python
def process_data(image, label):
  return tf.image.random_crop(image, [224, 224, 3]), label
```

```python
.map(process_data)\
```

Легенда(validation) с указанными соответствующими параметрами

![5](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomCrop/Validation.png)

Метрика качества(validation)
![1](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomCrop/epoch_categorical_accuracy_validation.svg)

Функция потерь(validation)
![2](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomCrop/epoch_loss_validation.svg)

Легенда(train) с указанными соответствующими параметрами

![1](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomCrop/Train.png)

Функция потерь(train)
![2](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomCrop/epoch_loss_train.svg)

Пример аугментированного изображения

![2](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/OptimalCropSample.jpg)

### 4.Графики обучения с использованием комбинации техник аугментации с оптимальными параметрами - random_crop -> RandomRotation -> RandomFlip

```python
example['image'] = tf.image.resize(example['image'], tf.constant([250, 250]), method = 'nearest')
```

```python
def process_data(image, label):
  return tf.image.random_crop(image, [224, 224, 3]), label
```

```python
.map(process_data)\
```

```python
tf.keras.layers.experimental.preprocessing.RandomRotation(0.05, fill_mode='constant', interpolation = 'bilinear' ,seed=None, name=None, fill_value=0.0)
```

```python
tf.keras.layers.experimental.preprocessing.RandomFlip(mode = 'horizontal' , seed=None, name=None)
```

Легенда

![5](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/OptimalComb/Legend.png)

Метрика качества
![1](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/OptimalComb/epoch_categorical_accuracy.svg)

Функция потерь
![2](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomCrop/epoch_loss_train.svg)

Пример аугментированного изображения

![2](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/OptimalComb/epoch_loss.svg)

## Анализ
Проанализировав графики фиксированных значений, был сделан вывод, что сеть, которая обучалась с параметром lr = 0.01, имеет значения точности ниже чем те, у которых темп обучения был равен 0.001 и 0.0001(на 5% и 8% соответсвенно). В дальнейшем работы проводились именно с ними. Была совершена попытка увеличить точность нейронной сети при помощи CosineDecay(косинусное затухание темпа обучения). Начальные значения темпа обучения были выбраны немного выше, чем предыдущие оптимальные - 0.0011, 0.00011 соответственно. Было выбрано кол-во итераций равное 500, а минимальное значение темпа обучения равное 0 (500 * 0,0). К каким либо улучшениям это не привело. В дальнейшем была использована CosineDecayRestarts (косинусное затухание с рестартами). Данный метод может помочь выбраться из "узкого" локального минимума. Параметры начального темпа обучения, кол-ва начальных итераций, и минимальной темп обучения остались неизменными, параметру отвечающему за кол-во итераций на i-том периоде было присвоено значение 2,0 (на первом периоде 500, на втором 1000 итераций и тд), а параметру отвечающему за начальный темп обучения на периоде - 1,0(остается всегда неизменным). Это также к значительным улучшениям не привело. Из этого можно утверждать что, был найден стабильный локальный минимум.
