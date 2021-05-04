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

Пример аугментированного изображения полученного при использовании оптимальных параметров

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

Пример аугментированного изображения полученного при использовании оптимальных параметров

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

Пример аугментированного изображения полученного при использовании оптимальных параметров

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

Пример аугментированного изображения комбинации техник аугментации с оптимальными параметрами

![2](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/OptimalComboSample.jpg)

## Анализ
### 1.Обучение с использованием техники аугментации данных - RandomFlip
Проанализировав графики обучения, был сделан вывод, что сеть, которая обучалась с параметром 'horizontal', имеет самые высокие значения точности, соответственно параметр является оптимальным. По сравнению с оригинальным датасетом(67,05%) точность возросла(67,71%), но незначительно(+0.66%). 

### 2.Обучение с использованием техники аугментации данных - RandomRotation
Была совершена попытка увеличить точность нейронной сети при помощи техники аугментации RandomRotation. Оптимальными параметрами оказались - factor = 0.05, fill_mode='constant', interpolation = 'bilinear' ,seed=None, name=None, fill_value=0.0. При таких параметрах точность данной нейронной сети(67,1%) была увеличена незначительно(+0,05%) относительно оригинала(67,05%). Также прослеживалась тенденция увеличения точности сети при уменьшении параметра factor(1.10 -> 0.05) c 64.6% до 67.1%. Стоит отметить, что при использоании большого значения параметра factor=1.10 наблюдается большой спад точности относительно других наборов параметров.

### 3.Обучение с использованием техники аугментации данных - random_crop
В дальнейшем была проведена работа с использованием техники аугментации данных - random_crop. Мы увеличили размер, к которому изначально приводятся изображения, для того, чтобы позже уменьшить его до значений которые принимает наша модель при помощи техники аугментации данных random_crop. Это было сделано потому что при применении random_crop с параметрами размера изображения являющимися меньшими, чем те, которые принимает наша модель, при повторном ресайзе мы потеряем качество изображения и, следовательно, точность.
Оптимальными оказались параметры размера изображения 250x250. С ними нейронная сеть достигает точности 68,13% что на 1,08% выше чем у оригинала(67,05%). Также наблюдается закономерность: падение точности(68,13% -> 36,76%) при увеличении приводимого изначально размера изображения (250x250 -> 800x800). Также наблюдается резкий спад точности при значении приводимого размера изображения 800х800. Это может быть обусловлено тем, что некоторые изображения в датасете имеют начальный размер меньше, чем 800х800, из-за чего мы теряем качество изображения.

### 4.Обучение с использованием комбинации техник аугментации данных
Комбинация техник аугментации к серьезным улучшениям не привело. Была достигнута точность равная 67,54%.
В сранении:
#### с оригиналом(67,05%) значение точности увеличилось на 0,49% 
#### с RandomFlip(67,71%) уменьшилось на 0,17% 
#### с RandomRotation(67.1%) увеличилось на 0,44%
#### c random_crop(68,13%) уменьшимлось на 0.59%
По сравнению с оригиналом наиболее выгодной техникой аугментации данных оказалась random_crop - +1,08%, однако все значение не сильно разнятся.
