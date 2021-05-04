# Отчет по ЛР №4

## Графики обучения нейронной сети EfficientNetB0
### 1.Графики обучения с использованием техники аугментации данных - RandomFlip

Легенда(validation) с указанными соответствующими параметрами

![1](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomFlip/Validation.png)

Метрика качества
![1](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomFlip/epoch_categorical_accuracy_validation.svg)

Функция потерь
![2](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomFlip/epoch_loss_validation.svg)

Легенда(train) с указанными соответствующими параметрами

![1](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomFlip/Train.png)

Функция потерь
![2](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomFlip/epoch_loss_train.svg)

### 2.Графики обучения с использованием техники аугментации - RandomRotation

Легенда(validation) с указанными соответствующими параметрами

![1](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomRotation/Validation.png)

Метрика качества
![1](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomRotation/epoch_categorical_accuracy_validation.svg)

Функция потерь
![2](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomRotation/epoch_loss_validation.svg)

Легенда(train) с указанными соответствующими параметрами

![1](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomRotation/Train.png)

Функция потерь
![2](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/Graphs/RandomRotation/epoch_loss_train.svg)

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
Проанализировав графики фиксированных значений, был сделан вывод, что сеть, которая обучалась с параметром lr = 0.01, имеет значения точности ниже чем те, у которых темп обучения был равен 0.001 и 0.0001(на 5% и 8% соответсвенно). В дальнейшем работы проводились именно с ними. Была совершена попытка увеличить точность нейронной сети при помощи CosineDecay(косинусное затухание темпа обучения). Начальные значения темпа обучения были выбраны немного выше, чем предыдущие оптимальные - 0.0011, 0.00011 соответственно. Было выбрано кол-во итераций равное 500, а минимальное значение темпа обучения равное 0 (500 * 0,0). К каким либо улучшениям это не привело. В дальнейшем была использована CosineDecayRestarts (косинусное затухание с рестартами). Данный метод может помочь выбраться из "узкого" локального минимума. Параметры начального темпа обучения, кол-ва начальных итераций, и минимальной темп обучения остались неизменными, параметру отвечающему за кол-во итераций на i-том периоде было присвоено значение 2,0 (на первом периоде 500, на втором 1000 итераций и тд), а параметру отвечающему за начальный темп обучения на периоде - 1,0(остается всегда неизменным). Это также к значительным улучшениям не привело. Из этого можно утверждать что, был найден стабильный локальный минимум.
