# Отчет по ЛР №5

## Графики обучения нейронной сети EfficientNetB0
### 1.Графики обучения с использованием техники аугментации и политики темпа обучения с оптимальными параметрами

Легенда

![1](https://github.com/k0styamba/CNN-food-101/blob/master/Graph/Original.png)

Метрика качества
![1](https://github.com/k0styamba/CNN-food-101/blob/master/Graph/epoch_categorical_accuracy_orig.svg)

Функция потерь
![2](https://github.com/k0styamba/CNN-food-101/blob/master/Graph/epoch_loss_orig.svg)

### 2.Графики обучения с использованием техники обучения FineTuning

Для данной техники обучения были подобраны значения темпа обучения 1е-6 и 1е-7:

Легенда(validation) с указанными соответствующими параметрами

![1](https://github.com/k0styamba/CNN-food-101/blob/master/Graph/FineTuneValidation.png)

Метрика качества(validation)
![1](https://github.com/k0styamba/CNN-food-101/blob/master/Graph/epoch_categorical_accuracy_fine_validation.svg)

Функция потерь(validation)
![2](https://github.com/k0styamba/CNN-food-101/blob/master/Graph/epoch_loss_fine_validation.svg)

Легенда(train) с указанными соответствующими параметрами

![1](https://github.com/k0styamba/CNN-food-101/blob/master/Graph/FineTuneTrain.png)

Функция потерь(train)
![2](https://github.com/k0styamba/CNN-food-101/blob/master/Graph/epoch_loss_fine_train.svg)

Пример аугментированного изображения полученного при использовании оптимальных параметров

![2](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab4/OptimalRotationSample.jpg)

## Анализ

