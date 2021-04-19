# Отчет по ЛР №2

## Графики обучения
![image](https://user-images.githubusercontent.com/76451709/114280593-a9aae380-9a42-11eb-9187-7c12c1de30e9.png)

1.EfficientNetB0 обученная с использовнием случайного начального приближения

График метрики качества EfficientNetB0 обученной с использовнием случайного начального приближения
![DefMetr](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab2/Graphs/3/epoch_categorical_accuracy.svg)

График функиции потерь EfficientNetB0 обученной с использовнием случайного начального приближения
![DefLoss](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab2/Graphs/3/epoch_loss.svg)

2.EfficientNetB0 обученная с использовнием Transfer Learning на базе Imagenet

График метрики качества EfficientNetB0 обученной с использовнием Transfer Learning на базе Imagenet
![MyMetr](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab2/Graphs/4/epoch_categorical_accuracy.svg)

График функиции потерь EfficientNetB0 обученной с использовнием Transfer Learning на базе Imagenet
![MyLoss](https://github.com/k0styamba/CNN-food-101/blob/myoutputLab2/Graphs/4/epoch_loss.svg)

## Анализ
В результате сравнения графиков обучения со случайным начальным приближением и обучения с использовнием Transfer Learning с предобученными весами на базе Imagenet, можно сделать вывод, что в данном случае метод Transfer Learning эффективней, чем метод с использованием случайного начального приближения(на графиках метрики точности +60% на train и +53% на validation).Также при помощи метода Transfer Learning удлаось значительно уменьшить время обучения с ≈5,5 часов (со случайным начальным приближением) до ≈1,5 часов (с методом Transfer Learning).
