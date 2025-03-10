# Отчет по ЛР №1

## Архитектура нейронной сети
1.Слой свертки(Conv2D), 3х3 ядро, 8 фильтров
```phyton
x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x)
```
2.Слой пулинга(MaxPool2D) - выбор макс. значения в окне
```phyton
x = tf.keras.layers.MaxPool2D()(x)
```
3. "Сдавливание"(Flatten) матрицы признаков в одномерный вектор
```phyton
x = tf.keras.layers.Flatten()(x)
```
4.Слой в котором каждый нейрон связян со всеми входами(Dense)
```phyton
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation=tf.keras.activations.softmax)(x)
```
NUM_CLASSES = 101 - кол-во выходов.

activation=tf.keras.activations.softmax - преобразует вектор входных данных в вектор вероятностных распределений

## Графики обучения
![image](https://user-images.githubusercontent.com/76451709/114280593-a9aae380-9a42-11eb-9187-7c12c1de30e9.png)

1.Оригинал
```phyton
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(input)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Flatten()(x)
```

График метрики качества оригинала
![DefMetr](https://github.com/k0styamba/CNN-food-101/blob/master/Graph/Default/epoch_categorical_accuracy.svg)

График функиции потерь оригинала
![DefLoss](https://github.com/k0styamba/CNN-food-101/blob/master/Graph/Default/epoch_loss.svg)

2.Модифицированная архитектура
```phyton
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(inputs)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Conv2D(filters=8, kernel_size=3)(x)
  x = tf.keras.layers.MaxPool2D()(x)
  x = tf.keras.layers.Flatten()(x)
```

График метрики качества модифицированной архитектуры
![MyMetr](https://github.com/k0styamba/CNN-food-101/blob/master/Graph/My/epoch_categorical_accuracy.svg)

График функиции потерь модифицированной архитектуры
![MyLoss](https://github.com/k0styamba/CNN-food-101/blob/master/Graph/My/epoch_loss.svg)

## Анализ
В результате сравнения графиков оригинальной архитектуры и модифицированной, можно сделать вывод, что добавление 3ех слоев свертки и 4ех слоев пулинга к исправлению ошибки при обучении не привело. Это связано с тем, что оригинальная архитектура содержала крайне малое кол-во слоев, а модифицированная была "собрана" произвольно.
