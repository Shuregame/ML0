# ML0
## Метрические алгоритмы классификации

## Алгоритм ближайшего соседа (1NN).
Алгоритм ближайшего соседа является самым простым алгоритмом классификации. Он относит классифицируемый объект к тому
классу, которому принадлежит ближайший обучающий объект. По сути это просто частный случай алгоритма ближайших соседей (KNN), где K>1.

## Ирисы Фишера
Для примера возьмем изветсную выборку "Ирисы Фишера". По легенде существует множество цветков(ирисов), которые пренадлежат разным видам(классам):  

Ирис щетинистый (Iris setosa), Ирис виргинский (Iris virginica) и Ирис разноцветный (Iris versicolor). 

Также были для каждого вида измерялись четыре характеристики (в сантиметрах):

1)Длина наружной доли околоцветника (англ. sepal length);

2)Ширина наружной доли околоцветника (англ. sepal width);

3)Длина внутренней доли околоцветника (англ. petal length);

4)Ширина внутренней доли околоцветника (англ. petal width).

## Задача
Нам нужно классифицировать произвольную точку и отнести ее к определенному классу цветков. Для этого будем классифицировать точку, согласно так называемому "ближайшему соседу" (ближайшей точке).

На графике видно как мы классифицировали произвольную точку, видим что она окрасилась в соответствующий цвет.
![Image alt](https://github.com/Shuregame/ML0/raw/master/algorithm kNN.png/algorithm kNN.png)



## Преимущества
1)Простота реализации

## Недостатки
1)Неустойчивость к погрешностям.

2)Отсутствие параметров, которые можно было бы настраивать по выборке. Алгоритм полностью зависит от того, насколько удачно выбрана метрика.

3)Низкое качество классификации.
