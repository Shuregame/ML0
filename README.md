# ML0
## Метрические алгоритмы классификации

## Гипотеза компактности: 
Схожие объекты, как правило, лежат в одном классе.


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

![Image alt](https://github.com/Shuregame/ML0/blob/master/algorithm%20kNN.png)




## Преимущества
1)Простота реализации

## Недостатки
1)Неустойчивость к погрешностям.

2)Отсутствие параметров, которые можно было бы настраивать по выборке. Алгоритм полностью зависит от того, насколько удачно выбрана метрика.

3)Низкое качество классификации.

## Алгоритм kNN

<a href="https://www.codecogs.com/eqnedit.php?latex=$w(i,u)&space;=&space;\left&space;[&space;i&space;\leqslant&space;k&space;\right&space;]$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$w(i,u)&space;=&space;\left&space;[&space;i&space;\leqslant&space;k&space;\right&space;]$" title="$w(i,u) = \left [ i \leqslant k \right ]$" /></a>

Преимущества:

-менее чувствителен к шумуж;

-появился параметр k.






<a href="https://www.codecogs.com/eqnedit.php?latex=LOO$(k,X^l)$&space;=&space;\sum_{i=1}^{l}$[a(x_i;X^l&space;/&space;\left&space;\{&space;x_i&space;\right&space;\},&space;\neq&space;y]&space;\rightarrow&space;$\min_{k}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?LOO$(k,X^l)$&space;=&space;\sum_{i=1}^{l}$[a(x_i;X^l&space;/&space;\left&space;\{&space;x_i&space;\right&space;\},&space;\neq&space;y]&space;\rightarrow&space;$\min_{k}" title="LOO$(k,X^l)$ = \sum_{i=1}^{l}$[a(x_i;X^l / \left \{ x_i \right \}, \neq y] \rightarrow $\min_{k}" /></a>
