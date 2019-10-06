# ML0
## Метрические алгоритмы классификации

## <a href="https://www.codecogs.com/eqnedit.php?latex=$a(u;X^l)&space;=&space;\arg&space;\max$&space;\sum_{l}^{i=1}[$y_u^{(i)}=y]w(i,u)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$a(u;X^l)&space;=&space;\arg&space;\max$&space;\sum_{l}^{i=1}[$y_u^{(i)}=y]w(i,u)" title="$a(u;X^l) = \arg \max$ \sum_{l}^{i=1}[$y_u^{(i)}=y]w(i,u)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=$w(i,u)$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$w(i,u)$" title="$w(i,u)$" /></a> - вес (степень важности) i-го соседа объекта u, неотрицателен, не возрастает по i.

<a href="https://www.codecogs.com/eqnedit.php?latex=\Gamma_y(u)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Gamma_y(u)" title="\Gamma_y(u)" /></a> - оценка близости объекта 'u' к классу 'y'(оценка степенм принадлежности).

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

Алгоритм kNN относит классифицируемый объект к классу в зависимости от k-ближайших соседей (обучающих объектов). 

<a href="https://www.codecogs.com/eqnedit.php?latex=$w(i,u)&space;=&space;\left&space;[&space;i&space;\leqslant&space;k&space;\right&space;]$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$w(i,u)&space;=&space;\left&space;[&space;i&space;\leqslant&space;k&space;\right&space;]$" title="$w(i,u) = \left [ i \leqslant k \right ]$" /></a>

Преимущества:

-менее чувствителен к шумуж;

-появился параметр k.

Возникает справедливый вопрос: как определить оптимально количество k? Ведь если взять слишком мало, или слишком много может оказаться, что погрешность выростет. 

## Реализация

```
colors <- c("setosa" = "red", "versicolor" = "green3",
"virginica" = "blue")
plot(iris[, 3:4], pch = 21, bg = colors[iris$Species],
col = colors[iris$Species])
euclideanDistance <- function(u, v)
  {
sqrt(sum((u - v)^2))
  }
sortObjectsByDist <- function(xl, z, metricFunction =
euclideanDistance)
  {
l <- dim(xl)[1]
n <- dim(xl)[2] - 1
distances <- matrix(NA, l, 2)
  for (i in 1:l)
  {
    distances[i, ] <- c(i, metricFunction(xl[i, 1:n], z))
  }
orderedXl <- xl[order(distances[, 2]), ]
return (orderedXl);
  }
NN <- function(xl, z, k)
orderedXl <- sortObjectsByDist(xl, z)
n <- dim(orderedXl)[2] - 1
classes <- orderedXl[1:k, n + 1]
counts <- table(classes)
class <- names(which.max(counts))
return (class) 
  }
colors <- c("setosa" = "red", "versicolor" = "green3",
"virginica" = "blue")
plot(iris[, 3:4], pch = 21, bg = colors[iris$Species], col
= colors[iris$Species], asp = 1)
z <- c(2.7, 1) 
xl <- iris[, 3:5] 
class <- kNN(xl, z, k=6) 
points(z[1], z[2], pch = 22, bg = colors[class], asp = 1) 
```

## KWNN

Алгоритм KWNN, в отличии от KNN отличается тем, что он учитывает не только ранг расстоянния <a href="https://www.codecogs.com/eqnedit.php?latex=(\rho&space;(u,X_u^{(1)})\leqslant&space;\rho&space;(u,X_u^{(2)})&space;\leqslant&space;\dots&space;\leqslant&space;\rho&space;(u,X_u^{(l)}))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(\rho&space;(u,X_u^{(1)})\leqslant&space;\rho&space;(u,X_u^{(2)})&space;\leqslant&space;\dots&space;\leqslant&space;\rho&space;(u,X_u^{(l)}))" title="(\rho (u,X_u^{(1)})\leqslant \rho (u,X_u^{(2)}) \leqslant \dots \leqslant \rho (u,X_u^{(l)}))" /></a> по убываюнию в качестве веса w(i,u), но и среднее расстоняие от k-ближайших объектов. Мы будет относить классифицируемый объект к тому классу, у которого среднее расстоние будет меньше. Таким образом качесвто классификации становиться лучше.

## Сравнение KNN и KWNN 

## Оптимизация числа соседей k (LOO):
Функционал скользящего контроля leave-one-out

<a href="https://www.codecogs.com/eqnedit.php?latex=LOO$(k,X^l)$&space;=&space;\sum_{i=1}^{l}$[a(x_i;X^l&space;/&space;\left&space;\{&space;x_i&space;\right&space;\},&space;\neq&space;y]&space;\rightarrow&space;$\min_{k}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?LOO$(k,X^l)$&space;=&space;\sum_{i=1}^{l}$[a(x_i;X^l&space;/&space;\left&space;\{&space;x_i&space;\right&space;\},&space;\neq&space;y]&space;\rightarrow&space;$\min_{k}" title="LOO$(k,X^l)$ = \sum_{i=1}^{l}$[a(x_i;X^l / \left \{ x_i \right \}, \neq y] \rightarrow $\min_{k}" /></a>

Алгоритм LOO выкидывает из выборки по одному обучающему элементу и смотрит правильный ли был ответ для i-го объекта. Если ответ был правильный добавляем 0, если неправилдьный, то 1. Проделывая такую операцию смотрим частоту ошибок при 1,2,3,..,k-1,k,k+1,...,n элементов. Это делается для того, чтобы посмотреть какое количество ближайших соседей нам нужно учитывать, чтобы отнести объект 'u' к тому или инному классу. 

