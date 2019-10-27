# ML0
## Метрические алгоритмы классификации

## <a href="https://www.codecogs.com/eqnedit.php?latex=$a(u;X^l)&space;=&space;\arg&space;\max$&space;\sum_{l}^{i=1}[$y_u^{(i)}=y]w(i,u)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$a(u;X^l)&space;=&space;\arg&space;\max$&space;\sum_{l}^{i=1}[$y_u^{(i)}=y]w(i,u)" title="$a(u;X^l) = \arg \max$ \sum_{l}^{i=1}[$y_u^{(i)}=y]w(i,u)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=$w(i,u)$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$w(i,u)$" title="$w(i,u)$" /></a> - вес (степень важности) i-го соседа объекта u, неотрицателен, не возрастает по i.

<a href="https://www.codecogs.com/eqnedit.php?latex=\Gamma_y(u)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Gamma_y(u)" title="\Gamma_y(u)" /></a> - оценка близости объекта 'u' к классу 'y'(оценка степени принадлежности).

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

```R
colors <- c("setosa" = "red", "versicolor" = "green3", "virginica" = "blue")
  plot(iris[, 3:4], pch = 21, bg = colors[iris$Species],
    col = colors[iris$Species])
      euclideanDistance <- function(u, v)
  {
    sqrt(sum((u - v)^2))
  }
sortObjectsByDist <- function(xl, z, metricFunction = euclideanDistance)
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
colors <- c("setosa" = "red", "versicolor" = "green3", "virginica" = "blue")
plot(iris[, 3:4], pch = 21, bg = colors[iris$Species], col = colors[iris$Species], asp = 1)
z <- c(2.7, 1) 
xl <- iris[, 3:5] 
class <- kNN(xl, z, k=6) 
points(z[1], z[2], pch = 22, bg = colors[class], asp = 1) 
```

## KWNN
<a href="https://www.codecogs.com/eqnedit.php?latex=w(i,u)=[i\leqslant&space;k,&space;w(i)];&space;\par&space;U(u;X^l,k)&space;=&space;arg&space;max&space;\sum\limits^k_{i=1}&space;[y_n^{(i)}=y]w(i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w(i,u)=[i\leqslant&space;k,&space;w(i)];&space;\par&space;U(u;X^l,k)&space;=&space;arg&space;max&space;\sum\limits^k_{i=1}&space;[y_n^{(i)}=y]w(i)" title="w(i,u)=[i\leqslant k, w(i)]; \par U(u;X^l,k) = arg max \sum\limits^k_{i=1} [y_n^{(i)}=y]w(i)" /></a>

Алгоритм KWNN, в отличии от KNN, учитывает не только ранг расстоянния <a href="https://www.codecogs.com/eqnedit.php?latex=(\rho&space;(u,X_u^{(1)})\leqslant&space;\rho&space;(u,X_u^{(2)})&space;\leqslant&space;\dots&space;\leqslant&space;\rho&space;(u,X_u^{(l)}))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(\rho&space;(u,X_u^{(1)})\leqslant&space;\rho&space;(u,X_u^{(2)})&space;\leqslant&space;\dots&space;\leqslant&space;\rho&space;(u,X_u^{(l)}))" title="(\rho (u,X_u^{(1)})\leqslant \rho (u,X_u^{(2)}) \leqslant \dots \leqslant \rho (u,X_u^{(l)}))" /></a> по убываюнию в качестве веса w(i,u), но и среднее расстоняие от k-ближайших объектов. Мы будет относить классифицируемый объект к тому классу, у которого среднее расстоние будет меньше. Таким образом качество классификации становиться лучше.

## Сравнение KNN и KWNN 
Покажем, на графике превосходство алготима KWNN над KNN. На картинке хорошо видно, что в слуячае с KNN алгоритм определил оъект ошибочно отнеся его к синиму, так как их количество (рангов) больше, а расстояние мы не учитытываем, в свою очередь KWNN учел расстояние и определил классифицируемый объект верно.

![Image alt](https://github.com/Shuregame/ML0/blob/master/KNN_vs_KWNN.jpg)


## Оптимизация числа соседей k (LOO):
Функционал скользящего контроля leave-one-out

<a href="https://www.codecogs.com/eqnedit.php?latex=LOO$(k,X^l)$&space;=&space;\sum_{i=1}^{l}$[a(x_i;X^l&space;/&space;\left&space;\{&space;x_i&space;\right&space;\},&space;\neq&space;y]&space;\rightarrow&space;$\min_{k}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?LOO$(k,X^l)$&space;=&space;\sum_{i=1}^{l}$[a(x_i;X^l&space;/&space;\left&space;\{&space;x_i&space;\right&space;\},&space;\neq&space;y]&space;\rightarrow&space;$\min_{k}" title="LOO$(k,X^l)$ = \sum_{i=1}^{l}$[a(x_i;X^l / \left \{ x_i \right \}, \neq y] \rightarrow $\min_{k}" /></a>

Алгоритм LOO выкидывает из выборки по одному обучающему элементу и смотрит правильный ли был ответ для i-го объекта. Если ответ был правильный добавляем 0, если неправилдьный, то 1. Проделывая такую операцию смотрим частоту ошибок при 1,2,3,..,k-1,k,k+1,...,n элементов. Это делается для того, чтобы посмотреть какое количество ближайших соседей нам нужно учитывать, чтобы отнести объект 'u' к тому или инному классу. 

## Парзеновское окно

Метод парзеновского окна использует весовую функцию w(i,u) как функцию не от ранга расстояния (как в KNN), а от расстояния. Зададим некоторый параметр h - ширина окна (напр. в двумерном пространстве это радиус круга).Бывают случаи, когда объект не лежит среди плотного распределения точек, в таком случае лучше сделать ширину с переменным значением.
Рассмотри некоторую функцию на промежутке [0,1], где по оси Ox мы определим ширину парзеновского окна. Внутри него лежат точки обучающей выборки, их вычисляем так:
Для того, чтобы сделать парзеновское окно с переменной шириной(пар. h), мы можем, например, поместить нашу 1 є K(r) на k+1 соседа, тогда для k+1 соседа вес будет нулевой, но для k-го будет лежать точно где-то в промежутке [0,1].
<a href="url"><img src="https://github.com/Shuregame/ML0/blob/master/Parsen_dlya_gip.png" height="500" width="760" ></a>


## Метод потенциальных функций

Так как функция расстояния - функция симметричная, можно рассматривать парзеновское окно с точки зрения объектов обучающей выборки.
Ширина окна будет соответсвенно привязывается не к классифицируемому объекту, а к объектам обучающей выборки. Зададим окрестности "h" некоторый потенциал, и будем говорить, что все элементы попадающие в окрестность данного объекта обучающей выборки будут принимать его класс. Если объект попал сразу в два окна, смотрим на потенциалы этих окон и расстояния до объектов обучающей выборки. Если мы посмотрим на график ниже, то увидим, что расстония от классифицируемого объекта до объектов классов примерно равны, но потенциалы значительно отличаются. Соответственно относим классифицируемый объект к классу "крестик".

<a href="https://www.codecogs.com/eqnedit.php?latex=a(u;X^l)=\arg&space;\max_{y\epsilon&space;Y}&space;\sum_{i=1}^l&space;[y_i=y]\gamma_i&space;K(\frac{p(u,x_i)}{h_i})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a(u;X^l)=\arg&space;\max_{y\epsilon&space;Y}&space;\sum_{i=1}^l&space;[y_i=y]\gamma_i&space;K(\frac{p(u,x_i)}{h_i})" title="a(u;X^l)=\arg \max_{y\epsilon Y} \sum_{i=1}^l [y_i=y]\gamma_i K(\frac{p(u,x_i)}{h_i})" /></a>

<a href="url"><img src="https://github.com/Shuregame/ML0/blob/master/Potencial.jpg" height="500" width="760" ></a>

## Отступ

Случается так, что обучающая выборка избыточна. К примеру у нас есть 1000 объектов, но нам будет достаточно и 50 для построения алгоритма. Вопрос в том какие 50 нам выбрать. Для этого вводим пониятие отступа, которое подразделяет объекты выборки по уровню точности оценки классифицируемого объекта. Для этого введем формулу отступа:

<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;M(x_i)=\Gamma_{yi}(x_i)-\max_{y\epsilon&space;Y&space;\setminus&space;y_{i}}\Gamma&space;_y(x_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;M(x_i)=\Gamma_{yi}(x_i)-\max_{y\epsilon&space;Y&space;\setminus&space;y_{i}}\Gamma&space;_y(x_i)" title="\large M(x_i)=\Gamma_{yi}(x_i)-\max_{y\epsilon Y \setminus y_{i}}\Gamma _y(x_i)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\Gamma_y(u)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Gamma_y(u)" title="\Gamma_y(u)" /></a> - оценка близости объекта 'u' к классу 'y'(оценка степенм принадлежности).

![Image alt](https://github.com/Shuregame/ML0/blob/master/Etaloni.png)

-Эталонные (Самые надежные, всегда оставляем)

-Неинформативные (При необходимости можно удалить из выборки)

-Пограничные (Классификация неустойчива)

-Ошибочные (Причина ошибки - плохая модель)

-Шумовые (Причина ошибки - плохие данные)

Понятно, что на нужно точно оставлять Эталонные объекты, и точно убирать Ошибочные и Шумы. В любом случае это улучшить алгоритм на первом этапе, а дальше нужно смотреть на соотношение других типов объектов 

# Алгоритм STOLP
Алгоритм STOLP служит для определения типов объектов обучающей выборки, по уровню достоверности.  

Алгоритм STOLP заключается в том, чтобы исключить выбросы, и возможно пограничные объекты, найти по одному эталону в каждом классе, и добавлять эталоны, пока есть отрицательные отступы. 

Вход:

<a href="https://www.codecogs.com/eqnedit.php?latex=X_l" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_l" title="X_l" /></a> — обучающая выборка;

<a href="https://www.codecogs.com/eqnedit.php?latex=\delta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta" title="\delta" /></a> — порог фильтрации выбросов;

<a href="https://www.codecogs.com/eqnedit.php?latex=l_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l_0" title="l_0" /></a> — допустимая доля ошибок;

Выход:

Множество опорных объектов <a href="https://www.codecogs.com/eqnedit.php?latex=\Omega&space;\subseteq&space;X^l;" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Omega&space;\subseteq&space;X^l;" title="\Omega \subseteq X^l;" /></a>

1: для всех <a href="https://www.codecogs.com/eqnedit.php?latex=x_i&space;\epsilon&space;X^l" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_i&space;\epsilon&space;X^l" title="x_i \epsilon X^l" /></a> проверить, является ли <a href="https://www.codecogs.com/eqnedit.php?latex=x_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_i" title="x_i" /></a> выбросом:

2: если <a href="https://www.codecogs.com/eqnedit.php?latex=M(x_i,&space;x_l)&space;<&space;\delta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?M(x_i,&space;x_l)&space;<&space;\delta" title="M(x_i, x_l) < \delta" /></a>

3: <a href="https://www.codecogs.com/eqnedit.php?latex=X^{l-1}&space;:=&space;X^l\&space;\backslash&space;\&space;\{x_i\};&space;l&space;:=&space;l-1;" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X^{l-1}&space;:=&space;X^l\&space;\backslash&space;\&space;\{x_i\};&space;l&space;:=&space;l-1;" title="X^{l-1} := X^l\ \backslash \ \{x_i\}; l := l-1;" /></a>

4: Инициализация: взять по одному эталону от каждого класса:

<a href="https://www.codecogs.com/eqnedit.php?latex=\Omega&space;:=&space;\{\arg\max_{x_i\epsilon&space;X^l_y}&space;M(x_i,&space;X^l)&space;|&space;\&space;y&space;\&space;\epsilon&space;\&space;Y\};" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Omega&space;:=&space;\{\arg\max_{x_i\epsilon&space;X^l_y}&space;M(x_i,&space;X^l)&space;|&space;\&space;y&space;\&space;\epsilon&space;\&space;Y\};" title="\Omega := \{\arg\max_{x_i\epsilon X^l_y} M(x_i, X^l) | \ y \ \epsilon \ Y\};" /></a>

5: пока <a href="https://www.codecogs.com/eqnedit.php?latex=\Omega&space;\neq&space;X^l;" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Omega&space;\neq&space;X^l;" title="\Omega \neq X^l;" /></a>

6:Выделить множество объектов, на которых алгоритм <a href="https://www.codecogs.com/eqnedit.php?latex=a(u;&space;\Omega)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a(u;&space;\Omega)" title="a(u; \Omega)" /></a> ошибается:

<a href="https://www.codecogs.com/eqnedit.php?latex=E&space;:=&space;\{x_i&space;\epsilon&space;X^l&space;\&space;\backslash&space;\Omega&space;:&space;M(x_i,&space;\Omega&space;<&space;0)&space;\};" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E&space;:=&space;\{x_i&space;\epsilon&space;X^l&space;\&space;\backslash&space;\Omega&space;:&space;M(x_i,&space;\Omega&space;<&space;0)&space;\};" title="E := \{x_i \epsilon X^l \ \backslash \Omega : M(x_i, \Omega < 0) \};" /></a>

7: если <a href="https://www.codecogs.com/eqnedit.php?latex=|E|&space;<&space;l_0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?|E|&space;<&space;l_0" title="|E| < l_0" /></a> то

8: выход;

9: Присоеденить к <a href="https://www.codecogs.com/eqnedit.php?latex=\Omega" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\Omega" title="\Omega" /> объект с наименьшим отступом:
  
<a> href="https://www.codecogs.com/eqnedit.php?latex=x_i&space;:=&space;\arg&space;\min_{x\epsilon&space;E}&space;M(x,\Omega);&space;\&space;\Omega&space;:=\Omega\&space;\cup&space;\&space;\{x_i\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_i&space;:=&space;\arg&space;\min_{x\epsilon&space;E}&space;M(x,\Omega);&space;\&space;\Omega&space;:=\Omega\&space;\cup&space;\&space;\{x_i\}" title="x_i := \arg \min_{x\epsilon E} M(x,\Omega); \ \Omega :=\Omega\ \cup \ \{x_i\}" /></a>

## Преимущества отбора эталонов
-сокращается число хранимых объектов;

-сокращается время классификации;

-объекты распределяются по величине отступов;

## Недостатки алгоритма 
-необходимость задавать параметр <a href="https://www.codecogs.com/eqnedit.php?latex=\delta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta" title="\delta" /></a> ;

-относительно низкая эффективность 
