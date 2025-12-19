### A Pluto.jl notebook ###
# v0.20.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ bad85a58-6c6f-11f0-2681-b7cbc064773f
begin
	using Pkg
	Pkg.activate()

	using PlutoUI

    using Random, Statistics, LinearAlgebra
    using Plots
    default(; legend=:top, framestyle=:box)

	using DataFrames

	TableOfContents(title="Contenido")
end

# ╔═╡ 20584b47-0106-4311-9c40-138af44285e7
md"# Introducción al aprendizaje automático supervisado"

# ╔═╡ 3b374eb4-0108-4fbd-8f4b-f882b4b2d383
md"En esta materia vamos a hacer una introducción a la Inteligencia Artificial (IA) aplicada a problemas de ingeniería, con foco en aprendizaje automático (Machine Learning, ML). El objetivo es que puedas entender, construir y evaluar modelos sencillos que hagan predicciones útiles en contextos reales (logística, manufactura, energía, finanzas, salud, etc.).

Trabajaremos con Julia, Pluto y la librería MLJ, alternando entre:

* intuición (juegos, reglas hechas a mano),
* implementación (algoritmos simples desde cero),
* uso profesional (pipelines con MLJ y herramientas SaaS).

Además, le prestaremos atención a la ética, fiabilidad y las posibles implicancias del uso de la IA.

Puntualmente, en este notebook hablaremos sobre algunas generalidades de IA y luego comenzaremos a tratar un tipo específico de problemas, los problemas de [**regresión**](https://es.wikipedia.org/wiki/An%C3%A1lisis_de_la_regresi%C3%B3n) mediante [**Aprendizaje Supervisado**](https://es.wikipedia.org/wiki/Aprendizaje_supervisado).

Este notebook continúa con una sección de configuración (necesaria para que se pueda ejecutar) y luego sigue con el contenido. Esa misma estructura se va a repetir en el resto de los notebooks.

Los paquetes a utilizar en este notebook son:

* PlutoUI
* Random
* Statistics
* LinearAlgebra
* Plots
* DataFrames
"

# ╔═╡ dcd19bd9-b27e-4fb5-a2b9-6d4d39358fb2
md"## Setup del notebook"

# ╔═╡ f4f54c42-1c1d-4b25-a350-1958eab8c2d6
md"## ¿Que es la IA?"

# ╔═╡ 05d04f8b-41d2-4242-ac0f-e463e516bbaf
md"La IA es un término paraguas que engloba técnicas que permiten a un sistema (un software) percibir, predecir y tomar decisiones en un contexto determinado. Dentro de la IA, el aprendizaje automático se centra en que los sistemas aprendan patrones a partir de datos en lugar de depender exclusivamente de reglas programadas a mano. La IA no se agota en el aprendizaje automático, sino que abarca a otras disciplinas, como la toma de decisiones mediante modelos (por ejemplo, los modelos de optimización vistos en Investigación Operativa)."

# ╔═╡ ec7a4883-d132-406e-be4b-535e28425924
md"## ¿Que es el aprendizaje automático?"

# ╔═╡ 30ab0a3a-ba33-4fd8-8803-859256d9f7a5
md"En ML buscamos una función que relacione entradas (características o features) con salidas (objetivo o target). A esa función la llamamos modelo.
El aprendizaje consiste en, dado una estructura general de un modelo, ajustar los parámetros del mismo para minimizar un error medido sobre ejemplos conocidos.

Cuando hablamos de ML, tenemos que tener en cuenta los siguientes conceptos:

* Entrada ($X$): variables descriptivas (p.ej., metros cuadrados, barrio, edad de una casa).
* Salida ($y$): valor a predecir (p.ej., precio).
* Modelo ($f(\theta)$): familia de relaciones funcionales con parámetros $\theta$.
* Pérdida (loss, $L$): penaliza la discrepancia entre predicción $\hat{y}$ y el valor real $y$.
* Costo ($J$): promedio de pérdidas sobre un conjunto de datos.

El desafío clave es la generalización: que el modelo funcione bien en datos nuevos, no solo en los que vio durante el ajuste.

Dentro del mundo del ML, tenemos varios tipos de aprendizaje, que podríamos clasificar en:

* Supervisado: tenemos pares $(X,y)$. Aprendemos a predecir $y$ (p.ej., precio, demanda, falla/no falla) en base a $X$.
* No supervisado: tenemos solamente $X$. Buscamos estructuras (p.ej., clústers, jerarquías).
* Por refuerzo: un agente aprende a actuar en un entorno para maximizar recompensas acumuladas.

En este curso nos vamos a focalizar principalmente en aprendizaje supervisado, porque es el punto de partida más directo para problemas de ingeniería.

"

# ╔═╡ 2634d4bc-9787-4ba1-af5c-1da8a024c29b
md"## Aprendizaje Supervisado"

# ╔═╡ 80e6264a-8ab5-4e0e-893b-040f59b93681
md"El objetivo del aprendizaje automático supervisado es estimar los parámetros $\theta$ de una función $y=f(X,\theta)$. Ok, ¿pero para que?. Bueno, lo que nos interesa, una vez conocidos los valores de $\theta$ usar la función $f$ para calcular el valor de $y$ que le correspondería a un valor de $X$.

Ufff, ¡que explicación mas densa! Útil para el que conoce del tema (ponele) pero inintelegible para un [no iniciado](https://medium.com/datos-y-ciencia/machine-learning-secretos-de-magia-revelados-182fe6ae28c3).

Vamos de nuevo. Hay algo que me gustaría saber, pero lo quiero saber antes de lo que lo sabría normalmente. Porque saberlo normalmente es **muy costoso**, o **muy difícil**, o los dos. Es decir, para saber algo, debo esperar hasta que ese algo suceda, o que sea visible, o que se presenten sus síntomas, etc. Pero hacer esto me genera costos de oportunidad, porque tener una estimación ahora de lo que va a pasar mañana me permite decidir mejor que esperar hasta mañana [a ver qué pasa](https://arxiv.org/abs/2503.00650). Esto es algo que parece raro, pero intuitivamente todos hacemos: 

* cuando tenemos algún pequeño dolor, vamos al médico y nos hacemos estudios, para detectar si tenemos algo mas grave y tratarlo antes que llegue a todo su esplendor;
* cuando el auto empieza a hacer algún ruido raro, lo llevamos al mecánico, no esperamos a que deje de funcionar;
* si tenemos una despensa o almacén, revisamos rutinariamente el stock de productos, no esperamos a que llegue un cliente para descubrir que no tenemos quesos y recién ahí hacer un pedido.

Contar con información sobre un evento antes que suceda, nos permite tomar mejores decisiones, e incluso responder preguntas del tipo [¿que pasaría sí?](https://visionedgemarketing.com/using-what-if-strategy-planning/)

La idea fundamental del aprendizaje supervisado, que es la misma de la inferencia estadística (sí, porque el aprendizaje automático no es más que estadística ultra potenciada por la aplicación de ciencias de la computación), es que, sacando una muestra de algunas características de un proceso, podemos inferir el valor de otras características del mismo proceso. Por ejemplo:

* a partir de las ventas de los últimos días, estimar cuanto voy a vender mañana,
* en base a las vibraciones de un equipo mécanico, detectar si hay algún engranaje roto, sin necesidad de desarmarlo,
* en base a una foto, determinar si una pieza está bien ensamblada, sin revisar toda la pieza manualmente, 
* en base a las características de una casa, estimar cuál será su precio de venta en el mercado inmobiliario.

En todos estos casos, la muestra que puede sacar de algunas características es menos costosa (o más fácil) que obtener el dato real que me interesa. 

Todo lo anterior funciona si realmente se cumplen algunos supuestos:

* Existe una relación funcional real entre lo que quiero averiguar (lo que quiero predecir) y las características que puedo recolectar; posiblemente exista una relación entre el tamaño de una casa y su precio, pero seguramente no existe ninguna relación entre la altura promedio de las personas que viven en la casa y el precio de la casa.
* Esa relación es más o menos constante. En otras palabras, el futuro se va a comportar de forma parecida al pasado.
* Puedo recolectar datos históricos del proceso, tanto de lo que quiero predecir como de las características a muestrear;es decir, en el caso de la casa, existe información disponible en la forma de, por ejemplo, un listado de casas vendidas en los últimos años, en el que se indique el precio y el tamaño de la casa, para cada una de las casas en el listado.
* Voy a seguir teniendo la posibilidad en el futuro de recolectar información sobre las características que quiero utilizar para predecir el dato que me interesa.

Cuando un [**científico de datos**](https://es.wikipedia.org/wiki/Cient%C3%ADfico_de_datos) quiere predecir, por ejemplo, el precio de una casa en función de sus características (tamaño, ubicación, antigüedad, etc) realiza dos cosas en paralelo, y cíclicamente:

* recolecta datos sobre precios de casas y sobre las características de dichas casas,
* analiza los datos junto con información de expertos, para determinar qué forma podría tener la relación entre el precio y las características, por ejemplo, la suposición de que el precio crece proporcionalmente al tamaño de la casa.

Partiendo de eso, con un conjunto de datos y un modelo (es decir, la relación funcional entre lo que quiero predecir y las características que voy a utilizar para predecirlo), se puede realizar el proceso de **entrenamiento**, [término antropomórfico pseudo-religioso que en realidad no implica nada más que estimar los mejores valores de los parámetros del modelo](https://medium.com/datos-y-ciencia/machine-learning-secretos-de-magia-revelados-182fe6ae28c3). ¿Y para que calculamos los parámetros? Bueno, porque con el modelo, los valores de sus parámetros y los valores de las características de un nuevo [elemento muestreal](https://es.wikipedia.org/wiki/Muestreo_(estad%C3%ADstica)), puedo estimar un valor para lo que quiero predecir para dicho elemento.

Traduzcamos todo a la [lengua de Cervantes](https://es.wikipedia.org/wiki/Miguel_de_Cervantes) (pero matematizada) usando el ejemplo de las casas. Si supongo que el precio de las casas ($y$) depende proporcionalmente de su tamaño medido en metros cuadrados ($x$), estoy suponiendo que la relación funcional es del tipo $y=m \cdot x + b$ (es decir lineal). En base a eso, recolecto un conjunto de datos históricos de $n$ casas, en el cual para cada casa tenga listado el precio al cual se vendió y los metros cuadrados de las casas, es decir, tuplas $(x,y)$. Luego, al **entrenar** el modelo, calculo los valores de $m$ y $b$. Supongamos que obtengo que $m=50000$ y $b=10000$. Con esos valores de $m$ y $b$, la próxima vez que quiera calcular el precio de una casa, solamente tengo que averiguar cuántos metros cuadrados tiene, multiplicar por $m=50000$ y sumarle $b=10000$. Por ejemplo, si la casa tuviera $100 m²$, la estimación de su precio sería $\hat{y}=50000 \cdot 100 + 10000$. Noten que al precio le agregamos un sombrerito ($\hat{y}$), lo que indica que no es el valor real, sinó una estimación. El valor real lo conoceremos al realizar la compra-venta de la casa, y dicho valor real será $y=\hat{y} + e$, es decir, el valor estimado mas el error de la estimación. Obviamente, que el error de la estimación no lo conocemos hasta no conocer el valor real, pero, sin importar esto, para que el modelo sea útil, vamos a querer que este error (desconocido al momento de realizar la estimación o predicción) sea lo más pequeño posible. No solamente para esta casa, sino para todas las casas. En otras palabras, cuando entrenamos el modelo, buscamos encontrar los parámetros del modelo que minimicen el error promedio. Es decir, **minimizamos una función del error del modelo**.

"

# ╔═╡ 36cd88c6-6a9c-4fb4-94d4-4df67b467b18
md"""
!!! info "Actividad para el alumno"
	Traten de explicar esto con sus palabras. O mejor, traten de armar su propio ejemplo, que no tenga nad que ver con los ejemplos descriptos hasta ahora.
"""

# ╔═╡ ed11b0db-0864-448e-91fb-33b651b2a018
md"### ¿Que significa supervisado?"

# ╔═╡ 2556052e-b072-4b7c-b856-563d1ad8ccb8
md"""
En aprendizaje supervisado tenemos ejemplos históricos con la forma pares $(X,y)$:

*_ $X$ son las características (**features**): lo que sí sabemos hoy —precio, día de la semana, temperatura, horas de uso de una máquina, etc. Usualmente, es un vector de características, no una única característica.

*_ $y$ es el objetivo (**target**): lo que queremos predecir antes de que ocurra —demanda de mañana, probabilidad de falla la semana que viene, si un pedido llegará tarde.

El nombre es **Supervisado** porque existe un **supervisor** que nos dice cuál fue la salida correcta $y$ para cada entrada $X$ en el pasado. Con esos ejemplos, ajustamos los parámetros $\theta$ de una función $f$ para calcular la estimación del target: $\hat{y}=f(X;θ)$.
"""

# ╔═╡ d6176296-d980-48cc-b1fe-b4356e7d76ff
md"### Dos _familias_ de problemas de aprendizaje supervisado"

# ╔═╡ 101f7fb3-ddc3-4f1e-93d0-f171f3383785
md"Si prestaron atención a los ejemplos que fuimos dando de aprendizaje automático, capaz que una cosa les hizo ruido: predecir el precio de algo es, superficialmente, similar a predecir las ventas de mañana, pero no tiene nada que ver con predecir si un equipo se está por romper. Digo, en los dos primeros casos, se predice un número, pero en el tercero, es mas bien una condición, una característica. Lo mismo pasa con el ejemplo de detección de una enfermedad. Y si, son cosas diferentes. De hecho, los problemas de aprendizaje supervisado se agrupan en dos grandes familias:

* Problemas de **regresión**: en los cuales lo que quiero estimar es un número real (real en el sentido matemático, obvio).
* Problemas de **clasificación**: cuando lo que quiero predecir es una etiqueta o clase (enfermo/no-enfermo, niño/joven/adulto, etc.).

En este notebook nos vamos a seguir tratando los problemas de regresión, dejando los de clasificación para un siguiente notebook.

"

# ╔═╡ 5f6b7a8a-b329-416f-bef8-c16e650ee573
md"### ¿Y de donde sale _f_?"

# ╔═╡ 5106fa31-4f1a-4c53-b681-9624b8c52f12
md"Bueno, en la caja de herramientas del científico de datos hay un montón de modelos para usar: lineales, árboles de decisión, ensembles (Random Forest, Gradient Boosting), SVM, redes neuronales, etc. Cada familia trae sus parámetros $\theta$ y su manera de representar relaciones.

Para poner algún ejemplo minimalista, tenemos los siguientes dos modelos de ejemplo:

* Lineal: $\hat{y} = \theta_{0} + \theta_{1} \cdot tamaño + \theta_{2} \cdot habitaciones$
* Árbol: reglas tipo “si tamaño > 100 y habitaciones = 2 → precio = 1000000”.

En el primer caso, entrenar es calcular los valores de $\theta_{0}$, $\theta_{1}$ y $\theta_{2}$, en el segundo caso es definir el esquema de comparaciones (que variable comparar con que valor en el primer paso, en base a los resultados, que otra varaible comparar con que otro valor, y así sucesivamente).

"

# ╔═╡ 3c34fb87-3651-4d3f-bc2f-fbfcf477d779
md"### FAQ"

# ╔═╡ 6abd3625-b1ce-496a-a383-3d05fe0ae8e4
md"""
* **¿No era que los algoritmos de ML aprenden de los datos?, al final tengo que hacer casi todo yo**
 * Exacto, no *aprenden*. El término es mas histórico que otra cosa, y funciona como una [buzzword](https://en.wikipedia.org/wiki/Buzzword). Digamos que vende mas decir _vamos a implementar una IA para predecir las ventas_ que decir _vamos a ajustar una recta y tratar que mas o menos le pase cerca a los puntos_.
* **Pero si tengo que andar pensando como están vinculados los datos y armar un modelo, ¿para que quiero usar ML?**
 * Es cierto, en el ejemplo de las casas hay que definir la forma en que se relacionan los datos, linealmente, por ejemplo, y solo usamos los datos para estimar los parámetros. O sea, podría haberlo hecho **sin** ML. El tema es que, quizás, la cantidad de ventas que tenga mañana se puede estimar en base a una relación lineal respecto a las ventas de los últimos días. Y que la dureza de una pieza metálica luego de un proceso de [recocido metalúrgico](https://es.wikipedia.org/wiki/Recocido) también se pueda estimar como una relación lineal respecto de la dureza inicial de la pieza y la configuración del horno. Como el modelo genérico es el mismo, puedo utilizar los mismos algoritmos de optimización para calcular el valor de los parámetros en base a datos históricos, sin necesidad de armar _a mano_ la ecuación de predicción para cada uno de los problemas. La idea es, si los datos se relacionan en forma similar, uso los mismos algoritmos para estimar parámetros; los creo una vez para resolver un problema genérico, lo uso en muchos problemas distintos.

"""

# ╔═╡ a7414592-a73e-45c1-95a4-08eaddf52b71
md"## Modelo de regresión lineal para problemas de regresión"

# ╔═╡ 3d59c3b1-b26f-49b4-952b-094e92057055
md"Muchas veces regresión en el título, ¿no? Bueno, aclaremos un poco. En esta sección vamos a comenzar a explorar la _familia de modelos_ de [**regresión lineal**](https://es.wikipedia.org/wiki/Regresi%C3%B3n_lineal) para resolver [_problemas de regresión_](https://www.ibm.com/think/topics/classification-vs-regression). O sea, vamos a utilizar un modelo con estructura de [**función lineal**](https://en.m.wikipedia.org/wiki/Linear_function) para resolver el problema de predecir el valor de una variable numérica.

El modelo de _regresión lineal_ es el modelo que fuimos poniendo de ejemplo en gran parte de este notebook: multiplico cada **feature** por un _coeficiente_, sumo los productos y le sumo un _termino independiente_. El resultado de esa cuenta es el valor predicho del **target**. El entrenamiento del modelo sería calcular el valor óptimo de los _coeficientes_ y del _termino independiente_.

Matemáticamente, un modelo de regresión lineal es un modelo de la forma:

$\hat{y}=\theta_{0} + \sum_{j=1}^{m}\theta_{j}x_{j}$

Donde $m$ es la cantidad de **features** que voy a utilizar para predecir $y$. Entonces, si tengo 3 features:

$\hat{y}=\theta_{0} + \theta_{1}x_{1}+ \theta_{2}x_{2}+ \theta_{3}x_{3}$

Una forma común de representarlo es en forma matricial. Si $\theta$ es el vector columna de componentes $(\theta_{1}, ..., \theta_{m})$ y $X$ el vector columna de componentes $(x_{1}, ..., x_{m})$, entonces:

$\hat{y}=\theta_{0} + \theta^{T}X$

_(la T significa transpuesta; vector fila por vector columna me da como resultado un escalar)_

"

# ╔═╡ 9930908a-4c0e-43da-8e4e-4849c578a611
md"""
!!! info "Actividad para el alumno"
	Armá un ejemplo de un vector columna $\theta$, un vector columna $X$ (ambos del mismo tamaño) y un valor de $\theta_{0}$. Con esos valores, tratá de hacer el cálculo manual para ver cuando daría $\hat{y}$
"""

# ╔═╡ 6d3a1038-29e3-4ef6-bb49-c0f1520cc801
md"Predecir, esto es, aplicar la fórmula, es la parte fácil. Cuando tengamos que usar el modelo, solamente reemplazamos las $\theta_{j}$ por sus valores, las $x_{j}$ por los valores que muestreamos, calculamos y listo. La multiplicación y la suma son problemas ya resueltos desde la época en que los [indios compartieron conocimiento con los persas](https://es.wikipedia.org/wiki/Numeraci%C3%B3n_indo-ar%C3%A1biga). El quid de la cuestión es cómo calcular los valores de las $\theta_{j}$. Y eso lo vamos a hacer resolviendo un problema de optimización.
"

# ╔═╡ 77d618a7-4394-483b-bd37-ffebc31ec2b5
md"""
!!! tip "Para tener en cuenta"
	Cuando piensen que el ML se inventó ayer y es un superavance de nuestros tiempos, recuerden que, hace un par de añitos, [Gauss](https://es.wikipedia.org/wiki/Carl_Friedrich_Gauss) (si, el mismo de, digamos, casi todas las matemáticas) [ajustó, a mano, una regresión lineal](https://www.youtube.com/watch?v=u95K_BBDLhI) para calcular la órbita de [Ceres](https://es.wikipedia.org/wiki/Ceres_(planeta_enano)). Usó [mínimos cuadrados](https://en.m.wikipedia.org/wiki/Least_squares), nosotros vamos a utilizar otro método.
"""

# ╔═╡ 2b8fad25-c174-4011-b9d1-7a9c98078f85
md"### ¿Como calculamos los $\theta$?"

# ╔═╡ d27f9dc4-f9e6-4e9d-b513-7516c1dea5a5
md"O, lo que es lo mismo, ¿como entrenamos nuestra regresión a partir de datos? Bueno, veámoslo con el ejemplo de las casas. Supongamos que, en nuestra aventura para poder predecir el precio de mercado de una casa, recolectamos un montón de información del mercado inmobiliario, la cual resumimos en la siguiente tabla:
"

# ╔═╡ d4368425-74ea-4be4-857a-c9a0306ff70b
begin
	datos_casas_01 = DataFrame(
		m2=[92,97,76,94,66,68], 
		precio=[275,426,281,290,251,304])
end

# ╔═╡ 82e405e1-af1d-4765-a8e1-0c6c5808043c
md"Si, es cierto, no recolectamos un mooontoooooon de datos, pero para arrancar quizás sirvan. Sabemos que es probable que a más grande la casa (más metros cuadrados, _m2_) mayor sea el precio. Es probable, pero quizás no pasa siempre, puede haber otros factores que hacen al precio de las casas. Pero por ahora, lo único que pudimos relevar es eso, así que evitemos la [parálisis por análisis](https://es.wikipedia.org/wiki/Par%C3%A1lisis_del_an%C3%A1lisis). Al supuesto anterior lo podría expresar, matemáticamente, mediante:

$\hat{precio} = \theta_{0} + \theta_{1} \cdot m2$

Simplifiquemos la nomenclatura y llamamemos $y$ al _precio_ y $x_{1}$ a los _m2_. Entonces:

$\hat{y} = \theta_{0} + \theta_{1} \cdot x_{1}$

Nuestra estimación del precio será igual a la cantidad de metros cuadrados de la casa por un coeficiente (el cual nos dice cuanto vale un metro cuadrado) sumado a un precio base (común a todas las casas). 
"

# ╔═╡ 05915b04-665f-4a1c-9d14-0166f4b8e3a4
md"""
!!! tip "Para tener en cuenta"
	El modelo es bastante lógico y razonable, pero recuerden que todos lo modelos tienen su límite. Por ejemplo, en el extremo, en este modelo, una casa con $0$ metros cuadrados se seguiría vendiendo a $\theta_{0}$, lo cual no luce razonable. Pero, para el rango de metros cuadrados relevados, puede ser una buena aproximación..
"""

# ╔═╡ 7df3ed5c-59da-472d-8f0f-f678b4f3916b
md"Nuestro $\hat{y}$ es una estimación, no es el valor real de la casa ($y$). Si mi modelo es bueno y está bien entrenado, $\hat{y}$ puede que esté muy cercano a $y$, puede que sea igual en algún caso, pero probablemente no sea igual para absolutamente todas las casas que se pudieran vender en todo el mercado. En otras palabras, siempre voy a tener un error $e$ en mi estimación, el cual se calcula como:

$e = \hat{y} - y$

Un buen entrenamiento me devolvería valores de $\theta_{0}$ y $\theta_{1}$ que hagan que ese error sea lo más chico posible. Un enfoque válido para calcular dichos parámetros es intentar minimizar el error $e$. Ahora bien, no me sirve minimizar el error de una casa, debería tratar de minimizar el error $e$ para todas las casas, o al menos para todas las casas de mi muestra. Tengo que buscar una función que me mida el error $e$ de todas las casas de mi muestra. Una forma posible sería promediar los errores. Ese indicador, el promedio de los errores, se conoce como **error medio** (mean error, o **ME**):

$ME = \frac{1}{n} \sum_{i=1}^{n}e_{i} = \frac{1}{n} \sum_{i=1}^{n}(\hat{y}_{i} - y_{i})$

Podríamos intentar entonces buscar los valores de $\theta_{0}$ y $\theta_{1}$ que minimizan **ME**. Pero estaríamos en un problema, los errores negativos se podrían cancelar con los positivos. Por ejemplo, si $\hat{y_{1}}= 0$, $\hat{y_{2}}=10$, $y_{1}=5$ y $y_{2}=5$, entonces:

$e_{1} = \hat{y_{1}} - y_{1} = 0 - 5 = -5$
$e_{2} = \hat{y_{2}} - y_{2} = 10 - 5 = 5$
$ME = \frac{1}{2} \cdot (-5+5)=\frac{1}{2} \cdot 0 = 0$

O sea, el ME nos estaría diciendo que nuestras estimaciones fueron perfectas, cuando en realidad sabemos que no fue así. Necesitamos un indicador que no cancele errores positivos con negativos. Una opción, en vez de usar el error $e$, es usar el valor absoluto de $e$ para crear el **error absoluto medio** (mean absolute error, **MAE**):

$MAE = \frac{1}{n} \sum_{i=1}^{n}|e_{i}| = \frac{1}{n} \sum_{i=1}^{n}|\hat{y}_{i} - y_{i}|$

Otra opción el trabajar con los cuadrados del error (porque $+ \cdot + = +$ y $- \cdot - = +$), para crear el **error cuadrático medio** (mean square error, **MSE**):

$MSE = \frac{1}{n} \sum_{i=1}^{n}(e_{i})^{2} = \frac{1}{n} \sum_{i=1}^{n}(\hat{y}_{i} - y_{i})^{2}$

"

# ╔═╡ 1ca896d7-dace-4843-a6ac-6f117d6f2673
md"""
!!! tip "Para tener en cuenta"
	A veces hablamos de la **raíz cuadrada del error cuadrático medio** (root mean square error, **RMSE**) porque el MSE eleva las unidades de medida del error al cuadrado. El RSME las devuelve a su magnitud original. Por ejemplo, el MSE del precio está en unidades de precios al cuadrado, el RSME está en unidades de precio.
"""

# ╔═╡ 14b32b00-742e-414a-8dc2-43959e3771dd
md"""
!!! info "Actividad para el alumno"
	¿Cuanto vale el MAE, el MSE y el RMSE del ejemplo dado?
"""

# ╔═╡ 790a2224-2741-4b76-8fcb-8df84462f15f
details(md"""Quiero ver la respuesta""", md"""
		
$MAE = \frac{1}{2}(|0-5| + |10-5|) = \frac{1}{2}(5+5) = 5$ 
$MSE = \frac{1}{2}((0-5)^{2} + (10-5)^{2}) = \frac{1}{2}(25+25) = 25$ 
$RMSE = \sqrt{\frac{1}{2}((0-5)^{2} + (10-5)^{2})} =  \sqrt{\frac{1}{2}(25+25)} = 5$ 
""")

# ╔═╡ 9f0d3ddb-7871-4165-89f0-60c5ce3ebbe4
md"Tanto el MSE como el MAE tienen sus pros y contras. El MAE pondera igual errores grandes y pequeños, mientras que el MSE le da mas importancia a los errores grandes (lo cual es bueno si buscamos minimizar errores). Por otro lado, el MAE es bastante insensible a la aparición de datos atípicos, mientras que el MSE suele variar bastente ante la inclusión de un data atípico (lo cual es malo si buscamos minimizar errores). [Difícil decisión](https://www.youtube.com/watch?v=pKSpl9ui7yY)."

# ╔═╡ d3ec59b7-9051-44d3-a47a-4ddea0cde918
md"""
!!! info "Actividad para el alumno"
    Bueno, ya vimos bastante. Ahora, en el contexto de la predicción del precio de las casas, identifiquen: entrada, salida, modelo, función de pérdida y función de costos.
"""

# ╔═╡ 863b65b2-a825-4770-be45-be80f171e413
details(md"""Quiero ver la respuesta""", md"""
		
La entrada son los metros cuadrados de cada caso. La salida es el precio. El modelo es una regresión lineal. Respecto de la pérdida y la función de costos, vimos tres alternativas:

* Alternativa 1: usamos el error como pérdida y el ME como función de costo.
* Alternativa 2: usamos el valor absoluto del error como pérdida y el MAE como función de costo.
* Alternativa 3: usamos el cuadrado del error como pérdida y el MSE como función de costo.
		
""")

# ╔═╡ 3967429e-eb9d-44f4-b3cc-b4fc9f3500b0
md"### ¡Manos a la obra!, calculemos $\theta_{0}$ y $\theta_{1}$"

# ╔═╡ 9fae6ca9-9946-4724-b3fe-17c839a6522a
md"Intentemos calcular manualmente los valores óptimos de $\theta_{0}$ y $\theta_{1}$. O, al menos, midamos el error para diferentes valores de estos dos parámetros.

El gráfico a continuación es interactivo (si estás ejecutando el notebook en Pluto, no si lo ejecutás como HTML estático). Aprovechando que nuestro modelo es una función con una única variable independiente ($x_{1}$) y una única variable dependiente ($y$), podemos graficar la recta asociada a distintos valores de nuestros parámetros.

En el gráfico podemos ver los puntos asociados a las muestras de metros cuadrados y precios. Moviendo los desplazadores de _pendiente_ y _ordenada_ le damos valores a los parámetros del modelo, _cambiando_ en consecuencia la posición e inclinación de la recta. A medida que la recta cambia, se actualizan los valores de **ME**, **MAE** y **RMSE**.

El gráfico también muestra los errores para cada una de las estimaciones mediante líneas que unen el punto (dato real) con la recta (estimación mediante el modelo). Para este modelo, como $y$ se representa en el eje vertical, el error se muestra como líneas verticales.

Opcionalmente, se puede generar un nuevo conjunto de datos (no relacionado con los anteriores) al azar, jugando con los controles del apartado _Datos a ajustar_. Si _Generar nuevos datos_ está activado, jugando con los valores de [_Semilla_](https://es.wikipedia.org/wiki/Semilla_aleatoria), _Cantidad de puntos_ y _Sigma_ podemos armar un nuevo conjunto de datos aleatorios.
"

# ╔═╡ 328e02e5-948c-400e-be1f-2614aaf5e270
md"""
##### Datos a ajustar:

Generar nuevos datos: $(@bind generar_nuevos_datos CheckBox(false))
"""

# ╔═╡ b974ef0c-d4a0-4d0a-a6aa-d54c6d74ae76
if generar_nuevos_datos
	md"""
Semilla: $(@bind seed Slider(1:999; default=42, show_value=true))

Cantidad de puntos (n): $(@bind n Slider(2:200; default=80, show_value=true))

Sigma: $(@bind sigma Slider(0.0:0.1:1.0; default=0.50, show_value=true))
	"""
else
	md"<-- No estamos generando nuevos datos, se utiliza el ejemplo previo -->"
end

# ╔═╡ e4acf6e2-91b9-42dd-9f8f-c8090bc0791c
md"""
##### Recta de ajuste
Ordenada ($\theta_{0}$): $(
if generar_nuevos_datos
    @bind b Slider(-20.0:0.1:20.0, default=0.0, show_value=true)
else
    @bind b Slider(0:1:(2*maximum(datos_casas_01.precio)), default=0.0, show_value=true)
end
)

Pendiente ($\theta_{1}$): $(
if generar_nuevos_datos
    @bind m Slider(-3.0:0.05:3.0, default=1.0, show_value=true)
else
    @bind m Slider(-30.0:0.05:30.0, default=0.0, show_value=true)
end
)

"""

# ╔═╡ 50b4bc69-cff5-4420-a077-039e6fa2309c
md"""
##### Configuración

Mostrar residuos: $(@bind mostrar_residuos CheckBox(true))
"""

# ╔═╡ 72ca99cc-6fc7-41c5-a57c-53abaa6cfebc
begin
	if generar_nuevos_datos
	    # Modelo "verdadero" para generar datos
	    a_true, b_true = 2.0, 1.5   # y = a_true + b_true*x + ruido
	
	    Random.seed!(seed)
	    x = sort(10 .* rand(n))    # x en [0, 10]
	    y = a_true .+ b_true .* x .+ sigma .* randn(n)
	else
		# Uso el dataframe de precios de casas
	    x = datos_casas_01[:,"m2"]   
	    y = datos_casas_01[:,"precio"]   
	end


	# Recta de ajuste
    ŷ = b .+ m .* x
    resid = y .- ŷ
    mae  = mean(abs.(resid))
    rmse = sqrt(mean(resid.^2))
	me = mean(resid)

	md"<-- Aquí hay una celda oculta con los cálculos necesarios para graficar. -->"
end

# ╔═╡ 8c0d6ee2-be75-4c73-84d7-0a0fbf1d9d9a
begin
    # Recta suave para dibujar
    xs = range(extrema(x)..., length=200)
    yline = b .+ m .* xs

    p = scatter(x, y;
        label="Datos",
        markersize=5,
        xlabel="x",
        ylabel="y",
        title="ME =  $(round(me, digits=2))   |    MAE = $(round(mae, digits=2))   |   RMSE = $(round(rmse, digits=2))")

    plot!(p, xs, yline; label="Recta (y = $(b) + $(m)·x)", lw=3)

    if mostrar_residuos
        # Segmentos verticales de residuo: (x_i, y_i) -> (x_i, ŷ_i)
        # Construimos una serie con NaN para cortar entre segmentos
        xs_seg = Vector{Float64}()
        ys_seg = Vector{Float64}()
        for i in eachindex(x)
            push!(xs_seg, x[i]); push!(ys_seg, y[i])
            push!(xs_seg, x[i]); push!(ys_seg, ŷ[i])
            push!(xs_seg, NaN);  push!(ys_seg, NaN)
        end
        plot!(p, xs_seg, ys_seg; label="Residuos", lw=1, color=:gray, alpha=0.8)
    end

    p
end

# ╔═╡ 5024f788-89b7-481f-9bbd-800eb60ab478
md"""
!!! info "Actividad para el alumno"
	Encuentren algún caso en que ME de un valor cercano a cero, mientras que el MAE y el RMSE den valores altos.
"""

# ╔═╡ 55adad7c-4c66-4330-b63f-cec456947ef7
details(md"""Quiero ver la respuesta""", md"""

Para estos conjuntos de datos, hay varias formas de lograr esto:
		
* Una opción sería hacer una recta con pendiente igual a cero y ajustar la ordenada hasta que el ME de cercano a cero.
* Otra opción sería dar a la pendiente un valor negativo y ajustar la ordenada hasta que el ME de cercano a cero. 
		
""")

# ╔═╡ 9c4a1d2f-e919-4d92-825c-ecee3e2de8c4
md"### Entrenando nuestro modelo"

# ╔═╡ e5639789-9cec-4e28-b84f-168d752032a2
md"Ok, ya calcularon los parámetros de la regresión lineal a mano, pero es obvio que el método manual no escala, o sea, no nos va a servir cuando tengamos muchas mas variables. Tenemos que buscar una manera de decidir, automáticamente, cuales son los mejores valores de los parámetros de la regresión. Es decir, calcular que valores deberían tener los parámetros para que la medida del error sea lo mas chica posible. ¡Eso parece ser un problema de optimización!

Podemos pensar el entrenamiento (o sea, el cálculo de los parámetros de nuestro modelo) como la resolución de un problema de optimización, en el cual queremos calcular los valores de los parámetros que minimizan la función de costo. Si nuestro función de costos es el MSE, tenemos el siguiente problema:

$Min \ J(\theta) = \frac{1}{2 \cdot n}\sum_{i=1}^{n}(y_{i} - \hat{y}_{i})^2$

Donde $\hat{y}_{i}$ es la predicción para el elemento $i$, es decir $\hat{y}_{i} = \theta_{0} + \theta_{1}x_{i}$.


"

# ╔═╡ e30f36d1-39b5-492e-bee1-7de0b97f0872
md"""
!!! info "Actividad para el alumno"
	¿Por que minimizamos el MSE y no el RMSE?
"""

# ╔═╡ d1ec2d81-3bef-463b-a36b-bbf18073479b
details(md"""Quiero ver la respuesta""", md"""

Los valores de $\theta_{0}$ y $\theta_{1}$ que minimizan el RSME son los mismos que minimizan el MSE (el mínimo de la raiz _sucede_ cuando el radicando es mínimo). Y, al momento de optimizar, el MSE es una función _mas fácil_ que el RSME.
		
""")

# ╔═╡ 605dfe66-f7a4-4228-80ce-fecbdcf44974
md"""
!!! info "Actividad para el alumno"
	¿Cambia algo tener un $2$ en el denominador?
"""

# ╔═╡ 61021325-abea-4b12-ac44-0a6af16751a3
details(md"""Quiero ver la respuesta""", md"""

Los valores de las variables que minimizan la función seguiran siendo los mismos. Así que no, no afecta. Pero nos va a ayudar mas tarde.
		
""")

# ╔═╡ 2d0f88f1-bed1-4816-809c-fc224d6a8959
md"Ahora bien, ¿como lo minimizamos? Bueno, para una regresión lineal canónica, como la que estamos viendo, podemos apelar al método de los [mínimos cuadrados](https://es.wikipedia.org/wiki/M%C3%ADnimos_cuadrados), un viejo conocido de la estadística, que encima nos garanztiza optimalidad en este caso. Pero acá vamos a hacer algo diferente, vamos a utilizar una heurística basada en [cálculo](https://es.wikipedia.org/wiki/C%C3%A1lculo) que nos va a servir para tratar funciones de costos mas complejas mas adelante. El método a utilizar se llama [descenso por gradiente](https://es.wikipedia.org/wiki/Descenso_del_gradiente).

El descenso por gradiente es un método de minimización de funciones (no sujetas a restricciones) que, como su nombre lo indica, se basa en el cálculo del gradiente de la función a optimizar. El gradiente de una función,de $J(\theta)$ en nuestro caso, es el vector cuyas componentes son las derivadas parciales de la función:

$\nabla{J(\theta)} = (\frac{\partial J}{\partial \theta_{0}}, \frac{\partial J}{\partial \theta_{1}})$

El gradiente nos dice cual es la _dirección_ en la cual la función crece mas rápido. Entonces, el $-\nabla{J(\theta)}$ nos dice cual es la dirección de máximo decrecimiento. Sabiendo esto, podemos plantear un algoritmo de optimización con la siguiente estructura:

1. Generamos una coordenada $(\theta_{0}, \theta_{1})$ inicial al azar. Esa es nuestra _coordenada actual_.
2. Calculamos el gradiente en la coordenada actual. Si es muy cercano al vector nulo, finalizamos. Si no, seguimos al paso 3.
3. Nos movemos en la dirección contraria al gradiente, una fracción $\alpha$ del módulo del gradiente.
4. Hacemos que la nueva coordenada sea la _coordenada actual_.
5. Si alcanzamos la cantidad máxima de iteraciones, terminamos. Si no, volvemos al paso 2.

O sea, actualizamos las componentes moviendonos iterativamente en la dirección del gradiente. En cada iteración:

$(\theta_{0},\theta_{1})  \leftarrow (\theta_{0},\theta_{1}) - \alpha \cdot \nabla{J(\theta)}$

El valor que le demos a $\alpha$, llamado _tasa de aprendizaje_ (learning rate) regula que tan rápido _aprende_ el algoritmo.

Lo importante acá es calcular $\nabla{J(\theta)}$. Para la $J$ que elegimos, tenemos que:

$\frac{\partial J}{\partial \theta_{0}} = \frac{1}{n}\sum_{i=1}^{n}[y_{i} - (\theta_{0} + \theta_{1}x_{i})]$

$\frac{\partial J}{\partial \theta_{1}} = \frac{1}{n}\sum_{i=1}^{n}[y_{i} - (\theta_{0} + \theta_{1}x_{i})] \cdot x_{i}$

Repitiendo los pasos 1 a 5 iterativamente, tenemos calculados nuestro parámetros.

"

# ╔═╡ de2a0723-72a8-4c13-8695-22666e599b80
md"""
!!! info "Actividad para el alumno"
	¿Se ve ahora por que habíamos agregado un $2$ al denominador?
"""

# ╔═╡ 9debc0a2-4874-4a34-bcf4-b8d950cb8a87
md"El descenso por gradiente viene en varios sabores. En cada iteración, podemos calcular $\nabla{J(\theta)}$ en la forma explicada, usando los $n$ datos que tenemos disponibles. O, podemos usar la versión conocida cómo **descenso por gradiente estocástico**, en la que en cada iteración elegimos un punto $i$ de nuestros $n$ puntos disponible, al azar, y calculamos el gradiente con dicho punto. Esta versión es, obviamente, mas rápida por cada iteración, pero es menos precisa (aunque puede ser buena si tenemos muchas _mesetas_). En el medio tenemos **descenso por gradiente por lotes** (o batch), en el cual en cada iteración se seleccionan, usualmente al azar, $m<n$ puntos, y se calcula el gradiente con estos. En todos los casos, cada vez que procesamos una cantidad de puntos igual a $n$ (en una iteración en el método convencional, en $n$ iteraciones en el método estocástico, en $\frac{n}{m}$ en el método batch), se dice que transcurrió una **epoch**."

# ╔═╡ b42d04c3-d810-410e-9898-62afa321384a
md"#### Tiempo de jugar"

# ╔═╡ 899c3944-4058-4fe5-a53a-fcb7cc8c34a1
md"A continuación hay una implementación _casera_, desde cero, sin utilizar ninguna biblioteca, del algoritmo de descenso por gradiente para resolver nuestro problema de ajustar un modelo de estimación de precios de casas. Jugando con los desplazadores pueden modificar la tasa de aprendizaje, la cantidad máxima de iteraciones, el tamaño de batch y si es necesario construir los lotes aleatoriamente.

Jueguen con los valores de los parámetros hasta obtener rectas que logren un buen ajuste a los puntos. Fijensé como evoluciona la función de costos a lo largo de las iteraciones, como afecta el tamaño de batch, si para alguna condición se vuelve incalculable (por ejemplo, si aparecen errores del tipo _NaN_), etc.

Por defecto, la actualización automática está desactivada, por lo que deben tildar el checkbox _¿Calcular?_ y hacer click en _Enviar_ para ver y/o actualizar los gráficos. Si la computadora que están usando es potente, marquen la opción de _Activar actualización automática_.
"

# ╔═╡ f5a39227-7e6e-4600-9145-a9f141545cfc
md"""
Activar actualización automática: $(@bind actualizarAutomaticamente CheckBox(false))
"""

# ╔═╡ d5e1d2db-a56f-4e65-9357-0a190d6afa1b
md"""
Tasa de aprendizaje: $(@bind alpha Slider(0.00000:0.00005:0.001, default=0.001, show_value=true))
"""


# ╔═╡ 88f8f89c-4db1-42ec-809b-d22c57bcbe33
md"""
Iteraciones: $(@bind iters Slider(10:10:2000, default=600, show_value=true))
"""


# ╔═╡ 7d05377e-878e-4e49-83ca-ac66d8ad935f
md"""
Tamaño de batch: $(@bind batch_size Slider(1:1:6, default=2, show_value=true))
"""


# ╔═╡ 6f05cbd8-3a32-4b2c-b3c5-ca3764145620
md"""
Mezclar batch: $(@bind mezclar CheckBox(true))
"""


# ╔═╡ fb6224da-229b-4e9d-b299-515d680ecbbe
begin
	if actualizarAutomaticamente==false
		md""" ¿Calcular? $(@bind calcular_gda confirm(CheckBox(default=false), label="Enviar"))"""
	else
		md"<-- Cálculos internos -->"
	end
end

# ╔═╡ 0359e14d-ee0d-46f1-a08f-38691de712eb
md"El algoritmo está implementado en la celda a continuación, cuyo código está oculto. La celda devuelve, como resultado, los valores finales calculados para los parámetros, así como el histórico de valores de la función de costos y el histórico de los parámetros."

# ╔═╡ 6ff7b246-f55f-41cb-adc3-3ae517d28f27
begin
    # Costo MSE sobre TODO el dataset (para monitorear la evolución)
    function costo_mse(X, y, θ)
        ŷ = X * θ
        return mean((y .- ŷ).^2)
    end

    # Gradiente MSE sobre un mini-batch de índices idx
    function gradiente_batch(X, y, θ, idx)
        XB = @view X[idx, :]
        yB = @view y[idx]

		r  = XB*θ .- yB
        # grad = 2/|B| * X_B' * (X_B θ - y_B)
        return (2/length(idx)) * (XB' * r)
    end

    # Descenso por gradiente con mini-batch (sin dependencias externas)
    function gd_minibatch(x, y; α=0.01, iters=500, B=32, mezclar=true, θ=zeros(size(x,2)))
		X = hcat(ones(length(x)), x)
		#X = hcat(ones(length(x)), (x .- mean(x)) ./ max(std(x), 0.001))
        n = size(X,1)
        θ = copy(θ)
        hist_cost = Vector{Float64}(undef, iters)
        hist_theta = Matrix{Float64}(undef, length(θ), iters)
        idx_all = collect(1:n)
        for t in 1:iters
            if mezclar
                randperm!(idx_all)  # mezcla in-place
            end
            # tomamos un mini-batch al azar (con o sin mezcla previa)
            if B == n
                idxB = idx_all
            else
                # sample sin reemplazo si B<=n
                idxB = @view idx_all[1:B]
            end
            g = gradiente_batch(X, y, θ, idxB)
            θ -= α .* g 
            hist_cost[t] = costo_mse(X, y, θ)  # costo full-data (para monitoreo)
            hist_theta[:, t] = θ
        end
        return θ, hist_cost, hist_theta
    end

	# Inicializamos en ceros (o podés probar con OLS como inicialización)
	θ = zeros(2)
	if actualizarAutomaticamente || calcular_gda
	    θ̂, J_hist, Θ_hist = gd_minibatch(x, y; α=alpha, iters=iters, B=batch_size, mezclar=mezclar, θ=θ)
		(; θ̂, J_hist, Θ_hist)
	else
		md"<-- Esperando órden de calcular -->"
	end
	
end

# ╔═╡ def6e375-89ad-4bda-b4cf-18fed7f4604d
begin
	if actualizarAutomaticamente || calcular_gda
	    xs_ = range(extrema(x)..., length=200)
	    y_line = θ̂[1] .+ θ̂[2] .* xs_
	
	    p1 = scatter(x, y; label="Datos", markersize=5, xlabel="x", ylabel="y",
	        title="Ajuste por GD (mini-batch).   θ̂ = [$(round(θ̂[1],digits=3)), $(round(θ̂[2],digits=3))]")
	
	    plot!(p1, xs, y_line; label="Recta GD", lw=3)
	
	    p1
	else
		md"<-- Esperando órden de calcular -->"
	end
		
end

# ╔═╡ 1efdb5e2-5a7d-4536-b9ac-82b89497803b
begin
	if actualizarAutomaticamente || calcular_gda
		md"""
		Escala Logarítmica: $(@bind logy CheckBox(false))
		"""
	else
		md"<-- Esperando órden de calcular -->"
	end
end

# ╔═╡ 421e2a74-a96c-43ca-807f-9fc163169ae9
begin
	if actualizarAutomaticamente || calcular_gda
	    its = 1:length(J_hist)
	    if logy
	        p2 = plot(its, J_hist; yscale=:log10, lw=2, label="J(θ) (log)",
	            xlabel="Iteración", ylabel="Costo MSE", title="Evolución del costo (escala log)")
	    else
	        p2 = plot(its, J_hist; lw=2, label="J(θ)",
	            xlabel="Iteración", ylabel="Costo MSE", title="Evolución del costo")
	    end
	    p2
	else
		md"<-- Esperando órden de calcular -->"
	end
end


# ╔═╡ 9f7a80f3-071c-4954-bc61-9ea08618538c
begin
	if actualizarAutomaticamente || calcular_gda
	    J_final = J_hist[end]
	    md"""
**Resumen**
	
- **Mini-batch (B)** = $(batch_size)  
- **α** = $(alpha) ·· **Iteraciones** = $(iters)
- **θ̂ (GD)** = [ $(round(θ̂[1],digits=4)), $(round(θ̂[2],digits=4)) ]  
- **Costo final** \(J(θ̂)\) = **$(round(J_final, digits=6))**
"""
	else
		md"<-- Esperando órden de calcular -->"
	end
end

# ╔═╡ 80fc8a1d-c765-4b70-9b81-fe79845ce39e
md"""
!!! info "Actividad para el alumno"
	En que versión del algoritmo de descenso por gradiente estoy cuando defino estos valores en los slides:

    1. _Tamaño de batch_ igual a 1 y _Mezclar batch_ activado.
    2. _Tamaño de batch_ igual a 6 y _Mezclar batch_ desactivado.
"""

# ╔═╡ 00e4e680-75da-48d5-b0d7-1624654b3869
details(md"""Quiero ver la respuesta""", md"""

En el primer caso, la configuración lo convierte en el algoritmo de descenso por gradiente estocástico. En el segundo, en la versión convencional del descenso por gradiente.
		
""")

# ╔═╡ 93d1b391-b756-4c03-b9a9-c392d7556cc1
md"## Conceptos claves"

# ╔═╡ 0e5a419c-7d73-4800-b69d-521926e78835
md"Los conceptos claves tratados en este notebook fueron:
* El **Aprendizaje Automático** es una metodología de la rama de la **Inteligencia Artificial** dedicada a la identificación de patrones en conjunto de datos.
* La rama de estudio se llama **Inteligencia Artificial**, los algoritmos y modelos no son inteligentes, son matemática + computación, requiriendo un montón de definiciones previas por la persona que los codifica.
* El **Aprendizaje Automático Supervisado** busca encontrar relaciones funcionales entre un conjunto de variables independientes (features, nuestras $X$) y una variable dependiente ($y$). Estos algoritmos solo ajustan parámetros, la forma de la relación debe ser definida explícitamente por la persona que los codifica.
* La **Regresión Lineal** es un modelo de **Aprendizaje Automático Supervisado** que sirve para predecir variables independientes numéricas.
* Hacer que la regresión lineal **aprenda** significa calcular el valor de sus parámetros, esto es, los valores de los coeficientes que multiplican a las $X$ y el del término independiente.
* Para calcular los parámetros, **entrenamos** el modelo a partir de datos, estos es, buscamos minimizar una función de costos **J** asociada al error del modelo.
* El entrenamiento lo podemos realizar a partir de un algoritmo denominado **Descenso por gradiente**.


"

# ╔═╡ Cell order:
# ╟─20584b47-0106-4311-9c40-138af44285e7
# ╟─3b374eb4-0108-4fbd-8f4b-f882b4b2d383
# ╟─dcd19bd9-b27e-4fb5-a2b9-6d4d39358fb2
# ╟─bad85a58-6c6f-11f0-2681-b7cbc064773f
# ╟─f4f54c42-1c1d-4b25-a350-1958eab8c2d6
# ╟─05d04f8b-41d2-4242-ac0f-e463e516bbaf
# ╟─ec7a4883-d132-406e-be4b-535e28425924
# ╟─30ab0a3a-ba33-4fd8-8803-859256d9f7a5
# ╟─2634d4bc-9787-4ba1-af5c-1da8a024c29b
# ╟─80e6264a-8ab5-4e0e-893b-040f59b93681
# ╟─36cd88c6-6a9c-4fb4-94d4-4df67b467b18
# ╟─ed11b0db-0864-448e-91fb-33b651b2a018
# ╟─2556052e-b072-4b7c-b856-563d1ad8ccb8
# ╟─d6176296-d980-48cc-b1fe-b4356e7d76ff
# ╟─101f7fb3-ddc3-4f1e-93d0-f171f3383785
# ╟─5f6b7a8a-b329-416f-bef8-c16e650ee573
# ╟─5106fa31-4f1a-4c53-b681-9624b8c52f12
# ╟─3c34fb87-3651-4d3f-bc2f-fbfcf477d779
# ╟─6abd3625-b1ce-496a-a383-3d05fe0ae8e4
# ╟─a7414592-a73e-45c1-95a4-08eaddf52b71
# ╟─3d59c3b1-b26f-49b4-952b-094e92057055
# ╟─9930908a-4c0e-43da-8e4e-4849c578a611
# ╟─6d3a1038-29e3-4ef6-bb49-c0f1520cc801
# ╟─77d618a7-4394-483b-bd37-ffebc31ec2b5
# ╟─2b8fad25-c174-4011-b9d1-7a9c98078f85
# ╟─d27f9dc4-f9e6-4e9d-b513-7516c1dea5a5
# ╟─d4368425-74ea-4be4-857a-c9a0306ff70b
# ╟─82e405e1-af1d-4765-a8e1-0c6c5808043c
# ╟─05915b04-665f-4a1c-9d14-0166f4b8e3a4
# ╟─7df3ed5c-59da-472d-8f0f-f678b4f3916b
# ╟─1ca896d7-dace-4843-a6ac-6f117d6f2673
# ╟─14b32b00-742e-414a-8dc2-43959e3771dd
# ╟─790a2224-2741-4b76-8fcb-8df84462f15f
# ╟─9f0d3ddb-7871-4165-89f0-60c5ce3ebbe4
# ╟─d3ec59b7-9051-44d3-a47a-4ddea0cde918
# ╟─863b65b2-a825-4770-be45-be80f171e413
# ╟─3967429e-eb9d-44f4-b3cc-b4fc9f3500b0
# ╟─9fae6ca9-9946-4724-b3fe-17c839a6522a
# ╟─328e02e5-948c-400e-be1f-2614aaf5e270
# ╟─b974ef0c-d4a0-4d0a-a6aa-d54c6d74ae76
# ╟─e4acf6e2-91b9-42dd-9f8f-c8090bc0791c
# ╟─50b4bc69-cff5-4420-a077-039e6fa2309c
# ╟─72ca99cc-6fc7-41c5-a57c-53abaa6cfebc
# ╟─8c0d6ee2-be75-4c73-84d7-0a0fbf1d9d9a
# ╟─5024f788-89b7-481f-9bbd-800eb60ab478
# ╟─55adad7c-4c66-4330-b63f-cec456947ef7
# ╟─9c4a1d2f-e919-4d92-825c-ecee3e2de8c4
# ╟─e5639789-9cec-4e28-b84f-168d752032a2
# ╟─e30f36d1-39b5-492e-bee1-7de0b97f0872
# ╟─d1ec2d81-3bef-463b-a36b-bbf18073479b
# ╟─605dfe66-f7a4-4228-80ce-fecbdcf44974
# ╟─61021325-abea-4b12-ac44-0a6af16751a3
# ╟─2d0f88f1-bed1-4816-809c-fc224d6a8959
# ╟─de2a0723-72a8-4c13-8695-22666e599b80
# ╟─9debc0a2-4874-4a34-bcf4-b8d950cb8a87
# ╟─b42d04c3-d810-410e-9898-62afa321384a
# ╟─899c3944-4058-4fe5-a53a-fcb7cc8c34a1
# ╟─f5a39227-7e6e-4600-9145-a9f141545cfc
# ╟─d5e1d2db-a56f-4e65-9357-0a190d6afa1b
# ╟─88f8f89c-4db1-42ec-809b-d22c57bcbe33
# ╟─7d05377e-878e-4e49-83ca-ac66d8ad935f
# ╟─6f05cbd8-3a32-4b2c-b3c5-ca3764145620
# ╟─fb6224da-229b-4e9d-b299-515d680ecbbe
# ╟─0359e14d-ee0d-46f1-a08f-38691de712eb
# ╟─6ff7b246-f55f-41cb-adc3-3ae517d28f27
# ╟─def6e375-89ad-4bda-b4cf-18fed7f4604d
# ╟─421e2a74-a96c-43ca-807f-9fc163169ae9
# ╟─1efdb5e2-5a7d-4536-b9ac-82b89497803b
# ╟─9f7a80f3-071c-4954-bc61-9ea08618538c
# ╟─80fc8a1d-c765-4b70-9b81-fe79845ce39e
# ╟─00e4e680-75da-48d5-b0d7-1624654b3869
# ╟─93d1b391-b756-4c03-b9a9-c392d7556cc1
# ╟─0e5a419c-7d73-4800-b69d-521926e78835
