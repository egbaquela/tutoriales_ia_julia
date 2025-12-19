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

# ╔═╡ 0f3a7b8e-7225-11f0-04e6-0d15e796d580
begin
	using Pkg
	Pkg.activate()

	using PlutoUI
	
	using Plots

	using Random
	using Statistics
	using LinearAlgebra

	TableOfContents(title="Contenido")
end

# ╔═╡ 8d9a4066-1d67-4006-abf9-430606fa53b2
md"# Introducción a los problemas de clasificación"

# ╔═╡ 731f3886-2f49-468d-a369-f658f0ec714a
md"En este notebook continuaremos explorando el aprendizaje automático supervisado, pero ahora desde una nueva prespectiva: en vez de predecir números, predecir cualidades no numéricas, clases. Por ejemplo, si un dispositivo mecánico está roto o no a partir de una lectura de ssu modos de vibración, si una persona tiene cierta enfermedad a partir de los resultados de un examen, si una transacción bancaria podría ser fraudulenta, etc. Este tipo de problemas se denominan **problemas de clasificación**, porque nuestro objetivo no el el predecir un valor concreto, sinó asociar a nuestra muestra a una categoría, a una **clase**.

Los paquetes a utilizar en este notebook son:

* PlutoUI
* Plots
* Random
* Statistics
* LinearAlgebra

"

# ╔═╡ bb12b1f9-9cd0-416a-a008-29f502d79607
md"## Setup del notebook"

# ╔═╡ 5444d716-3f89-4a9c-96c2-5a1d95b0e26c
md"## Prediciendo clases"

# ╔═╡ 6a23026a-8d0e-4020-a152-ac0532755d20
md"Hasta ahora conocemos solo un algoritmo de aprendizaje supervisado, la _regresión lineal_:

$\hat{y} = \theta_{0} + \sum_{j=1}^{m}\theta_{j}\cdot x_{j}$

Pero, si lo pensamos bien, ¿como podríamos predecir una categoría con esa fórmula? A ver, tomamos una muestra de atributos numéricos $x_{1}, ..., x_{j}$, multiplicamos a cada $x$ por un coeficiente, las sumamos y le sumamos un término independiente, ¿que sale después de todo eso? Claro, ¡un número!. De hecho, en base a los valores de los parámetros de la función y de los atributos muestreados, podría ser cualquier número entre $-\infty$ y $+\infty$. ¡¿Como hacemos para que el resultado sea _transacción fradulenta_ o _paciente sano_?!

Bueno, obviamente no podemos. Pero lo que si podemos hacer es, dado que todos los ejemplos de clasificación que enunciamos son del tipo binarios (es decir, admiten dos posibles valores, _enfermo_ o _sano_, _normal_ o _en falla_, _transacción normal_ o _transacción fraudulenta_) es asociarle un número a cada categoría, por convención $0$ y $1$, y tratar de predecir esos números. Entonces, si nuestro modelo predice $0$ sabríamos que el dispositivo mecánico esta trabajando en forma normal, pero si el resultado es $1$ está fallando. 

¡Genial!, aunque ahora nos debería surgir otra pregunta: ¿como hacemos para que una regresión lineal, que devuelve cualquier número, solo devuelva $0$ y $1$? La respuesta es _aplicando un filtro en la salida_. Pero antes de hacer eso, exploremos una función curiosa, la **función sigmoide**, que nos va a ayudar mas adelante con esto.

"

# ╔═╡ 1204bab8-9a29-4987-96c3-859895d25e09
md"""
!!! info "Actividad para el alumno"
	Armen ejemplos de problemas en los cuales nos interesaría predecir categorías. ¿Son los problemas que pensaron binarios?, ¿o tienen mas de 2 categorías?
"""

# ╔═╡ 17ccfd8a-6cc2-4f55-860f-78a50695c590
md"### Función sigmoide"

# ╔═╡ ffd10179-2546-47e8-9105-04c11c2d534f
md"""
La función sigmoide $\sigma(x)$ es una función cuyo dominio son todos los reales, pero su codominio está acotado al intervalo abierto $(0,1)$. Es decir, le podemos dar a $x$ cualquier valor en los reales (es decir entre $-\infty$ y $+\infty$) y el resultado siempre será un número entre $0$ y $1$. Mas aún, en su formulación se utiliza una variable intermedia $z$, la cual es linea respecto de $x$. Cuando $z$ es positivo, $\sigma(x)$ se vuelve rápidamente asintótica a $1$. Y cuando $z$ es negativo, $\sigma(x)$ se vuelve rápidamente asintótica a $0$. Mas aún, la función vale $0.5$ cuando $z=0$.

Se define mediante la siguiente ley:

$\sigma(x)=\frac{1}{1+e^{-z(x)}},\quad \text{con } z(x) = k\,(x-c)$

- **k** controla la **pendiente** (qué tan abrupta es la transición).  
- **c** mueve el **punto medio** (donde $\sigma=0.5$).  

Vemos que, cuando $z(x)=0$, $\sigma(x)=\frac{1}{2}$, que para valores negativos el denominador crece, acercando a $\sigma(x)$ a cero, y que para valores positivos de $z(x)$ sucede lo contrario. Cambiando los valores de $k$ y $c$ podemos lograr diferentes dinamicas respecto de $x$. Juguemos un poco con los parámetros de la función, recordando que la variable independiente es $x$:
"""

# ╔═╡ f34a6444-5af8-46cd-8f67-269596fcd5f3
md"""
Pendiente (k): $(@bind k Slider(0.1:0.1:10.0, default=2.0, show_value=true))
"""

# ╔═╡ 27142a32-9c9d-49ed-880f-0ab724e9b190
md"""
Centrado (c): $(@bind c Slider(-5.0:0.1:5.0, default=0.0, show_value=true))
"""

# ╔═╡ 5ad71ef5-ae41-4bf0-bacd-5924b38dc182
md"""
Ancho del eje x: $(@bind xspan Slider(5:1:30, default=10, show_value=true))
"""

# ╔═╡ 51875608-e199-4ec8-8ed3-4a274c13819c
md"""
Mostrar derivadas: $(@bind show_deriv CheckBox(default=true))
"""

# ╔═╡ 613dd654-a27b-4ce9-b84a-ec6b3ffc57ef
md"""
Punto a evaluar (x): $(@bind x_eval Slider(-30.0:0.1:30.0, default=0.0, show_value=true))
"""

# ╔═╡ 51e386a3-f9e6-4ce6-a1e0-1bdb18a6a96e
begin
	# Definición y utilidades
	σ(z) = 1 / (1 + exp(-z))                 # sigmoide
	logit(p) = log(p/(1-p))                  # inversa logística (para marcas 25% y 75%)
	x25(k,c) = c + (logit(0.25))/k           # ≈ c - log(3)/k
	x75(k,c) = c + (logit(0.75))/k           # ≈ c + log(3)/k
	md"<-- Cálculos internos -->"
end

# ╔═╡ b887f7a9-f215-475b-9ec2-ad869c78bce1
md"**Valor de $z(x)$: $(k*(x_eval - c))**

**Valor de $$\sigma(x)$$: $(σ(k*(x_eval - c)))**
"

# ╔═╡ 80994c8a-d837-44ba-9298-18fcaeec47d6
begin
	# Datos para el gráfico
	xs = range(-xspan, xspan; length=600)
	ys = σ.(k .* (xs .- c))
	
	# Derivada normalizada (máx real = k/4 en x=c). La escalamos a [0,1] para graficarla junto a σ.
	yprime = k .* ys .* (1 .- ys)             # σ'(z) = k σ (1-σ)
	yprime_norm = (4/k) .* yprime             # ahora su máximo teórico es 1 en x=c
	
	# Marcas de 25% y 75% (ancho de transición)
	x25_ = x25(k,c); x75_ = x75(k,c)
	
	# Punto evaluado
	y_eval = σ(k*(x_eval - c))
	
	# Plot
	p = plot(xs, ys; lw=3, label="σ(k(x-c))", xlabel="x", ylabel="p", ylim=(0,1),
	         title="Sigmoide (k=$(k), c=$(c))")
	
	# Líneas guía
	hline!(p, [0.5]; c=:gray, ls=:dash, label="p=0.5")
	vline!(p, [c]; c=:gray, ls=:dash, label="x=c")
	vline!(p, [x25_, x75_]; c=:gray, ls=:dot, label=["x25%" "x75%"])
	
	# Derivada (normalizada)
	if show_deriv
		plot!(p, xs, yprime_norm; ls=:dash, lw=2, label="σ'(x) (normalizada)")
	end
	
	# Punto evaluado
	scatter!(p, [x_eval], [y_eval]; ms=8, label="σ(k(xₑ-c))")
	
	p

end

# ╔═╡ ac1a77e0-5dde-4311-b10f-3562007426bd
md"### Regresión logística"

# ╔═╡ bbdfd75f-d1ae-45db-bebe-9df53408af0d
md"Muy linda la sigmoide, ¿pero para que la vimos? Bueno, es fácil. $z(x)$ está expresada como $k(x-c)$, pero si aplicamos propiedadad distributiva,  $z(x)=kx-kc$. Y dado que tanto $k$ como $c$ son constantes, lo que tenemos es una función lineal. De hecho, en realidad, $z(x)$ podría ser cualquier cosa que devuelva un número real, ni siquiera una función lineal. Entonces, a fin de tener una función que prediga $0$ y $1$ podríamos reemplazar $z(x)$ por la fórmula de la regresión lineal, usando la sigmoide para obtener nuestras predicciones:

$\hat{y} = \sigma(x)=\frac{1}{1+e^{-(\theta_{0} + \sum_{j=1}^{m}\theta_{j}x_{j})}}$

A esta formulación la llamamos [**Regresión Logística**](https://es.wikipedia.org/wiki/Regresi%C3%B3n_log%C3%ADstica). Entonces, si tenemos un conjunto de datos $(X,y)$ podemos ejecutar el algoritmo de descenso por gradiente para calcular los valores de cada uno de los parámetros, nuestra $\theta$.

"

# ╔═╡ 59520c13-f474-428c-ad45-839008b09f90
md"""
!!! info "Actividad para el alumno"
	¿Que implican los casos en los cuales el valor de $\sigma(x)$ es cercano a $0.5$? En términos de nuestros datos, ¿cuando se dan?
"""

# ╔═╡ 7595ea23-fa87-4ae3-ac10-02fae88d504c
md"Pero antes de ir al descenso por gradiente, veamos un poco como trabaja la predicción usando regresión logística. Supongamos que queremos predicir el funcionamiento de un equipo. Las categorías sería:

* 0: Funciona bien
* 1: Funciona mal

Y para estimar si está funcionando bien o mal, tenemos dos predictores, $x_{1}$ y $x_{2}$ (es decir, podemos medir dos propiedades del equipo y en base a ellas estimar como está funcionando). Para este caso, la familia de regresiones logísticas asociadas serían:

$\hat{y} = \sigma(x)=\frac{1}{1+e^{-(\theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2})}}$

Vemos que tenemos que calcular tres parámetros: $\theta_{0}$, $\theta_{1}$ y $\theta_{2}$. Vamos a intentar estimarlos a mano usando el siguiente gráfico. Vean que pasa cuando cambiamos los parámetros del modelo, como la predicción (_azul_ o _rojo_) varía a medida que los parámetros cambian. Si el gráfico anda lento, bajen la resolución del mapa, para reducir cálculos.

"

# ╔═╡ ed469785-311f-438d-88a6-c3b7aafbdfb6
md"""
**Controles**

Término independiente ($\theta_{0}$): $(@bind theta0 Slider(-10:0.5:10, default=0, show_value=true))

Coeficiente 1 ($\theta_{1}$): $(@bind theta1 Slider(-10:0.5:10, default=0, show_value=true))

Coeficiente 2 ($\theta_{2}$): $(@bind theta2 Slider(-10:0.5:10, default=0, show_value=true))

Resolución mapa: $(@bind res Slider(10:10:160, default=50, show_value=true))

Mucha mezcla de clases: $(@bind mezclar_mucho_las_clases CheckBox(default = false))

"""

# ╔═╡ ed00d65b-2271-470f-b3c3-278e2cccac06
begin
	#### Muestreo de datos ####
    Random.seed!(123)

    # 2 nubes gaussianas
    n_per = 50
	if mezclar_mucho_las_clases
	    μs = [[-1.0, -1.1], [1.0, 1.0]]
	    Σ  = [2.25 0.0; 0.0 2.25]
	else
	    μs = [[-2.0, -2.0], [2.0, 2.0]]
	    Σ  = [2.25 0.0; 0.0 2.25]
	end

    function sample_gauss(μ, Σ, n)
        A = cholesky(Symmetric(Σ)).L
        return [μ .+ A*randn(2) for _ in 1:n]
    end

    X1 = sample_gauss(μs[1], Σ, n_per)
    X2 = sample_gauss(μs[2], Σ, n_per)

    X = reduce(vcat, (X1, X2))              # Vector de SVector-like (2-element Vector)
    Y = vcat(fill(0, n_per), fill(1, n_per))  # etiquetas 1..C

    # Matriz 2×N para cálculo vectorizado
    Xmat = reduce(hcat, X)    # 2 × N

    classes = sort(unique(Y)) # [0,1]
    colors  = Dict(0 => :royalblue, 1 => :tomato)

    # Rangos para grilla
    xmin, xmax = minimum(Xmat[1, :]) - 2, maximum(Xmat[1, :]) + 2
    ymin, ymax = minimum(Xmat[2, :]) - 2, maximum(Xmat[2, :]) + 2


	#### Cálculo de clases ####
	function rl(x1, x2, theta0, theta1, theta2)
		z= theta0 + theta1*x1 + theta2*x2
		return σ(z)
	end
	
    xs_rl = range(xmin, xmax, length=res)
    ys_rl = range(ymin, ymax, length=res)
    Z  = Array{Int}(undef, length(xs_rl), length(ys_rl))  # clase predicha en cada celda

    # Recorremos la grilla (simple y claro; si va lento, bajá res)
    for (i, xv) in enumerate(xs_rl)
        for (j, yv) in enumerate(ys_rl)
            cls = rl(xv, yv, theta0, theta1, theta2)
            Z[i, j] = round(Int,cls)
        end
    end


	md"<-- Cálculos internos -->"
end

# ╔═╡ d744c7fd-89e1-4ab1-a312-4a5bc15316a0
begin
	#### Cálculo del error ####
	fallos = 0
    for i in 1:length(Y)
        if round(Int,rl(Xmat[1,i], Xmat[2,i], theta0, theta1, theta2))!=Y[i]
			global fallos += 1
		end
    end
	tasa_fallos = 100*fallos / length(Y)
	md"""**Tasa de fallos: $(tasa_fallos)%**"""
end

# ╔═╡ 86676f1f-dc86-4b9f-b58f-6324a0d3520a
begin
    p_rl = heatmap(
		range(xmin, xmax, length=res),
	    range(ymin, ymax, length=res),
	    Z';                             # z[j,i] → (x[i], y[j]) → transponemos
	    color = [colors[c] for c in classes],
	    colorbar = false,
	    xlabel = "X1",
	    ylabel = "X2",
	    title = "Reg. Logística — Clase predicha en el plano",
	    aspect_ratio = :equal,
	    alpha = 0.35
	)

    # Puntos de entrenamiento
    for c in classes
        inds = findall(==(c), Y)
        scatter!(p_rl, Xmat[1, inds], Xmat[2, inds];
                 ms=5, label="Clase $c", color=colors[c])
    end
	
	p_rl
end

# ╔═╡ 3ce2262e-07af-4305-be84-092678400d23
md"""
!!! tip "Para tener en cuenta"
	La regresión logística representa a un plano (un hiperplano, en realidad) que divide al espacio en dos subespacios: uno para la clase $0$, otro para la clase $1$. Mientras mas **separable** sean las dos clases, mayor será su efectividad.
"""

# ╔═╡ 6fdc346d-9e24-4865-a8c5-3da3388071f9
md"""
!!! tip "Para tener en cuenta"
	En el ejemplo de _Mucha mezcla de clases_ podemos intuir algo importante. La calidad final del modelo, es decir, que tan bien predice, viene determinada por el proceso en si. Si, para las dos variables independientes que podemos medir resulta que hay mucha superposición de clases, el error tenderá a ser alto. Una forma de mejorarlo sería analizar el sistema bajo estudio para identificar variables independientes que sean importantes pero que no hayamos tenido en cuenta..
"""

# ╔═╡ 0474e1b2-000d-4d87-a3ae-400b7b54c68a
md"""
!!! tip "Para tener en cuenta"
	Recuerden que la regresión logística nos devuelve un número en el entorno $(0,1)$. Si nos devuelve $0.17$, ¿como hacemos para asociarle la clase?

	Fácil, usamos como criterio de corte $0.5$. Si el resultado es menor a dicho número, asumimos que la clase predicha es $0$, en caso contrario es $1$. Dado que la regresión logística es asintótica a estos dos valores, en la mayoría de los casos esto funciona correctamente.
"""

# ╔═╡ 58def5dc-ed05-4047-8db1-c8bcff96db55
md"### Entropía cruzada como función costos"

# ╔═╡ 1d5a1222-f934-4972-a16b-cb7f0029bfdb
md"Para usar el descenso por gradiente, falta una parte: ¡la función de costos sobre la cual vamos a calcular el gradiente!

Es importante que, si bien similar, el significado de la función de costos (y de cada pérdida en si) es diferente que cuando predecíamos una varaible continua. Antes, el MEA y el MSE medían el nivel de cercanía del valor predicho contra el real. Pero acá, en un problema de predecir solo dos valores, no hay nivel de cercanía, o acertamos o le erramos, pero no hay intermedio.

Una forma intuitiva de medir el error sería contar la cantidad de errores en la muestra predicha y luego dividirlo respecto del tamaño de la muestra. Es decir, supongamos que tengo una muestra de tamaño 3:

* Elemento muestreal 1: Valor real = 0 - Valor Predicho = 0
* Elemento muestreal 2: Valor real = 1 - Valor Predicho = 0
* Elemento muestreal 3: Valor real = 1 - Valor Predicho = 1

Entonces, el cálculo anterior quedaría como:

$Costo = \frac{1}{n}\sum_{i=1}^{n}(y - \hat{y})^2 =\frac{(0-0)^{2} + (1-0)^2 + (1-1)^2}{3} = \frac{1}{3} = 33.3...\%$

Es básicamente el MSE. Pero, dado que los términos de la suma solo valen $0$ o $1$, es también una función de conteo, que tiene forma de escalón si la graficamos en forma acumulada contra el número de elementos muestreales: 

"

# ╔═╡ 209edc6e-0424-418d-9ead-1f2449aff416
md"""
Usar valores continuos: $(@bind usar_continuos CheckBox(default=false))
"""

# ╔═╡ beccda84-3492-4cb8-83dc-58136a537981
begin
	if usar_continuos
		md"""
**Real ($y$)**
	
_Elemento 1_ = $(@bind y1 Select([0,1], default=0))
_Elemento 2_ = $(@bind y2 Select([0,1], default=1))
_Elemento 3_ = $(@bind y3 Select([0,1], default=1))
	
**Predicción ($\hat{y}$)**
		
_Elemento 1_ = $(@bind yhat1 Slider(0.000001:0.000001:0.999999, default=0.000001, show_value=true))
		
_Elemento 2_ = $(@bind yhat2 Slider(0.000001:0.000001:0.999999, default=0.000001, show_value=true))
		
_Elemento 3_ = $(@bind yhat3 Slider(0.000001:0.000001:0.999999, default=0.000001, show_value=true))

	"""
	else
		md"""
**Real ($y$)**
	
_Elemento 1_ = $(@bind y1 Select([0,1], default=0))
_Elemento 2_ = $(@bind y2 Select([0,1], default=1))
_Elemento 3_ = $(@bind y3 Select([0,1], default=1))
	
**Predicción ($\hat{y}$)**
		
_Elemento 1_ = $(@bind yhat1 Select([0,1], default=0))
_Elemento 2_ = $(@bind yhat2 Select([0,1], default=1))
_Elemento 3_ = $(@bind yhat3 Select([0,1], default=1))

	"""
	end
end

# ╔═╡ f3b78cb7-9796-4450-886b-02485c0a5b99
begin
	granularidad = 300
	bs = range(0, 3; length=granularidad)

	if usar_continuos
		e1 = repeat([(y1 -yhat1)^2], granularidad)
		e2 = repeat([(y2 -yhat2)^2], granularidad)
		e3 = repeat([(y3 -yhat3)^2], granularidad)
	else
		e1 = repeat([(y1 -yhat1)^2], granularidad)
		e2 = repeat([(y2 -yhat2)^2], granularidad)
		e3 = repeat([(y3 -yhat3)^2], granularidad)
	end
		
	conteo_acumulado = e1
	conteo_acumulado[101:granularidad] .+= e2[101:granularidad]
	conteo_acumulado[201:granularidad] .+= e3[201:granularidad]
	conteo_acumulado = conteo_acumulado ./ 3
	
	# Gráfico combinado:
	# - Eje izquierdo: errores puntuales (escalón)
	# - Eje derecho: MSE (escalón)
	p_esc = plot(bs, conteo_acumulado; 
				 seriestype=:steppost, label="Conteo", legend=:topright, xlabel="Elemento muestreal", ylabel="Conteo de error acumulado", ylim=(-0.05,1.05))

	

	
	p_esc
end

# ╔═╡ 108cc91b-e594-4e34-b4b8-a0145b5f8de3
md"Si bien intuitamente pensamos en términos discretos, recuerden que si usamos la _regresión logística_ las predicciones serán asintóticas a $0$ o $1$ (por eso tenemos un checkbox de _Usar valores continuos_). 

Nuestra función de conteo es básicamente el MSE: 

$J(\theta) = \frac{1}{n}\sum_{i=1}^{n}(y_{1} - \sigma(\vec{x}_{i}, \theta))^2$

Y ya sabemos que tenemos que calcular el gradiente de esta función de costos para estimar los valores de $\theta$. La derivada parcial de $J$ respecto de un $\theta_{j}$ cualquiera será:

$\frac{\partial J}{\partial \theta_{j}}=2[\sum_{i=1}^{n}(y_{1} - \sigma(\vec{x}_{i}, \theta))\frac{\partial \sigma(x_{i}, \theta)}{\partial \theta_{j}}]$

_Wait a minute!_ Si, paremos acá. No es necesario seguir toda la línea de razonamiento, pero si sigo aplicando la regla de la cadena igual no voy a poder ignorar el hecho que multiplico la función sigmoide por su derivada. Ya sabemos que la sigmoide es basicamente plana en casi todo su dominio, entonces su derivada es nula en casi todo su dominio. Y en este caso, ellos haría que el gradiente de $J$, definida $J$ como el MSE, se vuelva nulo muy rápido, y nuestro descenso por gradiente no nos va a servir.

"

# ╔═╡ d9e96009-97cc-43eb-b8bd-9a1b2f0339e2
md"""
!!! tip "Para tener en cuenta"
	En esta sección, los subíndices están representando a los elementos muestreales. $y_{i}$ es la clase del elemento $i$, y $\vec{x}_{i}$ es el vector de todas las _features_ (variables independientes) asociadas a dicho elemento.
"""

# ╔═╡ cbdcbe8c-574a-48c5-b4d2-44c85271f161
md"""
!!! info "Actividad para el alumno"
	Continuén la regla de la cadena y traten de calcular la derivada parcial respecto de $\theta_{j}$. Un tip: 

    $\frac{d\sigma(z)}{dz}=\frac{e^{-z}}{(1+e^{-z})^2}=\sigma(z)(1-\sigma(z))$


"""

# ╔═╡ 925a0ae5-ca13-4777-b0a3-b572a2ddaf3a
md"Deberíamos intentar definir la pérdida de otra forma, no mediante el error cuadrático. Por suerte, como en ciencia y en ingeniería nos apoyamos siempre sobre [hombros de gigantes](https://es.wikipedia.org/wiki/A_hombros_de_gigantes), tenemos la suerte que ya se definió. La pérdida logística, o entropía cruzada, se define como: 

$\mathcal{L}(y,\hat{y}) = -\Big( y\log(\hat{y}) + (1-y)\log(1-\hat{y}) \Big), \qquad y\in\{0,1\}$

A primera vista mas complicado que el MSE. Pero sabiendo que $y$ vale $0$ o $1$, y que $\hat{y}$ es asintótica a esos dos valores, la evaluación de la fórmula anterior se reduce a solo 4 casos:

* $Si \ y=1 \ \text{y} \ \hat{y}\approx 0.99999... \rightarrow \mathcal{L}(y,\hat{y}) \approx -1\cdot log(0.99999...) \approx 0$
* $Si \ y=0 \ \text{y} \ \hat{y}\approx 0.00000... \rightarrow \mathcal{L}(y,\hat{y}) \approx -(1-0)\cdot log(1 - 0.00000...) \approx 0$
* $Si \ y=1 \ \text{y} \ \hat{y}\approx 0.0000... \rightarrow \mathcal{L}(y,\hat{y}) \approx -1\cdot log(0.0000...) \approx M$
* $Si \ y=0 \ \text{y} \ \hat{y}\approx 0.99999... \rightarrow \mathcal{L}(y,\hat{y}) \approx -(1-0)\cdot log(1 - 0.99999...) \approx M$

Donde $M$ es un número muy grande. Vemos que $\mathcal{L}(y,\hat{y})$ devuelve valores positivos cuando la predicción es erronea, y devuelve $0$ cuando hay acierto. Entonces, podríamos minimizar esta función. Pero..., ¿su gradiente?

Estamos de parabien, porque:

$\frac{\partial \mathcal{L}}{\partial z}
= \frac{\partial \mathcal{L}}{\partial \hat y}\cdot \frac{d\hat y}{dz}
= \left(-\frac{y}{\hat y} + \frac{1-y}{1-\hat y}\right)\,\hat y(1-\hat y)
= \boxed{\;\hat y - y\;}$

Lo que implica que:

$\frac{\partial \mathcal{L}}{\partial \theta_{j}} = (\hat y - y)x_{j}$

Lo cual, evidentemente, no nos genera un gradiente plano.

Para entenderla esta función de pérdida, juguemos un poco con $\mathcal{L}$:
"

# ╔═╡ e5282fec-95ca-478f-940a-cc7cfc7d839f
md"**Controles**"

# ╔═╡ 43d6b7fb-e700-49c5-9b90-e1f0fdb9c124
md"""
y (real): $(@bind y_real Slider([0;1], default=0.5, show_value=true))
"""

# ╔═╡ 613d6a09-96b6-4b9b-9ad1-eccb01940704
md"""
y (estimada): $(@bind yhat Slider(range(0.00000000001, 0.99999999999, length=1001), default=0.5, show_value=true))
"""

# ╔═╡ 1df521cc-6763-48c5-aa3a-c9db92c69130
begin

	loss(y, yhat; eps=1e-12) = - ( y*log(yhat + eps) + (1-y)*log(1 - yhat + eps) )

	# Curvas para y=0 y y=1
	xs_ = range(0.0, 1.0, length=1001)
	L0 = [loss(0, x) for x in xs_]
	L1 = [loss(1, x) for x in xs_]
	
	# Punto seleccionado
	Ly = loss(y_real, yhat)
	
	# Gráfico
	p_ = plot(xs_, L0; lw=2, label="L(y=0, ŷ)", xlabel="ŷ", ylabel="Pérdida", title="Entropía cruzada vs ŷ")
	plot!(p_, xs_, L1; lw=2, label="L(y=1, ŷ)")
	vline!(p_, [yhat]; ls=:dash, label="ŷ seleccionado")
	scatter!(p_, [yhat], [Ly]; ms=8, label="L(y=$(y_real), ŷ=$(yhat))")
	p_
end

# ╔═╡ 96644df8-9eb7-4693-ac3e-22697a8fd64a
md"""
**Pérdida $\approx$ $(Ly)**
"""

# ╔═╡ 04301c6c-79b2-4027-9c68-57eb3637ca90
md"## Conceptos claves"

# ╔═╡ e69cffa4-2a49-4712-846e-8c3958ad366c
md"""Los conceptos claves tratados en este notebook fueron:
* Podemos predecir **clases** (cualidades, condiciones, atributos, o sea, cosas no numéricas) si las asociamos a valores numéricos enteros.
* Los problemas de predicción de clases se conocen como problemas de **clasificación**.
* Acoplando una función sigmoide a una regresión lineal obtenemos una regresión logística.
* La regresión logística nos permite predecir una de dos clases (o sea, sirve para problemas de clasificación binaria).
* Usar el MSE puede ocasionar problemas numéricos. Nos conviene usar una función de costos basada en la **entropía cruzada**.
"""

# ╔═╡ Cell order:
# ╟─8d9a4066-1d67-4006-abf9-430606fa53b2
# ╟─731f3886-2f49-468d-a369-f658f0ec714a
# ╟─bb12b1f9-9cd0-416a-a008-29f502d79607
# ╟─0f3a7b8e-7225-11f0-04e6-0d15e796d580
# ╟─5444d716-3f89-4a9c-96c2-5a1d95b0e26c
# ╟─6a23026a-8d0e-4020-a152-ac0532755d20
# ╟─1204bab8-9a29-4987-96c3-859895d25e09
# ╟─17ccfd8a-6cc2-4f55-860f-78a50695c590
# ╟─ffd10179-2546-47e8-9105-04c11c2d534f
# ╟─f34a6444-5af8-46cd-8f67-269596fcd5f3
# ╟─27142a32-9c9d-49ed-880f-0ab724e9b190
# ╟─5ad71ef5-ae41-4bf0-bacd-5924b38dc182
# ╟─51875608-e199-4ec8-8ed3-4a274c13819c
# ╟─613dd654-a27b-4ce9-b84a-ec6b3ffc57ef
# ╟─51e386a3-f9e6-4ce6-a1e0-1bdb18a6a96e
# ╟─b887f7a9-f215-475b-9ec2-ad869c78bce1
# ╟─80994c8a-d837-44ba-9298-18fcaeec47d6
# ╟─ac1a77e0-5dde-4311-b10f-3562007426bd
# ╟─bbdfd75f-d1ae-45db-bebe-9df53408af0d
# ╟─59520c13-f474-428c-ad45-839008b09f90
# ╟─7595ea23-fa87-4ae3-ac10-02fae88d504c
# ╟─ed469785-311f-438d-88a6-c3b7aafbdfb6
# ╟─ed00d65b-2271-470f-b3c3-278e2cccac06
# ╟─d744c7fd-89e1-4ab1-a312-4a5bc15316a0
# ╟─86676f1f-dc86-4b9f-b58f-6324a0d3520a
# ╟─3ce2262e-07af-4305-be84-092678400d23
# ╟─6fdc346d-9e24-4865-a8c5-3da3388071f9
# ╟─0474e1b2-000d-4d87-a3ae-400b7b54c68a
# ╟─58def5dc-ed05-4047-8db1-c8bcff96db55
# ╟─1d5a1222-f934-4972-a16b-cb7f0029bfdb
# ╟─209edc6e-0424-418d-9ead-1f2449aff416
# ╟─beccda84-3492-4cb8-83dc-58136a537981
# ╟─f3b78cb7-9796-4450-886b-02485c0a5b99
# ╟─108cc91b-e594-4e34-b4b8-a0145b5f8de3
# ╟─d9e96009-97cc-43eb-b8bd-9a1b2f0339e2
# ╟─cbdcbe8c-574a-48c5-b4d2-44c85271f161
# ╟─925a0ae5-ca13-4777-b0a3-b572a2ddaf3a
# ╟─e5282fec-95ca-478f-940a-cc7cfc7d839f
# ╟─43d6b7fb-e700-49c5-9b90-e1f0fdb9c124
# ╟─613d6a09-96b6-4b9b-9ad1-eccb01940704
# ╟─96644df8-9eb7-4693-ac3e-22697a8fd64a
# ╟─1df521cc-6763-48c5-aa3a-c9db92c69130
# ╟─04301c6c-79b2-4027-9c68-57eb3637ca90
# ╟─e69cffa4-2a49-4712-846e-8c3958ad366c
