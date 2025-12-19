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

# ╔═╡ acb5e996-722c-11f0-063a-b7f2bb5d60c3
begin
	using Pkg
	Pkg.activate()

	using PlutoUI, Random, Statistics, LinearAlgebra
	using Plots
	default(dpi=140, legend=:topright)

	TableOfContents(title="Contenido")
	
end

# ╔═╡ 1988bf6c-988f-4238-a960-a7cc7b986a44
md"# Algoritmo de los vecinos mas cercanos (KNN)"

# ╔═╡ 48392ef4-cb55-4473-9905-8b9e61b38a24
md"En este notebook veremos un método _algorítmico_ de aprendizaje supervisado para regresión y clasificación. A diferencias de la regresión lineal y la regresión logística, las cuales ya estuvimos viendo, no partimos aquí de una ley analítica cuyos parámetros calculamos durante el entrenamiento a partir de datos, sinó que usamos directamente los datos al momento de predecir.

Los paquetes a utilizar en este notebook son:

* PlutoUI
* Random
* Statistics
* LinearAlgebra
* Plots

"

# ╔═╡ d8f52c6d-23c1-4dd6-8c5f-e1424719e4b6
md"## Setup del notebook"

# ╔═╡ f81aad0e-2089-4020-a8d8-1ed0746312c8
md"## Vecinos mas cercanos"

# ╔═╡ 4f61595d-6f51-4614-952c-d49173475e39
md"El algoritmo de [**k vecinos más próximos**](https://es.wikipedia.org/wiki/K_vecinos_m%C3%A1s_pr%C3%B3ximos) (K-Nearest Neighbors, KNN) es una técnica de predicción que asigna un valor (un número si estamos en un problema de regresión, una clase si estamos en un problema de clasificación) a un nuevo punto basándose en la cercanía de sus **vecinos** en el [**espacio de características**](https://es.wikipedia.org/wiki/Espacio_de_caracter%C3%ADsticas).

Para, para, para. ¿_Vecinos_?, ¿_espacio de características_?, _wtf_?! Bueno, empecemos viendo que es el _espacio de características_ (o de _features_, o de _predictores_). Supongamos que queremos predecir si una persona tiene determinado enfermedad en base a los resultados de un análisis de sangre. Puntualmente, queremos predecir si está enfermo o no en base a la cantidad de _glóbulos rojos_ y la cantidad de _glóbulos blancos_ por cada mililitro de sangre. Nuestra _variable dependiente_, lo que queremos _predecir_, es si está enfermo o no (es decir, queremos _clasificar_ en este ejemplo). Nuestras _varaibles independientes_, también conocidas como _características_, _features_ o _predictores_, son la cantidad de glóbulos rojos y glóbulos blancos. El espacio de características son todas las combinaciones de valores que pueden tomar los predictores, todas las posibles combinaciones de cantidades de glóbulos blancos y glóbulos rojos. Basicamente sería el producto cartesiano del conjunto de valores que puede tomar la cantidad de glóbulos rojos y del conjunto de valores que puede tomar la cantidad de glóbulos blancos. Cada predictor define una _dimensión_ del espacio de caraterísticas. O sea, si tenemos $15$ predictores, vamos a trabajar con coordenadas de $15$ componentes.

Ahora bien, si tenemos a nuestros elementos muestreales distribuidos en un espacio de caraterísticas, podemos hacer algo interesante. Podemos ver que tan cerca o lejos están entre sí. Es decir, podemos calcular [**distancias**](https://es.wikipedia.org/wiki/Distancia). Una distancia es cualquier función que podamos aplicar a un par de puntos de nuestro espacio que cumpla las siguientes condiciones:

* La distancia entre dos puntos distintos es siempre positiva.
* La distancia entre un punto consigo mismo mismo es cero.
* La distancia es simétrica, o sea, la distancia del punto $a$ al punto $b$ es la misma que la distancia del punto $b$ al punto $a$.
* Se cumple la desigualdad triangular. Es decir, si $d$ es la función de distancia, $d(a,b)\leq d(a,c) + d(c,b)$. Esto significa que ir directo desde $a$ a $b$ siempre es mas corto (o a lo sumo igual) que ir desde $a$ a $b$ pasando por $c$. 

La distancia mas conocida por nosotros es la [**distancia euclidea**](https://es.wikipedia.org/wiki/Distancia_euclidiana), o sea, la distancia en línea recta en un plano. En un espacio _n-dimensional_, con $a=(a_{1}, ..., a_{n})$ y $b=(b_{1}, ..., b_{n})$:

$d_{euclidea}(a,b) = \sqrt{\sum_{j=1}^{m}(a_{j}-b_{j})^2}$

La idea del _KNN_ es utilizar una base de datos formada poruna muestra de valores $(x,y)$, es decir, variables indpendientes junto al valor observado de la variable dependiente. Cada vez que quiero realizar una predicción de un nuevo punto con coordenadas $(x_{1}, ..., x_{n})$ calculo la distancia de este nuevo punto a cada uno de los que tengo en la base de datos, los ordeno del mas cercano al mas lejano y selecciono los $k$ elementos mas cercanos (que serían los $k$ primeros de la lista ordenada). Si estoy en un problema de regresión, **promedio** los valores de $y$ de esos $k$ elementos. Si estoy en uno de clasificación, cuanto la clase que mas se repite en ese conjunto de $k$ elementos (es decir, calculo la [**moda**](https://es.wikipedia.org/wiki/Moda_(estad%C3%ADstica))).

Vamos a ver como es esto de encontrar vecinos con el siguiente ejemplo de clasificación en un espacio de características _2-dimensional_. La clase a predecir es el color. Prueben cambiar las coordenadas para ver cuales son los vecinos mas  cercanos y traten de identificar que clase predeciría el algoritmo. Si se pone lento, bajen la resolución del mapa.

"

# ╔═╡ 275d670c-70fa-46df-9de1-9e6ef6d702fc
begin
    Random.seed!(123)

    # 3 nubes gaussianas
    n_per = 50
    μs = [[-2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]]
    Σ  = [0.8 0.0; 0.0 0.8]

    function sample_gauss(μ, Σ, n)
        A = cholesky(Symmetric(Σ)).L
        return [μ .+ A*randn(2) for _ in 1:n]
    end

    X1 = sample_gauss(μs[1], Σ, n_per)
    X2 = sample_gauss(μs[2], Σ, n_per)
    X3 = sample_gauss(μs[3], Σ, n_per)

    X = reduce(vcat, (X1, X2, X3))              # Vector de SVector-like (2-element Vector)
    Y = vcat(fill(1, n_per), fill(2, n_per), fill(3, n_per))  # etiquetas 1..C

    # Matriz 2×N para cálculo vectorizado
    Xmat = reduce(hcat, X)    # 2 × N

    classes = sort(unique(Y)) # [1,2,3]
    colors  = Dict(1 => :royalblue, 2 => :seagreen, 3 => :tomato)

    # Rangos para sliders y grilla
    xmin, xmax = minimum(Xmat[1, :]) - 2, maximum(Xmat[1, :]) + 2
    ymin, ymax = minimum(Xmat[2, :]) - 2, maximum(Xmat[2, :]) + 2

	md"<-- Cálculos internos -->"
end

# ╔═╡ a4bc4e34-02b1-4b74-8b6f-3bb0608d7949
md"**Controles**"

# ╔═╡ 0e9f19f8-450c-4bae-b482-72cc7900e79a
md"""
X1: $(@bind xq Slider(range(xmin, xmax, length=401), default=0.0, show_value=true))
"""

# ╔═╡ 6c24c1ba-4aac-4aba-9eff-2454e4768fbe
md"""
X2: $(@bind yq Slider(range(ymin, ymax, length=401), default=0.0, show_value=true))
"""

# ╔═╡ b8badc49-51b9-4530-acc6-327470af2bdf
md"""
K: $(@bind K  Slider(1:2:31, default=5, show_value=true)) 
"""

# ╔═╡ 2a331f78-316d-4692-a070-8d5d42946177
md"""
Resolución mapa: $(@bind res Slider(10:10:160, default=50, show_value=true))
"""

# ╔═╡ 8f5ef0e1-872c-4de6-ab67-ae5390c3b812
begin
	# Índices de los K vecinos más cercanos a q = [xq; yq]
	function knn_indices(Xmat::AbstractMatrix, q::AbstractVector, K::Int)
	    # Distancias euclídeas al cuadrado
	    dx = Xmat .- q
	    d2 = vec(sum(abs2, dx; dims=1))
	    return partialsortperm(d2, 1:K), d2
	end
	
	# Predicción por voto mayoritario (con desempate por menor suma de distancias por clase)
	function knn_predict(Xmat, Y, q, K)
	    idx, d2 = knn_indices(Xmat, q, K)
	    labs = Y[idx]
	    classes = unique(Y)
	
	    # Conteo por clase y suma de distancias (usamos d2, distancia^2, consistente para desempate)
	    counts = Dict(c => count(==(c), labs) for c in classes)
	    sums   = Dict(c => sum(d2[idx][labs .== c]) for c in classes)
	
	    # Orden: 1) más votos, 2) menor suma de distancias
	    ordered = sort(collect(classes);
	                   lt = (c1, c2) -> counts[c1] == counts[c2] ?
	                                    (sums[c1]   <  sums[c2]) :
	                                    (counts[c1] >  counts[c2]))
	    best = ordered[1]
	
	    p = counts[best] / K
	    return best, idx, p
	end
	md"<-- Cálculos internos -->"
end

# ╔═╡ bcf8ee1c-031f-404f-819c-f7b2915fb1a5
begin
    xs = range(xmin, xmax, length=res)
    ys = range(ymin, ymax, length=res)
    Z  = Array{Int}(undef, length(xs), length(ys))  # clase predicha en cada celda

    # Recorremos la grilla (simple y claro; si va lento, bajá res)
    for (i, xv) in enumerate(xs)
        for (j, yv) in enumerate(ys)
            cls, _, _ = knn_predict(Xmat, Y, [xv, yv], K)
            Z[i, j] = cls
        end
    end
    Z
	md"<-- Cálculos internos -->"
end

# ╔═╡ 749a0495-48ac-4b17-8ffe-e9be8baa508e
begin
    q = [xq, yq]
    ŷ, idxK, p̂ = knn_predict(Xmat, Y, q, K)
    (ŷ, idxK, p̂)
	md"<-- Cálculos internos -->"
end

# ╔═╡ 3226583d-af34-4eb5-9eaf-01dd83ac50c0
begin
    p = heatmap(
        range(xmin, xmax, length=res),
        range(ymin, ymax, length=res),
        Z';                             # z[j,i] → (x[i], y[j]) → transponemos
        color = [colors[c] for c in classes],
        colorbar = false,
        xlabel = "X1",
        ylabel = "X2",
        title = "KNN (K=$(K)) — Clase predicha en el plano",
        aspect_ratio = :equal,
        alpha = 0.35
    )

    # Puntos de entrenamiento
    for c in classes
        inds = findall(==(c), Y)
        scatter!(p, Xmat[1, inds], Xmat[2, inds];
                 ms=5, label="Clase $c", color=colors[c])
    end

    # Vecinos más cercanos
    scatter!(p, Xmat[1, idxK], Xmat[2, idxK];
             ms=10, msw=2, msc=:black, markerstrokewidth=1.5,
             label="Vecinos (K)")

    # Punto de consulta
    scatter!(p, [xq], [yq]; ms=11, marker=:star5, color=:black, label="Punto q")

    p
end

# ╔═╡ 6d7f4c7e-31d1-4df3-8214-7ea011a2bd6d
md"""
!!! tip "Para tener en cuenta"
	Para valores muy chicos de $K$, el algoritmo se vuelve muy sensible a la muestra que tenemos en la base de datos, o sea, es poco robusto. Pero para valores muy grandes, se vuelve insensible (en el límite, si $K$ es igual al tamaña de la muestra, siempre prediciría lo mismo). Valores corrientes de $K$ oscilan entre $5$ y $15$. El entrenamiento consiste justamente en probar valores de $K$ hasta encontrar el mejor.
"""

# ╔═╡ ac6c394e-6423-4c34-9570-560c7c7b1560
md"""
!!! tip "Para tener en cuenta"
	El valor de $K$ debería ser impar, para reducir las posibilidades de empate.
"""

# ╔═╡ 9ceab22d-6890-4c7b-87e5-ac1627934084
md"""
!!! tip "Para tener en cuenta"
	Es importante destacar que _KNN_ realiza una clasificación directa: no estima probabilidades, simplemente asigna la clase predominante entre sus vecinos más cercanos.
"""

# ╔═╡ 694c6612-312a-4072-a61d-d42766f1323b
md"## Algo mas sobre distancias"

# ╔═╡ 12cbffdf-9c25-49d6-b28b-71d2835e0dd9
md"La euclidea no es la única función de distancia. Hay un montón en realidad. Y uno tiene que elegir cual usar.

Como ejemplo de otras funciones de distancias, podemos indicar a la [**Distancia de Manhattan**](https://es.wikipedia.org/wiki/Geometr%C3%ADa_del_taxista) (o del peatón, o del taxista):

$d_{manhattan}(a,b) = \sum_{j=1}^{m}|a_{j} - b_{j}|$

Otra distancia conmunmente usada en la [**Distancia de Chebyshov**](https://es.wikipedia.org/wiki/Distancia_de_Chebyshov), que es la distancia mas grande medida en una dimensión a la vez:

$d_{chebyshov}(a,b) = \max_{j}(|a_{j} - b_{j}|)$

Juguemos un poco con las distancias, para ver que tan diferentes pueden ser:

"

# ╔═╡ 24f0dddd-30a5-42c5-9720-5aeab1cb747c
begin
	md"
**Coordenadas cartesianas**

**Punto A**
X1 : $(@bind xA Slider(-300.0:1.0:300.0, default=0.0))
X2: $(@bind yA Slider(-300.0:1.0:300.0, default=0.0))


**Punto B**
X1 : $(@bind xB Slider(-300.0:1.0:300.0, default=120.0))
X2: $(@bind yB Slider(-300.0:1.0:300.0, default=80.0))

"
end

# ╔═╡ cd801016-5fb0-43fc-92ea-9bafd3e29717
begin
	# L1 y L2 en el plano
	L1(dx,dy) = abs(dx) + abs(dy)
	L2(dx,dy) = sqrt(dx^2 + dy^2)

	Lchebychev(dx, dy) = max(abs(dx), abs(dy))	
	
	x1, y1, x2, y2 = xA, yA, xB, yB
	dx, dy = x2 - x1, y2 - y1
	
	d_L1 = L1(dx,dy)
	d_L2 = L2(dx,dy)
	d_chevyshov = Lchebychev(dx,dy)
	
	
	md"<-- Cálculos internos -->"
end

# ╔═╡ dce4a914-abc3-4500-a4cd-3b4dc7153330
md"
**Punto A = ($(xA), $(yA))**
	
**Punto B = ($(xB), $(yB))**"

# ╔═╡ 84ef6aca-9f1e-40bc-acb4-4a7de34e3775
# Puntos y caminos
begin
	xs_AB = [x1, x2]; ys_AB = [y1, y2]
	
	# Camino Manhattan (dos tramos ortogonales)
	xs_MAN = [x1, x2, x2]
	ys_MAN = [y1, y1, y2]

	
	p_dist = plot(; aspect_ratio=:equal, xlabel="X1", ylabel="X2")
	scatter!(p_dist, [x1, x2], [y1, y2], ms=6, label=["Punto A" "Punto B"])
	plot!(p_dist, xs_AB, ys_AB, label="Recta (Euclídea)")
	plot!(p_dist, xs_MAN, ys_MAN, ls=:dash, label="Ruta Manhattan")
	annotate!(p_dist, x1, y1, text("A", 10, :left))
	annotate!(p_dist, x2, y2, text("B", 10, :left))
	plot!(p_dist, title = "Comparación de distancias")
	p_dist
end

# ╔═╡ 1b14040a-a506-4420-a7a1-2736d5ccb0bc
begin
	md"""
	**Distancias:**  
	- **Manhattan (L1):** $(round(d_L1, digits=3))
	- **Euclídea (L2):** $(round(d_L2, digits=3))
	- **Chebyshov:** $(round(d_chevyshov, digits=3))
	"""

end

# ╔═╡ a209fd9b-a0a2-4954-81e8-ea0b3a2a282d
md"""
!!! tip "Para tener en cuenta"
	Vemos que las distancias cambian bastantes, por lo que pueden llevar a seleccionar vecinos diferentes en algunos casos.
"""

# ╔═╡ 58a44d15-0910-4b34-96cb-d839146c4af2
md"## Conceptos claves"

# ╔═╡ 67233bf8-8afa-4c12-8bae-d77cdd2ba8b8
md"Los conceptos claves de este notebook son:

* El KNN es un método que no se basa en definir una función que indique como se relacionan los datos, sino que predice directamente en base a los datos.
* Es un método local, presuponen que si las distancias son pequeñas (es decir, los puntos son parecidos), la variable dependiente será parecida.
* Se puede usar para regresión o clasificación.
* Es necesario definir el $K$ y la métrica de distancia.

"

# ╔═╡ Cell order:
# ╟─1988bf6c-988f-4238-a960-a7cc7b986a44
# ╟─48392ef4-cb55-4473-9905-8b9e61b38a24
# ╟─d8f52c6d-23c1-4dd6-8c5f-e1424719e4b6
# ╟─acb5e996-722c-11f0-063a-b7f2bb5d60c3
# ╟─f81aad0e-2089-4020-a8d8-1ed0746312c8
# ╟─4f61595d-6f51-4614-952c-d49173475e39
# ╟─275d670c-70fa-46df-9de1-9e6ef6d702fc
# ╟─a4bc4e34-02b1-4b74-8b6f-3bb0608d7949
# ╟─0e9f19f8-450c-4bae-b482-72cc7900e79a
# ╟─6c24c1ba-4aac-4aba-9eff-2454e4768fbe
# ╟─b8badc49-51b9-4530-acc6-327470af2bdf
# ╟─2a331f78-316d-4692-a070-8d5d42946177
# ╟─8f5ef0e1-872c-4de6-ab67-ae5390c3b812
# ╟─bcf8ee1c-031f-404f-819c-f7b2915fb1a5
# ╟─749a0495-48ac-4b17-8ffe-e9be8baa508e
# ╟─3226583d-af34-4eb5-9eaf-01dd83ac50c0
# ╟─6d7f4c7e-31d1-4df3-8214-7ea011a2bd6d
# ╟─ac6c394e-6423-4c34-9570-560c7c7b1560
# ╟─9ceab22d-6890-4c7b-87e5-ac1627934084
# ╟─694c6612-312a-4072-a61d-d42766f1323b
# ╟─12cbffdf-9c25-49d6-b28b-71d2835e0dd9
# ╟─24f0dddd-30a5-42c5-9720-5aeab1cb747c
# ╟─cd801016-5fb0-43fc-92ea-9bafd3e29717
# ╟─dce4a914-abc3-4500-a4cd-3b4dc7153330
# ╟─84ef6aca-9f1e-40bc-acb4-4a7de34e3775
# ╟─1b14040a-a506-4420-a7a1-2736d5ccb0bc
# ╟─a209fd9b-a0a2-4954-81e8-ea0b3a2a282d
# ╟─58a44d15-0910-4b34-96cb-d839146c4af2
# ╟─67233bf8-8afa-4c12-8bae-d77cdd2ba8b8
