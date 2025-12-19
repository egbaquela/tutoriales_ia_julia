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
md"# Árboles de decisión"

# ╔═╡ 48392ef4-cb55-4473-9905-8b9e61b38a24
md"En este notebook veremos un método _algorítmico_ de aprendizaje supervisado para regresión y clasificación. A diferencias de la regresión lineal la regresión logística y los algoritmos de vecinos mas cercanos, las cuales ya estuvimos viendo, no partimos aquí de una ley analítica cuyos parámetros calculamos durante el entrenamiento a partir de datos, ni tampoco de una medida de similaritud respecto de los datos usados en el entrenamiento, sinó que usamos una estructura jerárquica en forma de árbol para predecir la variable de interés.

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
md"## Árboles de decisión"

# ╔═╡ 4f61595d-6f51-4614-952c-d49173475e39
md"Un [árbol de decisión](https://es.wikipedia.org/wiki/%C3%81rbol_de_decisi%C3%B3n) es un proceso de secuencias de comparaciones que utilizamos para arribar a una decisión a tomar. Por ejemplo, analicemos el caso de la promoción directa:

> 1. Si mínimo(notas) >= 6
>     1. Verdadero -> Promociona
>     2. Falso -> Si mínimo(notas) >= 4
>         1. Verdadero -> Regulariza  
>         2. Falso -> Queda libre

Tenemos un primer condicional de cuyo resultado depende si tenemos un resultado (_promociona_) o si debemos hacer otro chequeo. Es decir, del primer **nodo**, el **nodo raíz**, salen dos **ramas**, una termina en una **hoja**, el otro en otro nodo, el cual a su vez se divide en dos ramas, cada una ***terminal** (es decir, no tienen nodos sino hojas).

Un árbol es fácil de entender, fácil de comunicar, pero no tan fácil de construir. Tiene una gran desventaja, es muy sensible a los datos utilizados para entrenarlo.

El entrenamiento del árbol consiste en determinar qué predictor usar como nodo raíz y contra qué valor compararlo. Luego, se itera calculando si, en cada rama, crear un nuevo nodo de decisión mejora mi predicción o no. Si la mejora, se crea un nodo, se lecciona una variable y se repite el proceso. Si no se crea el nodo, se crea una hoja, cuyo valor es el promedio de los valores de la variable dependiente asociada a ese nodo (si estoy haciendo regresión) o la moda (si estoy haciendo clasificación).

¿Pero cómo predecimos con un árbol? Fácil, nos paramos en el nodo raíz y seguimos la secuencia de comparaciones.

Construyamos algunos árboles a mano y veamos como predicen:

"

# ╔═╡ 275d670c-70fa-46df-9de1-9e6ef6d702fc
begin
    Random.seed!(123)

    # Tres nubes gaussianas en 2D
    n_per = 60
    μs = [[-2.0, -2.0], [2.2, 2.0], [-2.0, 2.2]]
    Σ  = [0.8 0.0; 0.0 0.8]

    function sample_gauss(μ, Σ, n)
        A = cholesky(Symmetric(Σ)).L
        [μ .+ A*randn(2) for _ in 1:n]
    end

    X1 = sample_gauss(μs[1], Σ, n_per)
    X2 = sample_gauss(μs[2], Σ, n_per)
    X3 = sample_gauss(μs[3], Σ, n_per)

    X = reduce(vcat, (X1, X2, X3))                 # Vector de vectores 2D
    Y = vcat(fill(1, n_per), fill(2, n_per), fill(3, n_per))  # etiquetas
    Xmat = reduce(hcat, X)                          # 2×N

    classes = sort(unique(Y))                       # [1,2,3]
    colors  = Dict(1 => :royalblue, 2 => :seagreen, 3 => :tomato)

    x1min, x1max = minimum(Xmat[1,:]) - 2, maximum(Xmat[1,:]) + 2
    x2min, x2max = minimum(Xmat[2,:]) - 2, maximum(Xmat[2,:]) + 2
	md"<-- Cálculos internos -->"
end

# ╔═╡ a4bc4e34-02b1-4b74-8b6f-3bb0608d7949
md"**Controles**"

# ╔═╡ 0e9f19f8-450c-4bae-b482-72cc7900e79a
begin
	root_node_dict = Dict("X1" => :x1, "X2" => :x2)
	md"""
	Nodo raiz: $(@bind feat_root Select(["X1", "X2"]))
	"""
end

# ╔═╡ 6df74ef6-8704-468a-b614-a6070e0c288c
begin
	rng_root = feat_root == :x1 ? range(x1min, x1max, length=401) : range(x2min, x2max, length=401)
	def_root = feat_root == :x1 ? median(Xmat[1,:]) : median(Xmat[2,:])
	md"""
	Menor o igual a: $(@bind thr_root  Slider(rng_root, default=def_root, show_value=true))
	"""
end

# ╔═╡ 60171e2a-5d6a-40ee-96ad-fa4a463f02a3
begin
	# Nodo izquierdo (subárbol LEFT de la raíz)
	feat_left_dict = Dict("X1" => :x1, "X2" => :x2)
	md"""
	Nodo izquierdo: $(@bind feat_left Select(["X1", "X2"]))
	"""
end

# ╔═╡ 51147b9d-bc11-4366-b165-f5f89f8d91c0
begin
	rng_left = feat_left == :x1 ? range(x1min, x1max, length=401) : range(x2min, x2max, length=401)
	def_left = feat_left == :x1 ? median(Xmat[1,:]) : median(Xmat[2,:])
	md"""
	Menor o igual a: $(@bind thr_left  Slider(rng_left, default=def_left, show_value=true))
	"""
end

# ╔═╡ 008ec3c8-4772-474f-a0b7-cdb822252c50
begin
	# Nodo derecho (subárbol RIGHT de la raíz)
	feat_right_dict = Dict("X1" => :x1, "X2" => :x2)
	md"""
	Nodo derecho: $(@bind feat_right Select(["X1", "X2"]))
	"""
end

# ╔═╡ 9bc27c30-408f-4e07-bc50-3404d71ad9f8
begin
	rng_right = feat_right == :x1 ? range(x1min, x1max, length=401) : range(x2min, x2max, length=401)
	def_right = feat_right == :x1 ? median(Xmat[1,:]) : median(Xmat[2,:])
	md"""
	Menor o igual a: $(@bind thr_right Slider(rng_right, default=def_right, show_value=true))
	"""
end

# ╔═╡ b4ad6cb1-20c7-4132-aedf-fe71b04cac44
md"Resolución: $(@bind res Slider(10:10:160, default=30, show_value=true))"

# ╔═╡ d1038ddb-f00e-442c-8346-ac0ac9b5bff0
begin
	# Dado (x,y), qué hoja (1..4) cae según las reglas
	# Mapeo: 1 = (root LEFT)  & (left-node <=)   → "LL"
	#        2 = (root LEFT)  & (left-node  >)   → "LR"
	#        3 = (root RIGHT) & (right-node <=)  → "RL"
	#        4 = (root RIGHT) & (right-node  >)  → "RR"
	function leaf_id(x, y, feat_root, thr_root, feat_left, thr_left, feat_right, thr_right)
	    go_left  = (feat_root == :x1) ? (x <= thr_root) : (y <= thr_root)
	    if go_left
	        left_left = (feat_left == :x1) ? (x <= thr_left) : (y <= thr_left)
	        return left_left ? 1 : 2
	    else
	        right_left = (feat_right == :x1) ? (x <= thr_right) : (y <= thr_right)
	        return right_left ? 3 : 4
	    end
	end
	
	# Mayoría por hoja (si no hay puntos en una hoja, asigna la clase 1 por defecto)
	function leaf_majorities(Xmat, Y, feat_root, thr_root, feat_left, thr_left, feat_right, thr_right, classes)
	    counts = [Dict(c => 0 for c in classes) for _ in 1:4]  # 4 hojas
	    N = size(Xmat, 2)
	    for j in 1:N
	        xj, yj = Xmat[1,j], Xmat[2,j]
	        lid = leaf_id(xj, yj, feat_root, thr_root, feat_left, thr_left, feat_right, thr_right)
	        counts[lid][Y[j]] += 1
	    end
	    # argmax por hoja (empate → clase de menor id)
	    majors = similar(fill(0, 4))
	    for h in 1:4
	        if sum(values(counts[h])) == 0
	            majors[h] = first(classes)
	        else
	            best_c, best_cnt = first(classes), -1
	            for c in classes
	                cnt = counts[h][c]
	                if cnt > best_cnt || (cnt == best_cnt && c < best_c)
	                    best_c, best_cnt = c, cnt
	                end
	            end
	            majors[h] = best_c
	        end
	    end
	    return majors, counts
	end
	md"<-- Cálculos Internos -->"
end

# ╔═╡ 74da191b-109b-4a8d-85e4-71729c0b5a06
begin
    # Clases por hoja según mayoría de puntos de entrenamiento
    majors, leaf_counts = leaf_majorities(Xmat, Y, root_node_dict[feat_root], thr_root, feat_left_dict[feat_left], thr_left, feat_right_dict[feat_right], thr_right, classes)

    # Pintamos el plano: para cada celda → hoja → clase de esa hoja
    x1s = range(x1min, x1max, length=res)
    x2s = range(x2min, x2max, length=res)
    Z  = Array{Int}(undef, length(x1s), length(x2s))

    for (i, x1v) in enumerate(x1s), (j, x2v) in enumerate(x2s)
        lid = leaf_id(x1v, x2v, root_node_dict[feat_root], thr_root, feat_left_dict[feat_left], thr_left, feat_right_dict[feat_right], thr_right)
        Z[i, j] = majors[lid]
    end

    (x1s, x2s, Z, majors, leaf_counts)
	md"<-- Cálculos Internos -->"
end

# ╔═╡ 93ff933b-fdac-41c7-b97d-7c0a3a1f9cf0
begin

	function draw_split!(p; orientation::Symbol, thr::Real,
	                     xlo::Real, xhi::Real, ylo::Real, yhi::Real,
	                     ls=:dash, lw=2, color=:black, label=nothing)
	    if orientation === :x1
	        (xlo <= thr <= xhi) || return p
	        plot!(p, [thr, thr], [ylo, yhi]; ls=ls, lw=lw, color=color, label=label)
	    else
	        (ylo <= thr <= yhi) || return p
	        plot!(p, [xlo, xhi], [thr, thr]; ls=ls, lw=lw, color=color, label=label)
	    end
	    return p
	end
	
	palette = cgrad([colors[c] for c in classes]; categorical=true)
	
	p = heatmap(
	    x1s, x2s, Z';
	    color = palette,
	    clims = (1, length(classes)),    # fija el rango de clases
	    colorbar = false,
	    xlabel = "X1", ylabel = "X2",
	    title = "Árbol binario — regiones por hoja",
	    aspect_ratio = :equal, alpha = 0.35
	)

    # Puntos de entrenamiento por clase
    for c in classes
        inds = findall(==(c), Y)
        scatter!(p, Xmat[1, inds], Xmat[2, inds];
                 ms=5, label="Clase $c", color=colors[c])
    end

	#=
    # Líneas de split (completas para claridad)
    if root_node_dict[feat_root] == :x1
        vline!(p, [thr_root]; ls=:dash, lw=2, color=:black, label="Split raíz: X1 ≤ $(round(thr_root, digits = 2))")
    else
        hline!(p, [thr_root]; ls=:dash, lw=2, color=:black, label="Split raíz: X2 ≤ $(round(thr_root, digits = 2))")
    end

    if feat_left_dict[feat_left] == :x1
        vline!(p, [thr_left];  ls=:dot, lw=2, color=:gray, label="Split nodo izq: X1 ≤ $(round(thr_left, digits = 2))")
    else
        hline!(p, [thr_left];  ls=:dot, lw=2, color=:gray, label="Split nodo izq: X2 ≤ $(round(thr_left, digits = 2))")
    end

    if feat_right_dict[feat_right] == :x1
        vline!(p, [thr_right]; ls=:dot, lw=2, color=:gray, label="Split nodo der: X1 ≤ $(round(thr_right, digits = 2))")
    else
        hline!(p, [thr_right]; ls=:dot, lw=2, color=:gray, label="Split nodo der: X2 ≤ $(round(thr_right, digits = 2))")
    end
	=#


    # ---- Líneas de split recortadas ----
    # Rectángulo total
    xlo, xhi = x1min, x1max
    ylo, yhi = x2min, x2max

    # Raíz: barra todo el plano
    if root_node_dict[feat_root] == :x1
        draw_split!(p; orientation=:x1, thr=thr_root, xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi, ls=:dash, lw=2, color=:black, label="Root: X1 ≤ $(round(thr_root,digits=2))")
        # Subregiones
        left_rect  = (xlo, thr_root, ylo, yhi)
        right_rect = (thr_root, xhi, ylo, yhi)
    else
        draw_split!(p; orientation=:x2, thr=thr_root, xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi, ls=:dash, lw=2, color=:black, label="Root: X2 ≤ $(round(thr_root,digits=2))")
        left_rect  = (xlo, xhi, ylo, thr_root)
        right_rect = (xlo, xhi, thr_root, yhi)
    end

    # Nodo izquierdo: solo dentro del subplano izquierdo
    lxlo, lxhi, lylo, lyhi = left_rect
    if feat_left_dict[feat_left] == :x1
        draw_split!(p; orientation=:x1, thr=thr_left, xlo=lxlo, xhi=lxhi, ylo=lylo, yhi=lyhi, ls=:dot, lw=2, color=:gray, label="Left: X1 ≤ $(round(thr_left,digits=2))")
    else
        draw_split!(p; orientation=:x2, thr=thr_left, xlo=lxlo, xhi=lxhi, ylo=lylo, yhi=lyhi, ls=:dot, lw=2, color=:gray, label="Left: X2 ≤ $(round(thr_left,digits=2))")
    end

    # Nodo derecho: solo dentro del subplano derecho
    rxlo, rxhi, rylo, ryhi = right_rect
    if feat_right_dict[feat_right] == :x1
        draw_split!(p; orientation=:x1, thr=thr_right, xlo=rxlo, xhi=rxhi, ylo=rylo, yhi=ryhi, ls=:dot, lw=2, color=:gray, label="Right: X1 ≤ $(round(thr_right,digits=2))")
    else
        draw_split!(p; orientation=:x2, thr=thr_right, xlo=rxlo, xhi=rxhi, ylo=rylo, yhi=ryhi, ls=:dot, lw=2, color=:gray, label="Right: X2 ≤ $(round(thr_right,digits=2))")
    end

    p
end

# ╔═╡ 85da43e3-ca48-4e20-9eb7-a7bc40fcf3c7
md"""
**Clase por hoja (mayoría):**
- Si $(feat_root) <= $(thr_root):
  - Si $(feat_left) <= $(thr_left)
    - **Hoja 1**: Clase $(majors[1])
  - en caso contrario:
    - **Hoja 2**: Clase $(majors[2])
- en caso contrario:
  - Si $(feat_right) <= $(thr_right):
    - **Hoja 3**: Clase $(majors[3])
  - en caso contrario:
    - **Hoja 4**: Clase $(majors[4])

> *Tip:* mové los umbrales para ver cómo cambian las regiones y las mayorías.
"""

# ╔═╡ 58a44d15-0910-4b34-96cb-d839146c4af2
md"## Conceptos claves"

# ╔═╡ 67233bf8-8afa-4c12-8bae-d77cdd2ba8b8
md"Los conceptos claves de este notebook son:

* El árbol es un método _interpretable_, fácil de usar.
* Muy sensible a los datos usados para entrenarlos.
* Es necesario definir la profundidad máxima que queremos darle al árbol, para que no se vuelva muy grande.

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
# ╟─6df74ef6-8704-468a-b614-a6070e0c288c
# ╟─60171e2a-5d6a-40ee-96ad-fa4a463f02a3
# ╟─51147b9d-bc11-4366-b165-f5f89f8d91c0
# ╟─008ec3c8-4772-474f-a0b7-cdb822252c50
# ╟─9bc27c30-408f-4e07-bc50-3404d71ad9f8
# ╟─b4ad6cb1-20c7-4132-aedf-fe71b04cac44
# ╟─d1038ddb-f00e-442c-8346-ac0ac9b5bff0
# ╟─74da191b-109b-4a8d-85e4-71729c0b5a06
# ╟─93ff933b-fdac-41c7-b97d-7c0a3a1f9cf0
# ╟─85da43e3-ca48-4e20-9eb7-a7bc40fcf3c7
# ╟─58a44d15-0910-4b34-96cb-d839146c4af2
# ╟─67233bf8-8afa-4c12-8bae-d77cdd2ba8b8
