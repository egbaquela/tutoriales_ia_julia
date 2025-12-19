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

# ╔═╡ 020eb1e1-0834-4537-be6d-e316019e9e37
begin
	using Pkg
	Pkg.activate()

	using PlutoUI, Random, Distributions, StatsBase, Statistics
	using DataFrames, Tables
	using MLJ, MLJModels, MLJDecisionTreeInterface, ScientificTypes
	using CategoricalArrays
	using Plots
	default(fmt=:png)

	TableOfContents(title="Contenido")
end	

# ╔═╡ a734a5ac-9324-11f0-1514-edd9c458392b
md"# Introducción a los ensambles"

# ╔═╡ e1729238-2ba3-4768-9de4-169c9fa3d351
md"## Setup"

# ╔═╡ 9d4fef24-135c-4cbb-bbc3-86f96ee6211a
md"## Un dataset raro"

# ╔═╡ 3f67fa0c-d902-4ef5-bfc1-0e57a4fb4b62
md"""
|Parámetro|Valor|
|---------|-----|
|Semilla|$(@bind seed Slider(1:1:2_000, default=1234, show_value=true))|
|Elementos por anillo|$(@bind n_per_ring Slider(200:50:2000, default=600, show_value=true))|
|σ|$(@bind σ Slider(0.1:0.1:2.0, default=0.6, show_value=true))|
|Cant. variables de ruido|$(@bind n_noise_features Slider(0:1:20, default=20, show_value=true))|
"""

# ╔═╡ b556913f-4ce3-4ea0-8357-ed22c29c9065
begin
	function make_concentric_rings(; n_per_ring=600, n_noise_features=8, radii=(0.8, 1.6),
		                            σ=0.1, seed=1234)
		Random.seed!(seed)
		n = 2n_per_ring
		θ = 2π * rand(n)
		r = [radii[1] .+ σ*randn(n_per_ring); radii[2] .+ σ*randn(n_per_ring)]
		x1 = r .* cos.(θ)
		x2 = r .* sin.(θ)
		y  = vcat(zeros(Int32, n_per_ring), ones(Int32, n_per_ring)) # 0: inner, 1: outer (clases alternadas)

		# Features de ruido (gaussianas i.i.d), para que RF con submuestreo de features brille:
		noise = [randn(n) for _ in 1:n_noise_features]
		X = hcat(x1, x2, noise...)
		names = union(["x1","x2"], ["z$(k)" for k in 1:n_noise_features])
		df = DataFrame(X, Symbol.(names))
		df.y = y  # MLJ pide categórico para clasificación
		coerce!(df, :y  => ScientificTypes.OrderedFactor)
		return df
	end

	df = make_concentric_rings(n_per_ring=n_per_ring,
	                           n_noise_features=n_noise_features,
	                           seed=seed, σ=σ)

	first(df, 5)
end

# ╔═╡ 99c57140-0c22-436e-bbdf-d9dc561ba832
md"""
**Dimensiones del dataset:** $(size(df))  
**Clase:** `y` (binaria)
"""

# ╔═╡ 929750ab-db03-4117-88f0-950c34ad8a87
begin
	scatter(df.x1, df.x2, group=df.y, ms=2.5, leg=false,
	        xlabel="x1", ylabel="x2", title="Dataset: anillos concéntricos (x1,x2)")
end

# ╔═╡ 76100140-6a32-4eb2-b833-b94b9182f986
md"Tamaño del conjunto de entrenamiento: $(@bind train_ratio Slider(0.5:0.05:0.9, default=0.7, show_value=true))"

# ╔═╡ a600f11d-14c4-4b2a-aeea-ea6ef66c313d
begin
	n = nrow(df)
	train_idx = sample(1:n, round(Int, train_ratio*n), replace=false)
	test_idx  = setdiff(1:n, train_idx)

	X = select(df, Not(:y))
	y = df.y

	Xtrain = X[train_idx, :]
	ytrain = y[train_idx]

	Xtest  = X[test_idx, :]
	ytest  = y[test_idx]

	nothing
end

# ╔═╡ 9e795fcd-55a1-4678-a0fd-797984e3b414
md"Máxima profundidad del árbol: $(@bind max_depth Slider(2:1:20, default=6, show_value=true))"

# ╔═╡ 919796d0-e3ab-4100-b4ac-0c9af0214461
begin
	tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=2, min_samples_leaf=1)
	mach_tree = machine(tree, Xtrain, ytrain)
	MLJ.fit!(mach_tree)
	yhat_tree = MLJ.predict(mach_tree, Xtest)
	acc_tree  = MLJ.accuracy(mode.(yhat_tree), ytest)

	md"**Accuracy árbol (test):** $(round(acc_tree, digits=4))   —   *max_depth* = $max_depth"
end

# ╔═╡ d1fff1e6-8f34-4354-8848-58dec6c253f5
begin
    # Frontera de decisión en 2D (visual: solo x1,x2)
    x1grid = range(extrema(df.x1)..., length=200)
    x2grid = range(extrema(df.x2)..., length=200)
    grid = [(x,y) for x in x1grid, y in x2grid]
    grid_df = DataFrame(x1 = [p[1] for p in grid[:]], x2 = [p[2] for p in grid[:]])

    # Asegurar que grid_df tenga TODAS las columnas de X; las faltantes en 0
    need_cols = setdiff(names(X), names(grid_df))  # ambos son Vector{Symbol}
    for c in need_cols
        grid_df[!, c] = zeros(nrow(grid_df))
    end

    # Predicción sobre la grilla
    ygrid_tree_dist = MLJ.predict(mach_tree, grid_df)  # Vector{UnivariateFinite}
    ygrid_tree = mode.(ygrid_tree_dist)                # Vector{CategoricalValue}

    # Tomar el nivel "positivo" tal como está en ytrain (2º nivel para binaria)
    pos = levels(ytrain)[end]  # asumiendo niveles ordenados ["0","1"] ⇒ pos = "1"
    ygrid01 = map(x -> x == pos ? 1 : 0, ygrid_tree)

    p_tree = heatmap(x1grid, x2grid, reshape(ygrid01, 200, 200)',
                     alpha=0.25, legend=false, xlabel="x1", ylabel="x2",
                     title="Frontera Árbol (solo visualizando ejes x1,x2)")
    scatter!(p_tree, df.x1, df.x2, group=df.y, ms=2, leg=false)
    p_tree
end


# ╔═╡ d2a9441e-849f-4a63-8b54-763bf34ffc79
md"
|Parametro|Valor|
|---------|-----|
|Cantidad de árboles|$(@bind n_trees Slider(1:1:400, default=50, show_value=true))|
|Fracción de muestreo|$(@bind sample_ratio Slider(0.3:0.05:1.0, default=0.7, show_value=true))|
"

# ╔═╡ 28c02eb0-19d1-4c45-ab28-e994157d2667
begin
	"""
	Entrena un ensemble por **bagging a mano**:
	- n_trees: cantidad de árboles
	- sample_ratio: fracción de datos de entrenamiento por árbol (con reemplazo)
	- max_depth: profundidad del árbol base
	- n_subfeatures: 0 o missing ⇒ usa todas (bagging puro). Si >0 ⇒ RF-style (*por split*, DecisionTree lo maneja internamente).
	"""
	function manual_bagging_classifier(Xtrain::DataFrame, ytrain, Xtest::DataFrame;
			n_trees::Int=50, sample_ratio::Float64=0.7,
			max_depth::Int=6, n_subfeatures::Union{Int,Missing}=missing,
			seed::Int=12345)

		Random.seed!(seed)
		n = nrow(Xtrain)
		Tree = @load DecisionTreeClassifier pkg=DecisionTree verbosity=0

		# Armamos los modelos base
		machines = Machine[]
		for t in 1:n_trees
			idx = sample(1:n, round(Int, sample_ratio*n), replace=true)
			Xb = Xtrain[idx, :]
			yb = ytrain[idx]

			model = isnothing(n_subfeatures) || n_subfeatures === missing ?
			    Tree(max_depth=max_depth) :
			    Tree(max_depth=max_depth, n_subfeatures=n_subfeatures)

			push!(machines, fit!(machine(model, Xb, yb)))
		end

		# Predicciones por voto mayoritario
		preds = [MLJ.predict(m, Xtest) for m in machines]  # vector de CategoricalArrays
		# Convertimos a matriz de códigos 0/1 para voto:
		classes = levels(ytrain)
		@assert length(classes) == 2 "Ejemplo implementado para binaria."
		cls1 = classes[2]  # asumiendo niveles ordenados ["0","1"] => tomamos "1" como positivo

		# Conteo de votos por la clase positiva:
		votes_pos = zeros(Int, nrow(Xtest))
		for p in preds
			votes_pos .+= (mode.(p) .== cls1)
		end
		yhat = ifelse.(votes_pos .>= ceil.(n_trees/2), cls1, classes[1])

		return categorical(yhat), votes_pos ./ n_trees # también devuelvo proporción de votos (pseudo-prob)
	end

	yhat_bag, ppos_bag = manual_bagging_classifier(Xtrain, ytrain, Xtest;
		n_trees=n_trees, sample_ratio=sample_ratio, max_depth=max_depth, n_subfeatures=missing, seed=seed)

	acc_bag = MLJ.accuracy(yhat_bag, ytest)

	md"""
	**Bagging a mano (test accuracy):** $(round(acc_bag, digits=4))  
	Árbol base: *max_depth* = $max_depth — Árboles = $n_trees — *sample_ratio* = $sample_ratio
	"""
end

# ╔═╡ 129f6489-26ab-46b0-a920-66b0e7b39a8a
begin
	rf = RandomForestClassifier(n_trees=300, max_depth=max_depth)
	mach_rf = machine(rf, Xtrain, ytrain)
	MLJ.fit!(mach_rf)
	yhat_rf = MLJ.predict(mach_rf, Xtest)
	acc_rf  = MLJ.accuracy(mode.(yhat_rf), ytest)

	md"**Accuracy Random forest (test):** $(round(acc_rf, digits=4))"
end

# ╔═╡ Cell order:
# ╟─a734a5ac-9324-11f0-1514-edd9c458392b
# ╟─e1729238-2ba3-4768-9de4-169c9fa3d351
# ╠═020eb1e1-0834-4537-be6d-e316019e9e37
# ╟─9d4fef24-135c-4cbb-bbc3-86f96ee6211a
# ╟─3f67fa0c-d902-4ef5-bfc1-0e57a4fb4b62
# ╟─b556913f-4ce3-4ea0-8357-ed22c29c9065
# ╟─99c57140-0c22-436e-bbdf-d9dc561ba832
# ╟─929750ab-db03-4117-88f0-950c34ad8a87
# ╟─76100140-6a32-4eb2-b833-b94b9182f986
# ╠═a600f11d-14c4-4b2a-aeea-ea6ef66c313d
# ╟─9e795fcd-55a1-4678-a0fd-797984e3b414
# ╠═919796d0-e3ab-4100-b4ac-0c9af0214461
# ╟─d1fff1e6-8f34-4354-8848-58dec6c253f5
# ╟─d2a9441e-849f-4a63-8b54-763bf34ffc79
# ╠═28c02eb0-19d1-4c45-ab28-e994157d2667
# ╠═129f6489-26ab-46b0-a920-66b0e7b39a8a
