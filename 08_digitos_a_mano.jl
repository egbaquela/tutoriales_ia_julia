### A Pluto.jl notebook ###
# v0.20.9

using Markdown
using InteractiveUtils

# ╔═╡ e6c5d66f-40bd-4833-bc08-ea05ca7f187b
begin
	using Pkg
	Pkg.activate()
	
	using PlutoUI
	using HypertextLiteral

	TableOfContents(title="Contenido")
end

# ╔═╡ 585d9b13-d67b-4c4b-a74b-bf64621d2d74
begin
	using MLJ
	using MLJLinearModels
	using DecisionTree
	using NearestNeighborModels
	using CategoricalArrays
	using Random, Statistics, StatsBase, Distributions
	using DataFrames, Tables
	using Plots
	using ScientificTypes
	using Optim
end

# ╔═╡ 0968ab81-cffc-4d65-afa5-a117ec93f391
### Aceptación automática para DataDeps (¡antes de usar MLDatasets!)
begin
    # Acepta siempre los términos/licencias sin pedir confirmación
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    # (Opcional) elegí dónde guardar los datasets
    # ENV["DATADEPS_DIR"] = "/ruta/a/mis/datasets"
    using DataDeps
    using MLDatasets
end

# ╔═╡ 250a3157-b556-4b8e-87cd-884d3e7ee69b
md"# Clasificando dígitos escritos a mano"

# ╔═╡ 6fb9a062-1aac-4c4b-975d-cfb3acea9720
md"En este notebook vamos a realizar una primera introducción al reconocimiento de imágenes, utilizando para ello el dataset [MNIST](https://es.wikipedia.org/wiki/Base_de_datos_MNIST).

Los paquetes que vamos a utilizar en este notebook son:

* PlutoUI
* HypertextLiteral
* MLJ
* MLJLinearModels
* DecisionTree
* NearestNeighborModels
* MLDatasets
* DataDeps
* CategoricalArrays
* Random
* Statistics
* StatsBase
* Distributions
* DataFrames
* Tables
* Plots
* Optim
"

# ╔═╡ 12f82652-cf7b-4b9e-ade0-d97336c433c4
md"## Setup"

# ╔═╡ 0d8b29ec-a8ff-4ebe-9372-74b9ccebb97b
md"## Obtención de datos"

# ╔═╡ 7dbf1afc-52f5-4055-8d34-2d811ce7987f
md"Muchos datasets de uso común, para aprendizaje o investigación, suelen estar almacenados en paquetes para su fácil acceso, evitando el tener que compartir archivos adicionales. En este caso, vamos a obtener el dataset _MNIST_ del paquete _MLDatasets_."

# ╔═╡ 83def174-795a-4f33-9709-97034e92d9bf
begin
# Para reproducibilidad:
Random.seed!(123)

# Cargar MNIST (train y test)
train_imgs, train_labels = MLDatasets.MNIST(split=:train)[:]
test_imgs,  test_labels  = MLDatasets.MNIST(split=:test)[:]

# Convertir a Float32 y normalizar a [0,1]
normf(x) = Float64.(x)
train_imgs_f = normf(train_imgs)
test_imgs_f  = normf(test_imgs)

end

# ╔═╡ a96d862a-6721-4a4f-81d3-eeb51e50fabd
DataFrame(train_imgs[:,:,1]', :auto)

# ╔═╡ f4fd99e4-02fc-4321-9e42-df2f93f2d57e
plot(Gray.(train_imgs[:,:,1]'))

# ╔═╡ 479452b9-8581-419f-8939-5310ec97d0db
begin
	# Aplanar a (n_samples, 784)
	#flatten(imgs) = reshape(imgs, size(imgs,3), :)
	flatten(imgs) = reshape(permutedims(imgs, (3, 1, 2)), size(imgs, 3), :)
	Xtrain = flatten(train_imgs_f)       # 60000 × 784
	Xtest  = flatten(test_imgs_f)        # 10000 × 784
	
	# Etiquetas como categóricas (necesario para MLJ)
	ytrain = coerce(categorical(train_labels),  Multiclass)
	ytest  = coerce(categorical(test_labels),  Multiclass)
	
	(size(Xtrain), eltype(Xtrain), levels(ytrain)[1:5])
end

# ╔═╡ 94253e5a-56c7-4687-b03b-c28052aad064
begin
	ntrain = 50000
	
	# Subconjunto aleatorio de entrenamiento de tamaño ntrain
	idx = 1:ntrain
	Xtr = DataFrame(Xtrain[idx, :], :auto)
	ytr = ytrain[idx]
	
	Xte = DataFrame(Xtest, :auto)
	yte = ytest
	
	(size(Xtr), size(Xte))
end

# ╔═╡ bed0fde1-b859-458c-964f-d9a67bcf5b52
Xtr

# ╔═╡ 310e37f3-f349-4032-8783-5e492526aef0
Xte

# ╔═╡ cc31dd0e-5678-41d6-aba1-abea956c5948
begin
	opt = Optim.Options(iterations = 2000, g_tol = 1e-8, f_tol = 1e-8, allow_f_increases = false)
    solver = MLJLinearModels.LBFGS(optim_options = opt)

	# Pipelines:
	logit_model =  MultinomialClassifier(solver=solver,penalty=:l2, lambda=0.001,fit_intercept=true)
	#logit_model =  KNNClassifier(K=7)
	logit_pipe  =  logit_model

end

# ╔═╡ a4a43a59-a90b-408f-a392-b36db8651a81
ScientificTypes.schema(Xtr)

# ╔═╡ 80cc17a7-6a05-4176-a4e3-841ad0012025
logit_mach = machine(logit_pipe, Xtr, ytr)

# ╔═╡ c8b9e681-657e-4af2-98a2-3294da265ae7
MLJ.fit!(logit_mach)

# ╔═╡ 34d21389-aa55-4fee-a710-484597cb7c5f
begin
	# Predicciones de clase (modo) en test
	ŷ_logit_dist = MLJ.predict(logit_mach, Xte[1:100,:])
	ŷ_logit = mode.(ŷ_logit_dist)
	acc_logit = MLJ.accuracy(ŷ_logit, yte[1:100])

end

# ╔═╡ 7bcad630-4bfb-4e97-9214-57a49284a867
ŷ_logit_dist[5]

# ╔═╡ fdf0fa78-4549-46af-aa93-d730461c359a
ŷ_logit[5]

# ╔═╡ 9bfc733f-7ef2-4188-8565-759c57424343
yte[5]

# ╔═╡ 95d25baa-f337-4468-850b-c848aa678351
plot(Gray.(test_imgs[:,:,5]'))

# ╔═╡ 0fd67c6a-5105-45ec-a728-8210abd818a5
with_terminal() do

# Asumimos que ya tenés Xtr, ytr, Xte, yte y el modelo entrenado `logit_mach`


println("Niveles ytr: ", levels(ytr))
println("Niveles yte: ", levels(yte))
println("¿Niveles iguales?: ", levels(ytr) == levels(yte))


println("Clases predichas únicas: ", unique(ŷ_logit))
println("Frecuencias predichas: "); println(countmap(ŷ_logit))
println("Baseline (clase mayoritaria en test): ", maximum(values(countmap(yte))) / length(yte))

println("Accuracy logística: ", acc_logit)
end

# ╔═╡ Cell order:
# ╟─250a3157-b556-4b8e-87cd-884d3e7ee69b
# ╟─6fb9a062-1aac-4c4b-975d-cfb3acea9720
# ╟─12f82652-cf7b-4b9e-ade0-d97336c433c4
# ╠═e6c5d66f-40bd-4833-bc08-ea05ca7f187b
# ╠═585d9b13-d67b-4c4b-a74b-bf64621d2d74
# ╠═0968ab81-cffc-4d65-afa5-a117ec93f391
# ╟─0d8b29ec-a8ff-4ebe-9372-74b9ccebb97b
# ╟─7dbf1afc-52f5-4055-8d34-2d811ce7987f
# ╠═83def174-795a-4f33-9709-97034e92d9bf
# ╠═a96d862a-6721-4a4f-81d3-eeb51e50fabd
# ╠═f4fd99e4-02fc-4321-9e42-df2f93f2d57e
# ╠═479452b9-8581-419f-8939-5310ec97d0db
# ╠═94253e5a-56c7-4687-b03b-c28052aad064
# ╠═bed0fde1-b859-458c-964f-d9a67bcf5b52
# ╠═310e37f3-f349-4032-8783-5e492526aef0
# ╠═cc31dd0e-5678-41d6-aba1-abea956c5948
# ╠═a4a43a59-a90b-408f-a392-b36db8651a81
# ╠═80cc17a7-6a05-4176-a4e3-841ad0012025
# ╠═c8b9e681-657e-4af2-98a2-3294da265ae7
# ╠═34d21389-aa55-4fee-a710-484597cb7c5f
# ╠═7bcad630-4bfb-4e97-9214-57a49284a867
# ╠═fdf0fa78-4549-46af-aa93-d730461c359a
# ╠═9bfc733f-7ef2-4188-8565-759c57424343
# ╠═95d25baa-f337-4468-850b-c848aa678351
# ╠═0fd67c6a-5105-45ec-a728-8210abd818a5
