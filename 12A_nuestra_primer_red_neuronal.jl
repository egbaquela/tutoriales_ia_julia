### A Pluto.jl notebook ###
# v0.20.18

using Markdown
using InteractiveUtils

# ╔═╡ 25832ff4-5f7e-4297-804a-665ea80e823c
begin
	using Pkg
	Pkg.activate()

	using Images,Flux,BSON
	using PlutoUI

	TableOfContents(title="Contenido")
end

# ╔═╡ 8fed5aa6-475c-46f5-92e3-eaee6a62a657
begin
    # Acepta siempre los términos/licencias sin pedir confirmación
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    # (Opcional) elegí dónde guardar los datasets
    # ENV["DATADEPS_DIR"] = "/ruta/a/mis/datasets"
    using DataDeps
    using MLDatasets
end

# ╔═╡ 3016f4b8-a38a-11f0-2e29-4fbc9f2b1823
md"# Implementando nuestra primer red neuronal con Flux"

# ╔═╡ 381ddb18-e012-4e07-981e-25225a62308b
md"En este notebook vamos a implementar nuestra primera red neuronal, usando para ello la biblioteca [_Flux_](https://fluxml.ai/Flux.jl/stable/).

Los paquetes a utilizar en este notebook son:

* Images
* Flux
* BSON
* PlutoUI
* DataDeps
* MLDatasets

"

# ╔═╡ 05be1eac-54b8-4370-9377-5f33c4d94c2c
md"## Setup"

# ╔═╡ 1f010f1c-71bf-4831-84b3-55173a298595
md"## Entrenando el MNIST"

# ╔═╡ da6f59b5-ab8f-477b-bd53-1bd3459475ba
begin
	train_imgs, train_labels = MLDatasets.MNIST(split=:train)[:]
	test_imgs,  test_labels  = MLDatasets.MNIST(split=:test)[:]
	nothing
end

# ╔═╡ c00937f2-710d-4ad6-bb3b-648204507b66
Gray.(test_imgs[:,:,1]')

# ╔═╡ 5a757276-ae77-4ab0-beda-86738ffbf190
test_labels[1]

# ╔═╡ ea0d4d2c-fbdb-4efe-91c3-a7bbce5cc3d3
md"Creemos primero una función para medir la accuracy en nuestro dataset:"

# ╔═╡ 7a472165-b89e-4797-8fdc-89217eca3947
function accuracy(model, test_imgs, test_labels)
    correct = 0
    for index in 1:length(test_labels)
        probs = model(Flux.unsqueeze(test_imgs[:,:,index],dims=3))
        predicted_digit = argmax(probs)[1]-1
        if predicted_digit == test_labels[index]
            correct +=1
        end
    end
    return correct/length(test_labels)
end

# ╔═╡ 67ffa1bf-91d4-448f-a9aa-3ed422ceb6a5
md"Creemos un modelo (sin entrenar) de red neuronal:"

# ╔═╡ cb97693a-28f0-4899-8b47-c19e442142bb
model_v1 = Chain(
    Flux.flatten,
    Dense(784=>15,relu),
    Dense(15=>10,sigmoid),
    softmax
)

# ╔═╡ e3daadd8-b7da-4be5-9277-32f709516152
md"Probemos predecir uno de los valroes del conjunto de test:"

# ╔═╡ e9d376ff-c19d-4d9b-b39d-acf1ef0a8533
predict = model_v1(Flux.unsqueeze(test_imgs[:,:,1],dims=3))

# ╔═╡ 4f4a1f0a-18d4-4b54-be4f-f2b5cb7cfbc0
accuracy(model_v1, test_imgs, test_labels)

# ╔═╡ e2c37aef-ac19-4cc7-b755-07b49ad5e7a3
md"Cualquier, no tiene ni idea de que predecir. Claro, no está entrenada. Vamos a entrenarla:"

# ╔═╡ 53433272-0622-4bfd-8bea-ef14607c2e58
begin
	data_v1 = Flux.DataLoader((train_imgs,train_labels), shuffle=true, batchsize=1)
	
	optimizer_v1  = Flux.setup(Adam(), model_v1 )
	
	function loss_v1(model_v1 , x, y)
	    return Flux.crossentropy(model_v1(x),Flux.onehotbatch(y,0:9))
	end
	
	println(accuracy(model_v1 , train_imgs,train_labels))
	for epoch in 1:2
	    Flux.train!(loss_v1, model_v1, data_v1, optimizer_v1)
		println(accuracy(model_v1, train_imgs,train_labels))
	end
end

# ╔═╡ 750f7524-fd9c-4538-91cb-b77eff010a24
accuracy(model_v1, test_imgs, test_labels)

# ╔═╡ 88f22d6e-7a42-464a-a384-546b85fed49c
md"Guardemos ahora nuestra red neuronal un archivo con formato BSON:"

# ╔═╡ d7649a15-f257-4974-b8e3-39fa41029da2
BSON.@save "trained_models/red_neuronal_flux_simple.bson" model_v1

# ╔═╡ b664584f-b50c-43d1-8e5c-c49db1499d8d
md"""
!!! info "Actividad para el alumno"
	Traten de repetir el entrenamiento cambiando el valor de _batchsize_, en el _DataLoader_, de 1 a 64. ¿Qué cambios notan en la velocidad del entrenamiento?, ¿y en los resultados?
"""

# ╔═╡ Cell order:
# ╟─3016f4b8-a38a-11f0-2e29-4fbc9f2b1823
# ╟─381ddb18-e012-4e07-981e-25225a62308b
# ╟─05be1eac-54b8-4370-9377-5f33c4d94c2c
# ╠═25832ff4-5f7e-4297-804a-665ea80e823c
# ╠═8fed5aa6-475c-46f5-92e3-eaee6a62a657
# ╟─1f010f1c-71bf-4831-84b3-55173a298595
# ╠═da6f59b5-ab8f-477b-bd53-1bd3459475ba
# ╠═c00937f2-710d-4ad6-bb3b-648204507b66
# ╠═5a757276-ae77-4ab0-beda-86738ffbf190
# ╟─ea0d4d2c-fbdb-4efe-91c3-a7bbce5cc3d3
# ╠═7a472165-b89e-4797-8fdc-89217eca3947
# ╟─67ffa1bf-91d4-448f-a9aa-3ed422ceb6a5
# ╠═cb97693a-28f0-4899-8b47-c19e442142bb
# ╟─e3daadd8-b7da-4be5-9277-32f709516152
# ╠═e9d376ff-c19d-4d9b-b39d-acf1ef0a8533
# ╠═4f4a1f0a-18d4-4b54-be4f-f2b5cb7cfbc0
# ╟─e2c37aef-ac19-4cc7-b755-07b49ad5e7a3
# ╠═53433272-0622-4bfd-8bea-ef14607c2e58
# ╠═750f7524-fd9c-4538-91cb-b77eff010a24
# ╟─88f22d6e-7a42-464a-a384-546b85fed49c
# ╠═d7649a15-f257-4974-b8e3-39fa41029da2
# ╟─b664584f-b50c-43d1-8e5c-c49db1499d8d
