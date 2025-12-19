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
md"# Implementando nuestra primer red neuronal convolucional con Flux"

# ╔═╡ 381ddb18-e012-4e07-981e-25225a62308b
md"En este notebook vamos a implementar nuestra primera red neuronal convolucional, usando para ello la biblioteca [_Flux_](https://fluxml.ai/Flux.jl/stable/). Vamos a utilizar nuevamente el MNIST, y compararemos el proceso de entrenamiento y los resultados con la red _fully connected_ sin convolución que hicimos unos notebooks atrás.

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
md"## Cargando los datos del MNIST"

# ╔═╡ da6f59b5-ab8f-477b-bd53-1bd3459475ba
begin
	train_imgs, train_labels = MLDatasets.MNIST(split=:train)[:]
	test_imgs,  test_labels  = MLDatasets.MNIST(split=:test)[:]

	# Para que las capas de convolución funcionen, necesito especificar el canal.
	train_imgs = reshape(train_imgs, 28, 28, 1, :)
	test_imgs = reshape(test_imgs, 28, 28, 1, :)
	nothing
end

# ╔═╡ c00937f2-710d-4ad6-bb3b-648204507b66
Gray.(test_imgs[:,:,1, 1]')

# ╔═╡ 5a757276-ae77-4ab0-beda-86738ffbf190
test_labels[1]

# ╔═╡ a065f8a3-6276-44c0-89c0-34b605fd8639
md"## Creando la red neuronal convolucional"

# ╔═╡ ea0d4d2c-fbdb-4efe-91c3-a7bbce5cc3d3
md"Creemos primero una función para medir la accuracy en nuestro dataset. Tengamos en cuenta que necesitamos un array de 4 dimensiones:"

# ╔═╡ 7a472165-b89e-4797-8fdc-89217eca3947
function accuracy(model, test_imgs, test_labels)
    correct = 0
    for index in 1:length(test_labels)
        probs = model(Flux.cat(test_imgs[:,:,1,index],dims=4))
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
model_v1 = 
	 Chain(
		 Conv((5,5),1 => 6, relu, stride=1, pad=0),
         MaxPool((2,2)),
         Conv((5,5),6 => 16, relu, stride=1, pad=0),
         MaxPool((2,2)),
         Flux.flatten,
         Dense(256=>15,relu),
         Dense(15=>10, sigmoid),
         softmax
    )

# ╔═╡ f2fd40b3-9314-447a-a981-c451b35645ba
md"¡Tataannn! Le agregamos capas de convolución en forma previa a las capas _densas_ y la cantidad de parámetros ahora es menos de la mitad. ¿Qué significan la sintaxis de la capa de _convolución_ y la de _pooling_?:

```
Conv((5,5),1 => 6, relu, stride=1, pad=0)
```

Con la sentencia anterior estamos creando una capa de convolución, en la cual se van a entrenar filtros de tamaño $5x5$. Se recibe una imagen de entrada y se convierte en 6 imágenes tras la aplicación de los filtros (6 filtros). Luego de aplicar la convolución estándar, se aplica una función de activación _ReLU_ a cada uno de los nuevos datos. El _stride_ es de $1$ y no utilizamos _padding_ (_pad=0_).

```
MaxPool((2,2))
```

Con la sentencia anterior se aplica un _pooling_ con una función _máximo_, utilizando bloques de tamaño $2x2$, con stride igual a $2$ (por default, igual al tamaño de bloque) y sin padding.

"

# ╔═╡ e3daadd8-b7da-4be5-9277-32f709516152
md"Probemos predecir uno de los valroes del conjunto de test:"

# ╔═╡ e9d376ff-c19d-4d9b-b39d-acf1ef0a8533
predict = model_v1(Flux.cat(test_imgs[:,:,1,1], dims=4))

# ╔═╡ 4f4a1f0a-18d4-4b54-be4f-f2b5cb7cfbc0
accuracy(model_v1, test_imgs, test_labels)

# ╔═╡ e2c37aef-ac19-4cc7-b755-07b49ad5e7a3
md"Obviamente, predice mal porque no está entrenada. Vamos a entrenarla:"

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

# ╔═╡ 987aa172-f519-4afd-938e-6815a826001c
md"¿Notaron como al tener menos parámetros se entrenó mas rápido? Calculemos la _accuracy_ ahora:"

# ╔═╡ 750f7524-fd9c-4538-91cb-b77eff010a24
accuracy(model_v1, test_imgs, test_labels)

# ╔═╡ 88f22d6e-7a42-464a-a384-546b85fed49c
md"Guardemos ahora nuestra red neuronal un archivo con formato BSON:"

# ╔═╡ d7649a15-f257-4974-b8e3-39fa41029da2
BSON.@save "trained_models/red_neuronal_flux_conv.bson" model_v1

# ╔═╡ db6b30e4-c71f-4e95-8d05-2182c2370245
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
# ╟─a065f8a3-6276-44c0-89c0-34b605fd8639
# ╟─ea0d4d2c-fbdb-4efe-91c3-a7bbce5cc3d3
# ╠═7a472165-b89e-4797-8fdc-89217eca3947
# ╟─67ffa1bf-91d4-448f-a9aa-3ed422ceb6a5
# ╠═cb97693a-28f0-4899-8b47-c19e442142bb
# ╟─f2fd40b3-9314-447a-a981-c451b35645ba
# ╟─e3daadd8-b7da-4be5-9277-32f709516152
# ╠═e9d376ff-c19d-4d9b-b39d-acf1ef0a8533
# ╠═4f4a1f0a-18d4-4b54-be4f-f2b5cb7cfbc0
# ╟─e2c37aef-ac19-4cc7-b755-07b49ad5e7a3
# ╠═53433272-0622-4bfd-8bea-ef14607c2e58
# ╟─987aa172-f519-4afd-938e-6815a826001c
# ╠═750f7524-fd9c-4538-91cb-b77eff010a24
# ╟─88f22d6e-7a42-464a-a384-546b85fed49c
# ╠═d7649a15-f257-4974-b8e3-39fa41029da2
# ╟─db6b30e4-c71f-4e95-8d05-2182c2370245
