### A Pluto.jl notebook ###
# v0.20.18

using Markdown
using InteractiveUtils

# ╔═╡ 689030e6-a2d8-11f0-2ea1-f948893d4237
begin
	using Pkg
	Pkg.activate()

	using Images,Flux,BSON
	using PlutoUI

	TableOfContents(title="Contenido")
end

# ╔═╡ 9241dc35-c577-4211-98dd-a7af95edc107
begin
    # Acepta siempre los términos/licencias sin pedir confirmación
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    # (Opcional) elegí dónde guardar los datasets
    # ENV["DATADEPS_DIR"] = "/ruta/a/mis/datasets"
    using DataDeps
    using MLDatasets
end

# ╔═╡ 5b7af166-5931-48fa-9c8d-1710d7cdb0d8
md"# Red convolucional LeNet"

# ╔═╡ 02ac8df6-b076-4f6e-9e6a-cc906fe62fed
md"En este notebook construiremos una versión de [**LeNet-5**](https://en.wikipedia.org/wiki/LeNet) para predicción del _MNIST_. _LENET-5_ es una arquitectura importante porque fue uno de los primeros modelos en aplicar capas de convolución para el reconocimiento de imágenes, y sentó el precedente del desarrollo posterior del campo.

Los paquetes a utilziar en este notebook son:

* PlutoUI
* Images
* DataDeps
* MLDatasets
* Flux
* BSON
"

# ╔═╡ d7d2d33b-6410-4150-a5ad-df088a2dc4eb
md"## Setup"

# ╔═╡ 0fa4d648-6652-4ec7-a245-57c5834e022c
md"## Carga de datos"

# ╔═╡ 476e0202-745c-4329-97d8-648f813ce6c8
md"Como en los notebooks previos, arranco cargando los datos y reestructurando las matrices para que se vuelvan arrays de 4 dimensiones: ancho, alto, canal, imagen."

# ╔═╡ 2c9297b3-7071-47a3-b9bb-55206d37f27c
begin
	train_imgs, train_labels = MLDatasets.MNIST(split=:train)[:]
	test_imgs,  test_labels  = MLDatasets.MNIST(split=:test)[:]
	nothing
end

# ╔═╡ 7f99e4dc-d977-4577-85d2-daf5304096cc
size(train_imgs)

# ╔═╡ c8779401-a8cb-477b-9d3d-0bce602f9406
Gray.(train_imgs[:,:,1]')

# ╔═╡ 6b2835cf-644c-405f-b014-5359b2fcc77b
begin
	x_train = reshape(train_imgs, 28, 28, 1, :)
	x_test = reshape(test_imgs, 28, 28, 1, :)
	nothing
end

# ╔═╡ 89849e88-4c78-43e3-8e90-04d19620d817
md"## LeNet"

# ╔═╡ 6e12c5ad-0f1b-4309-846f-c66772629afa
md"""
LeNet es el nombre que se dio a una serie de redes neuronales convolucionales (CNN) desarrolladas en AT&T Bell Labs a fines de los 80 y durante los 90. Su versión más conocida, LeNet-5 (1998), demostró que las CNN podían resolver con gran precisión el reconocimiento de dígitos manuscritos y, crucialmente, se usó en sistemas reales de lectura de cheques (línea NCR/AT&T), procesando millones de cheques por mes en producción: un hito que validó a las CNN en la práctica mucho antes del “boom” de 2012.

El trabajo estuvo liderado por [Yann LeCun](http://yann.lecun.com/), con colaboradores como Léon Bottou, Yoshua Bengio y Patrick Haffner, dentro de un programa más amplio de reconocimiento de documentos y “graph transformer networks” (GTN). El paper de referencia es “Gradient-Based Learning Applied to Document Recognition” (Proc. IEEE, 1998), donde se describe LeNet-5 y su integración en un sistema completo de OCR para cheques.

LeNet-5 toma imágenes en escala de grises 32×32 y organiza el procesamiento en 7 capas con pesos (contando convoluciones y fully-connected), usando tanh y submuestreo por promedio (average pooling) con coeficientes aprendibles. Un desglose típico es:

* C1: conv 5×5, 6 mapas → salida 28×28×6
* S2: subsampling 2×2 → 14×14×6
* C3: conv 5×5, 16 mapas con conectividad parcial (no todos los mapas se conectan con todos los anteriores) → 10×10×16
* S4: subsampling 2×2 → 5×5×16
* C5: conv 5×5 que, sobre 5×5×16, actúa como una capa densa de 120 unidades
* F6: fully-connected de 84 unidades
* Salida: 10 unidades; en el diseño original se usaron unidades tipo RBF, aunque las reimplementaciones modernas suelen reemplazarlas por softmax.

"""

# ╔═╡ 81d782b2-9347-44d5-92a3-b09452043395
md"""$(LocalResource("imagenes/LeNet-5_architecture.svg")) _Por Zhang, Aston and Lipton, Zachary C. and Li, Mu and Smola, Alexander J. - [https://github.com/d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en), CC BY-SA 4.0, [https://commons.wikimedia.org/w/index.php?curid=152265656](https://commons.wikimedia.org/w/index.php?curid=152265656)_"""

# ╔═╡ 95e52218-eb77-44e0-b3c3-a0e740611f8a
md"""
El modelo ronda las ~60k variables entrenables y fue entrenado en dígitos manuscritos (USPS/MNIST).

LeNet consolidó el “patrón” de las CNN modernas: bloques conv → pooling → (opcional) más conv → capas densas → salida, y mostró cómo integrarlas en pipelines reales de OCR. Gran parte de la arquitectura y del enfoque de entrenamiento por gradiente que hoy damos por sentado quedó fijado en aquel trabajo.
"""

# ╔═╡ 01e5e9ff-f8c5-4c56-a369-e8435c5005fa
md"## Construyendo nuestra versión de LeNet"

# ╔═╡ f5436811-2957-475e-94a4-69a7f6892785
md"Vamos a usar _Flux_, con las herramientas que tenemos disponibles en la actualidad (funciones _ReLU_, por ejemplo), para construir nuestra versión de _LeNet_. A fin de que se entrene rápido, vamos a entrenar solo en dos _epochs_, ustedes pueden subirla un poco mas. Como siempre, empezamos definiendo el modelo:"

# ╔═╡ bc3ee121-9c3d-4dff-9c3e-206d2c2b4e8d
model_cnn = Chain(
    Conv((5,5),1 => 6, relu),
    MaxPool((2,2)),
    Conv((5,5),6 => 16, relu),
    MaxPool((2,2)),
    Flux.flatten,
    Dense(256=>120,relu),
    Dense(120=>84, relu),
    Dense(84=>10, sigmoid),
    softmax
)


# ╔═╡ e7fefd7d-05b4-40f5-b32c-0d0272e3ab52
md"Armamos una función para medir la _accuracy_ en este dataset:"

# ╔═╡ d1771dff-1565-44ab-a920-fed88169cf66
function accuracy_cnn(model, x_test, y_test)
    correct = 0
    for index in 1:length(y_test)
        probs = model(Flux.unsqueeze(x_test[:,:,:,index],dims=4))
        predicted_digit = argmax(probs)[1]-1
        if predicted_digit == y_test[index]
            correct +=1
        end
    end
    return correct/length(y_test)
end

# ╔═╡ bdeac038-da0e-4823-8f77-95c20baec826
md"Entrenamos, utilizando _ADAM_ y ejecutando solo dos _epochs_:"

# ╔═╡ 07f99dae-cf47-485c-8354-3f94b0b9ddb0
begin
	# Preparamos los datos
	train_data_cnn = Flux.DataLoader((x_train,train_labels), shuffle=true, batchsize=64)

	# Inicializamos ADAM
	optimizer_cnn = Flux.setup(Adam(), model_cnn)
	
	# Definimos la función de pérdida
	function loss_cnn(model_cnn, x, y)
	    return Flux.crossentropy(model_cnn(x),Flux.onehotbatch(y,0:9))
	end
	
	# Entrenamos el modelo. Para que no sea tan costoso, solo en 2 epochs
	println(accuracy_cnn(model_cnn, x_train, train_labels))
	for epoch in 1:2
	    Flux.train!(loss_cnn, model_cnn, train_data_cnn, optimizer_cnn)
	    println(accuracy_cnn(model_cnn, x_train, train_labels))
	end
end

# ╔═╡ a794fd39-1ac9-4201-9b92-f589ef6258a1
md"Y medimos la _accuracy_ en el conjunto de test:"

# ╔═╡ d7b32ddb-a672-4f2f-bfd3-c423ad115d62
accuracy_cnn(model_cnn, x_test, test_labels)

# ╔═╡ 60b5e8a9-0d86-4b7d-8139-ea31c78a43c7
md"Prueben darle pasadas de entrenamiento adicionales al modelo, para ver como mejora la accuracy."

# ╔═╡ d7085577-0f2b-4e55-9c31-ca4e72e397fd
md"Cuando estemos satisfechos con la _accuracy_, lo guardamos:"

# ╔═╡ 8bb74dac-70bf-4c3d-9c42-ed797f65be80
BSON.@save "trained_models/lenet.bson" model_cnn

# ╔═╡ Cell order:
# ╟─5b7af166-5931-48fa-9c8d-1710d7cdb0d8
# ╟─02ac8df6-b076-4f6e-9e6a-cc906fe62fed
# ╟─d7d2d33b-6410-4150-a5ad-df088a2dc4eb
# ╠═689030e6-a2d8-11f0-2ea1-f948893d4237
# ╠═9241dc35-c577-4211-98dd-a7af95edc107
# ╟─0fa4d648-6652-4ec7-a245-57c5834e022c
# ╟─476e0202-745c-4329-97d8-648f813ce6c8
# ╠═2c9297b3-7071-47a3-b9bb-55206d37f27c
# ╠═7f99e4dc-d977-4577-85d2-daf5304096cc
# ╠═c8779401-a8cb-477b-9d3d-0bce602f9406
# ╠═6b2835cf-644c-405f-b014-5359b2fcc77b
# ╟─89849e88-4c78-43e3-8e90-04d19620d817
# ╟─6e12c5ad-0f1b-4309-846f-c66772629afa
# ╟─81d782b2-9347-44d5-92a3-b09452043395
# ╟─95e52218-eb77-44e0-b3c3-a0e740611f8a
# ╟─01e5e9ff-f8c5-4c56-a369-e8435c5005fa
# ╟─f5436811-2957-475e-94a4-69a7f6892785
# ╠═bc3ee121-9c3d-4dff-9c3e-206d2c2b4e8d
# ╟─e7fefd7d-05b4-40f5-b32c-0d0272e3ab52
# ╠═d1771dff-1565-44ab-a920-fed88169cf66
# ╟─bdeac038-da0e-4823-8f77-95c20baec826
# ╠═07f99dae-cf47-485c-8354-3f94b0b9ddb0
# ╟─a794fd39-1ac9-4201-9b92-f589ef6258a1
# ╠═d7b32ddb-a672-4f2f-bfd3-c423ad115d62
# ╟─60b5e8a9-0d86-4b7d-8139-ea31c78a43c7
# ╟─d7085577-0f2b-4e55-9c31-ca4e72e397fd
# ╠═8bb74dac-70bf-4c3d-9c42-ed797f65be80
