### A Pluto.jl notebook ###
# v0.20.18

using Markdown
using InteractiveUtils

# ╔═╡ 137a74aa-a8f9-11f0-2088-bfbb6e35b706
begin
	using Pkg
	Pkg.activate()

	using Images,Flux,BSON,Metalhead, XLSX, DataFrames
	using PlutoUI

	TableOfContents(title="Contenido")
end

# ╔═╡ 81e5011f-4b42-489c-822d-2f6a46f0f07a
md"# Usando modelos pre-entrenados"

# ╔═╡ 93563191-c446-4b2d-a8b8-416f201d78ac
md"En este notebook vamos a utilizar un modelo preentrenado para reconocimiento de imágenes. Debido al costo computacional, sumado al costo de armar el dataset, es común, en reconocimiento de imágenes, utilizar arquitecturas diseñadas por otros, con pesos ajustados si están disponibles.

Los paquetes a utilizar en este notebook son:

* PlutoUI
* Flux
* Metalhead
* XLSX
* DataFrames



"

# ╔═╡ 47ffbc0a-3749-4c38-b727-d097a89717ee
md"## Setup"

# ╔═╡ 00327732-054c-4a53-9b3e-17c442760344
md"## Acerca de Imagenet"

# ╔═╡ 082256e9-5f3f-43f3-bf13-bc8acc1b3de1
md"""
[Imagenet](https://image-net.org/) es una biblioteca jerárquica de imágenes (muy, muy grande) utilizada para entrenamiento de modelos de visión. Desde 2010 hasta 2017, ImageNet organizó el **ImageNet Large Scale Visual Recognition Challenge (ILSVRC)**, un benchmark anual con tareas de clasificación, localización y detección a gran escala. El subconjunto clásico de clasificación usa 1000 clases y aproximadamente 1200000 imágenes de entrenamiento (más 100000 de test).

En 2012, el modelo **[AlexNet](https://en.wikipedia.org/wiki/AlexNet)**, entrenado mediante GPU, redujo drásticamente la tasa de error respecto de modelos previos y, a partir de 2015, las **[Residual Networks (ResNet)](https://es.wikipedia.org/wiki/Red_neuronal_residual)** lograron tasas de acierto por encima del $96\%$).

En el archivo _"datasets/labels_imagenet.xlsx"_ están disponibles las etiquetas de las 1000 categorías utilizadas:

"""

# ╔═╡ 307cc5bc-afd6-41cc-9416-048336e17d85
df_imagenet_labels = DataFrame(XLSX.readtable("datasets/labels_imagenet.xlsx", "Hoja1"))

# ╔═╡ 374510d3-d3c5-4b90-ad1f-ae80bff14dcd
md"Para poder utilizar un modelo preentrenado en este dataset, es necesario usar imágenes con el mismo formato, 224x244 píxeles a tres canales, normalizadas. Para ello, creemos una función de preprocesamiento:"

# ╔═╡ 6f840f37-acc7-4405-80fd-cffc6247b749
function preprocesar_imagen(img)
	# Convertimos la imagen a Float32, con las dimensiones de ancho, alto, canal y cantidad de imágenes (WHCN) y normalizamos los valores.
	
    img_resized = Images.imresize(img, (224, 224)) # Cambio de tamaño
    img_float = Float32.(channelview(img_resized)) # La vuelvo CHW, tipo Float32
    img_whc = permutedims(img_float, (3, 2, 1)) # CHW -> WHC
    img_batch_cpu = Flux.unsqueeze(img_whc, 4) # WHCN
	
    return img_batch_cpu[:,:,1:3,:]
end


# ╔═╡ 972b8ad7-45d4-43ea-915b-f4c5b9a0ae5d
md"## Arquitectura VGG"

# ╔═╡ d7ae991c-8d83-43d5-afa6-1e9119ea648a
md"[VGG](https://en.wikipedia.org/wiki/VGGNet), desarrollada por el [Visual Geometry Group (Oxford)](https://www.robots.ox.ac.uk/~vgg/), es una familia de CNN _muy profundas_ diseñada para escalar la profundidad (16–19 capas con pesos) usando únicamente filtros 3×3 apilados y max-pooling 2×2. Fue una de las arquitecturas que marcó el ILSVRC 2014: el equipo obtuvo 1er puesto en localización y 2do en clasificación. Actualmente superadas por arquitecturas como la _ResNet_, es sin embargo un buen ejemplo de lo que se puede lograr con capas convolucionales. Vamos a trabajar con la versión **VGG16**, cargando un modelo preentrenado gracias al paquete **Metalhead**:
"

# ╔═╡ a3ada25b-24cb-4718-9cd7-7a2e614b01fa
model = VGG(16; pretrain = true);

# ╔═╡ 7e7e10ee-c5c3-4e1a-bb1d-f23bf3e725c3
md"Ok, ¿que cargamos? Vamos a _visualizarlo_:"

# ╔═╡ 7fd70fb2-88ff-433d-b73d-d655e17d0ede
model

# ╔═╡ 22e968d5-d70b-42a7-956a-eca1ed255b03
md"Podemos ver la cantidad de parámetros que se tuvieron que ajustar. Pero bueno, no importa, ya los tenemos ajustados, vamos a hacer algunas predicciones. Para ello, carguemos en memoria un viejo conocido:"

# ╔═╡ d6a1b350-2b67-4a0d-8efd-917f030dde91
img=load("imagenes/disparo-vertical-de-un-leopardo-en-su-habitat-en-un-safari-en-el-delta-del-okavanga-en-botswana.jpg")

# ╔═╡ 9f4da778-189f-4303-b642-4231bc162e54
md"Tenemos a nuestro leopardo, vamos a proprocesarlo y ver si el modelo lo puede clasificar correctamente:"

# ╔═╡ 0099a9b4-c533-4b3a-bccd-2003830faf67
img_data = preprocesar_imagen(img)

# ╔═╡ 7cb91f6c-01e2-4014-b910-467523a8be6a
output = model(img_data)

# ╔═╡ 0f9fff86-4256-41a4-b255-012e5d424fe9
md"OK, hace la predicción pero, si se fijan bien, _VGG_ no tiene capa softmax, entonces no nos predice probabilidades. Vamos a calcularlas:"

# ╔═╡ 1872bd5a-a458-43ec-b2c8-dc4aab4dd351
probabilities = softmax(vec(cpu(output)))

# ╔═╡ a2879262-befd-4786-be59-eb52a71613c0
md"Son 1000 categorías, no podemos revisar todas, veamos el _id_ de la clase con mayor probabilidad:"

# ╔═╡ b40a51fa-e570-4845-8462-3e12f586f91c
top_class_idx = argmax(probabilities)

# ╔═╡ d702465f-fd90-4246-a12a-cf30339dd4d8
md"¿Y que era esa clase?:"

# ╔═╡ 73076d2b-a4d1-4880-8ea6-d95f99570632
filter(r->r["Class ID"] == top_class_idx+1, df_imagenet_labels)

# ╔═╡ 168b3d24-f44a-4ee4-9353-49c499181dfd
md"¡Que buena predicción! Ahora bien, mucho lío recuperar así la clase correspondiente. Encapsulemos todo en una función:"

# ╔═╡ 7e0f31ec-1f02-48ba-ad0b-baed6b12a8b4
function predict_result(labels, output)
	probabilities = softmax(vec(cpu(output)))
	top_class_idx = argmax(probabilities)
	@show top_class_idx
	return filter(r->r["Class ID"] == top_class_idx+1, labels)[:,"Class Name"]
end

# ╔═╡ 35feb4a4-2b6b-4f22-a229-36fa2fb97ce0
predict_result(df_imagenet_labels, output)

# ╔═╡ 132fb873-4d57-4220-a7c3-1e9cf7dc1811
md"Genial, probemos otra cosa. En la carpeta _imagenes_ hay varias imágenes listas para usar:"

# ╔═╡ e0755526-fdac-417a-9a83-42f71f12cc4d
img_2=load("imagenes/mono.jpg")

# ╔═╡ c33284ec-7bc6-44b5-9e66-33f2b8682af1
img_data_2 = preprocesar_imagen(img_2)

# ╔═╡ 5d2f5fb8-92c8-4191-ab9d-2a15bd8b4ddb
output_2 = model(img_data_2)

# ╔═╡ 5dba704b-80bd-4774-b3e6-ba3def2e9da4
predict_result(df_imagenet_labels, output_2)

# ╔═╡ Cell order:
# ╟─81e5011f-4b42-489c-822d-2f6a46f0f07a
# ╟─93563191-c446-4b2d-a8b8-416f201d78ac
# ╟─47ffbc0a-3749-4c38-b727-d097a89717ee
# ╠═137a74aa-a8f9-11f0-2088-bfbb6e35b706
# ╟─00327732-054c-4a53-9b3e-17c442760344
# ╟─082256e9-5f3f-43f3-bf13-bc8acc1b3de1
# ╠═307cc5bc-afd6-41cc-9416-048336e17d85
# ╟─374510d3-d3c5-4b90-ad1f-ae80bff14dcd
# ╠═6f840f37-acc7-4405-80fd-cffc6247b749
# ╟─972b8ad7-45d4-43ea-915b-f4c5b9a0ae5d
# ╟─d7ae991c-8d83-43d5-afa6-1e9119ea648a
# ╠═a3ada25b-24cb-4718-9cd7-7a2e614b01fa
# ╟─7e7e10ee-c5c3-4e1a-bb1d-f23bf3e725c3
# ╠═7fd70fb2-88ff-433d-b73d-d655e17d0ede
# ╟─22e968d5-d70b-42a7-956a-eca1ed255b03
# ╠═d6a1b350-2b67-4a0d-8efd-917f030dde91
# ╟─9f4da778-189f-4303-b642-4231bc162e54
# ╠═0099a9b4-c533-4b3a-bccd-2003830faf67
# ╠═7cb91f6c-01e2-4014-b910-467523a8be6a
# ╟─0f9fff86-4256-41a4-b255-012e5d424fe9
# ╠═1872bd5a-a458-43ec-b2c8-dc4aab4dd351
# ╟─a2879262-befd-4786-be59-eb52a71613c0
# ╠═b40a51fa-e570-4845-8462-3e12f586f91c
# ╟─d702465f-fd90-4246-a12a-cf30339dd4d8
# ╠═73076d2b-a4d1-4880-8ea6-d95f99570632
# ╟─168b3d24-f44a-4ee4-9353-49c499181dfd
# ╠═7e0f31ec-1f02-48ba-ad0b-baed6b12a8b4
# ╠═35feb4a4-2b6b-4f22-a229-36fa2fb97ce0
# ╟─132fb873-4d57-4220-a7c3-1e9cf7dc1811
# ╠═e0755526-fdac-417a-9a83-42f71f12cc4d
# ╠═c33284ec-7bc6-44b5-9e66-33f2b8682af1
# ╠═5d2f5fb8-92c8-4191-ab9d-2a15bd8b4ddb
# ╠═5dba704b-80bd-4774-b3e6-ba3def2e9da4
