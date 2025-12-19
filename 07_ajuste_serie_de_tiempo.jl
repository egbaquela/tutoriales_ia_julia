### A Pluto.jl notebook ###
# v0.20.9

using Markdown
using InteractiveUtils

# ╔═╡ 40399ecd-1ae4-47fa-bddb-6ecab250479a
begin
	using Pkg
	Pkg.activate()

	using PlutoUI
	TableOfContents(title="Contenido")
end

# ╔═╡ e34848cf-a4e4-420e-959c-ba3725e2bc91
begin
	using DataFrames # Tablas de datos
	using MLJ # Biblioteca principal de funciones de ML
	using MLJLinearModels # Modelo de regresiones
	using MLJDecisionTreeInterface # Modelos de árboles de decisión
	using NearestNeighborModels # Modelos de vecinos mas cercanos (KNN)
	using ScientificTypes # Manipular tipos de datos
	using CSV # Leer datos desde un CSV
	using XLSX # Leer datos desde un XLSX
	using Statistics # Para funciones estadísticas
	using Distributions # Para funciones relativas a distribuciones de probabilidad
	using Plots # Para gráficos
end

# ╔═╡ 3b6a7084-8804-11f0-1c3c-6b3d5008c407
md"# Plantilla para ajustar modelos de regresion"

# ╔═╡ 5bb9aba6-15cd-45bf-ba2e-397c8635347d
md"Esta es una plantilla para ajustar un modelo de regresión (predicción de variable continua) en un notebook de Pluto, usando la biblioteca MLJ."

# ╔═╡ 2f4bd978-921f-4aab-9683-1dd224126438
md"## Setup"

# ╔═╡ bb69fca4-b8b9-4412-bffb-453d7e61ef9d
md"## Lectura de archivos"

# ╔═╡ ba0eb9b6-57db-4231-9018-8f7852ba3796
md"Poné el nombre completo del archivo que contiene los datos en la variable *ruta\_y\_nombre\_del\_archivo* (la ruta es mejor definirla en forma relativa a la ubicación de este notebook, pero podés definirla en forma absoluta). Si es un archivo XLSX, cargá el nombre de la hoja en *nombre\_de\_la\_hoja* y hacé clic en el botón de _play_. Si tus datos están en formato CSV, comentá la línea que tiene el *XLSX.readtable* y descomentá la que tiene el *CSV*, y dale clic al botón de play."

# ╔═╡ ee4066fe-c113-41fd-a905-fd4cf6936481
begin
	ruta_y_nombre_del_archivo = "serie_estacionalidad_demo.xlsx"

	nombre_de_la_hoja = "sheet1"

	# Comentar la siguiente línea si el archivo es un CSV
	df = DataFrame(XLSX.readtable(ruta_y_nombre_del_archivo, nombre_de_la_hoja))

	# Descomentar la siguiente línea si el archivo es un CSV
	#df = CSV.read(ruta_y_nombre_del_archivo, DataFrame)
	
end

# ╔═╡ 3a826bde-7695-4c38-b8ee-9ccd5c38ab3a
md"## Mini procesamiento de datos"

# ╔═╡ f592ab13-bfdd-4a23-8003-17cb07f47549
begin
	dropmissing!(df)
	coerce!(df, ScientificTypes.Count => ScientificTypes.Continuous)
	coerce!(df, ScientificTypes.Textual => ScientificTypes.Multiclass)
	ScientificTypes.schema(df)
end

# ╔═╡ 2060be89-0119-4c5a-9f0b-7606e283530c
md"Fijate si quedó alguna columna en _types_ del tipo _Any_ y forzala a que sea _Float64_ o _String_, según el caso, y convertila en caso de ser necesario a _Multiclass_ o _Continuous_. Acá no hace falta, pero queda de ejemplo:"

# ╔═╡ d2229664-a415-472b-b49e-b6f71586b31e
begin
	df.y = Float64.(df.y)

	coerce!(df, :y => ScientificTypes.Continuous)

	ScientificTypes.schema(df)
end

# ╔═╡ 7a6ddacb-df65-465e-90d7-962743622ab0
md"Declará el tipo correcto para la varaible que querés predecir. Como es regresion, debería ser _Continuous_. De nuevo, acá no hace falta, pero lo dejamos como ejemplo:"

# ╔═╡ ce1bde1f-f3bc-43d8-beb4-58823d7ca8bb
begin
	coerce!(df, :Cantidad => ScientificTypes.Continuous)

	ScientificTypes.schema(df)
end

# ╔═╡ 60fb3b57-9f41-49da-a642-86ab6f85caa6
md"## Armado de conjuntos de entrenamiento y testeo"

# ╔═╡ 8617df23-4406-42ff-8318-b7d283446fba
md"Hagamos primero el unpack, sin mezclar filas. Cambiá los nombres de las columnas por los nombres correctos para tu dateset. Por ejemplo, acá voy a querer predecir _Cantidad_ en base a _Año_ y _Producto_:"

# ╔═╡ b6fa6512-4972-433f-81b1-3f6e15db6fbb
target, features, resto = unpack(
	df,
	==(:y),
	∈([:t, :idx]),
	shuffle=false
)

# ╔═╡ 7447e0de-118d-4fd1-8ad6-dc5bf9879c35
md"Ahora separemos en entrenamiento y test. Como **no quiero mezclar las filas** le voy a poner _shuffle=false_ y no voy a declarar el parámetro _rng_. No vamos a estratificar en este ejemplo. Cambiá el 0.7 por el valor la fracción de los datos que queres usar para entrenar:"

# ╔═╡ 2e0cee41-451e-4f5c-af75-0edcd90f297c
(Xtrain, Xtest), (ytrain, ytest) = partition((features, target), 0.7, multi=true, shuffle=false)

# ╔═╡ 189b5a36-b473-44db-a0dc-89a73781210f
md"## Entrenamiento"

# ╔═╡ 8530f0f9-4940-4dcd-9d1f-d962944e85f4
md"Hagamos un entrenamiento con un pipeline básico. Solo tenés que cambiar el tipo de modelo, llegado el caso. Elegí el que queres (descomentalo y comentá el resto) y cambiale los parámetros si la deseas:"

# ╔═╡ ddb7e971-cefb-4a96-8b8c-afb4740616a2
#model =  ContinuousEncoder() |> Standardizer() |> KNNRegressor(K=5)
model = ContinuousEncoder() |> Standardizer() |> DecisionTreeRegressor()
#model = ContinuousEncoder() |> Standardizer() |> LinearRegressor()


# ╔═╡ 99445806-058c-4b8f-a187-0cd0528404d2
mach = machine(model,Xtrain, ytrain)

# ╔═╡ da8263da-2109-417e-8c9c-e401ffcc732c
fit!(mach)

# ╔═╡ 86250c80-1b04-481f-947b-535c608d8d24
md"## Chequeo de parámetros ajustados"

# ╔═╡ a1358f85-32d9-46e1-ba11-58dc6c56c5e1
fitted_params(mach)

# ╔═╡ 6a64d1a6-edd3-4df2-bffb-aa99dd098a67
md"## Medición del error en conjunto de tests"

# ╔═╡ f4a008cc-cade-44a4-9c49-6b5f7dde6715
md"Generá las predicciones:"

# ╔═╡ 4c256409-1209-4b33-99d8-0fcc513fbb3b
y_preddist = MLJ.predict(mach, Xtest)

# ╔═╡ e21ec125-ca5e-47f8-a2a6-1e6a8ca01bc2
y_pred =  mean.(y_preddist)

# ╔═╡ 212e75bc-fbf5-4a68-b002-2c36b1b0a76c
md"Reemplazá las métricas por las que quieras utilizar:"

# ╔═╡ 2d72c241-0c0b-4e2e-abf0-970c96e887a3
MLJ.rmse(y_pred, ytest)

# ╔═╡ 4515e77c-47c8-44e3-9f42-6e3c817af6bf
MLJ.mae(y_pred, ytest)

# ╔═╡ 3bf6615b-12f8-4964-9f3b-cc1a9c02cc21
begin
	plt=plot(1:length(df.y),df.y, label="y")
	vline!([length(df.y)], label="Punto de corte")
	plot!((length(df.y)+1):(length(df.y)+length(ytest)), ytest, lw=2, label="Real (test)")
	plot!((length(df.y)+1):(length(df.y)+length(ytest)), y_pred, lw=2, label="Predicción (test)")
end

# ╔═╡ b1623357-908e-4ed6-8bd2-30c331ddfa67
md"""
!!! info "Actividad para el alumno"
	¿Por que se comporta así?, ¿que estaría prediciendo?
"""

# ╔═╡ 3ab38c4c-5b2d-4b65-a722-3a6fce8f2d98
md"## Agregando _features_ adicionales"

# ╔═╡ 9d187ced-6882-4cc9-838d-9da762054859
md"Ya vimos que, así como está, no podemos predecir bien la evolución de nuestra serie de tiempo. El problema principal es que nuestros modelos son muy simples para capturar toda la dinámica de la serie de tiempo a partir de un único valor (el paso del tiempo). ¿Pero qué hacemos entonces? Bueno, probemos agregar features adicionales. Hagamos para eso una copia de nuestro dataframe y trabajemos sobre esta copia:"

# ╔═╡ d16d5892-4663-425b-a0f5-d33cad5c1978
df2 = deepcopy(df)

# ╔═╡ 7cfffecf-52d6-46ba-8ec0-737e755ef7a9
md"Con nuestra copia, _df2_, agreguemos features adicionales. Fijémosnos en el gráfico. Se observa una recurrencia de picos y valles cada, más o menos, 25 periodos. Agreguemos entonces, como dato para predecir la variable $y$ del periodo $t$, el valor de la variable $y$ en el periodo $t-25$."

# ╔═╡ 1e91de03-23d7-4259-816e-c290aafa53a2
begin
	df2.lag25 .= 0.0 # Primero creo una nueva columna llena de ceros
	df2.lag25[26:nrow(df)] .= df2.y[1:(nrow(df)-25)] # Cargo en cada fila el valor de y pero de 25 periodos atrás
	
	df2 # Muestro el dataframe
end

# ╔═╡ 236c2e13-05aa-41ec-9cf0-80020280ae2c
md"Chequeamos los tipos de datos:"

# ╔═╡ 79af1cac-a79d-457a-9bb7-b29c4476b97e
ScientificTypes.schema(df2)

# ╔═╡ 52d694dc-36cd-45a4-aa69-8a3d09c6616e
md"Separamos en features y target:"

# ╔═╡ 3c989de9-1a37-4404-a7f3-7db2af042d65
target2, features2, resto2 = unpack(
	df2,
	==(:y),
	∈([:t, :lag25]),
	shuffle=false
)

# ╔═╡ 3623aa31-4e0c-4ada-b485-85e1adcb7d8b
md"Separamos entre entrenamiento y test:"

# ╔═╡ 88e10c1b-f4c4-416e-af6e-2d603df8a8e4
(Xtrain2, Xtest2), (ytrain2, ytest2) = partition((features2, target2), 0.7, multi=true, shuffle=false)

# ╔═╡ 11c9d457-018d-4412-ba63-865937ec5cb9
md"Definimos el modelo a utilizar:"

# ╔═╡ 6eabcfc1-a29f-4882-b97d-9c1a516dab22
#model2 =  ContinuousEncoder() |> Standardizer() |> KNNRegressor(K=5)
model2 = ContinuousEncoder() |> Standardizer() |> DecisionTreeRegressor()
#model2 = ContinuousEncoder() |> Standardizer() |> LinearRegressor()

# ╔═╡ 0c2d1371-9d7e-4743-9881-a8df0ab6aa56
md"Vinculamos datos con modelo:"

# ╔═╡ f6642995-9be6-49b9-b22b-cbc20541ecfe
mach2 = machine(model2,Xtrain2, ytrain2)

# ╔═╡ 38dcc8b9-9667-45fc-b1fa-9ab2a274bdd3
md"Entrenamos:"

# ╔═╡ db202368-36de-440d-b717-eb0259f33e2f
fit!(mach2)

# ╔═╡ 9a88bc2b-7212-4734-981a-13d1a62f1924
md"Mostramos los parámetros (si nos interesan):"

# ╔═╡ 298ae140-5bde-4e50-a1aa-758094d1051b
fitted_params(mach2)

# ╔═╡ 187eb015-8537-470e-8b20-e88b700e9f46
md"Predecimos en el conjunto de test:"

# ╔═╡ c3ceb883-46e8-4677-8f25-46b73a5b5c94
begin
	y_preddist2 = MLJ.predict(mach2, Xtest2)
	y_pred2 = mean.(y_preddist2)
end

# ╔═╡ 4042d061-a517-4bd8-a306-e3fd08952ea6
md"Medimos el error en el conjunto de test:"

# ╔═╡ b91eb796-ed0d-4e35-993b-2813fb717aec
MLJ.rmse(y_pred2, ytest2)

# ╔═╡ 8f7ef2ad-c54d-41b8-a30a-d42061eb02b4
MLJ.mae(y_pred2, ytest2)

# ╔═╡ 4bdab339-67e1-4b58-9384-6034668f1db7
md"Y visualizamos el error con un gráfico:"

# ╔═╡ 156571a7-46e2-488c-ac77-31432d2225d6
begin
	plt2=plot(1:length(df2.y),df.y, label="y")
	vline!([length(df2.y)], label="Punto de corte")
	plot!((length(df2.y)+1):(length(df2.y)+length(ytest2)), ytest2, lw=2, label="Real (test)")
	plot!((length(df2.y)+1):(length(df2.y)+length(ytest2)), y_pred2, lw=2, label="Predicción (test)")
end

# ╔═╡ de9f97f3-dbe3-4a05-90f0-0f22f07ded2c
md"""
!!! info "Actividad para el alumno"
	Para el mismo modelo, usando los mismos conjuntos de entrenamiento y test, los resultados mejoran, porque agregamos información extra. Comparen qué pasa con el resto de los modelos.
"""

# ╔═╡ 1124812d-b1e7-4a2d-ab4b-953f13765ecc
md"""
!!! info "Actividad para el alumno"
	Prueben agregar diferentes tipos de lags (junto al lag 25). Por ejemplo, prueben usar como predictores la variable $t$, un lag de tamaño 25 y un lag de tamaño 10. Tiene que agregar la columna al dataframe y, cuando separan entre _features_ y _target_, declarar los lags. ¿Esto mejora los resultados?
"""

# ╔═╡ Cell order:
# ╟─3b6a7084-8804-11f0-1c3c-6b3d5008c407
# ╟─5bb9aba6-15cd-45bf-ba2e-397c8635347d
# ╟─2f4bd978-921f-4aab-9683-1dd224126438
# ╠═40399ecd-1ae4-47fa-bddb-6ecab250479a
# ╠═e34848cf-a4e4-420e-959c-ba3725e2bc91
# ╟─bb69fca4-b8b9-4412-bffb-453d7e61ef9d
# ╟─ba0eb9b6-57db-4231-9018-8f7852ba3796
# ╠═ee4066fe-c113-41fd-a905-fd4cf6936481
# ╟─3a826bde-7695-4c38-b8ee-9ccd5c38ab3a
# ╠═f592ab13-bfdd-4a23-8003-17cb07f47549
# ╟─2060be89-0119-4c5a-9f0b-7606e283530c
# ╠═d2229664-a415-472b-b49e-b6f71586b31e
# ╟─7a6ddacb-df65-465e-90d7-962743622ab0
# ╠═ce1bde1f-f3bc-43d8-beb4-58823d7ca8bb
# ╟─60fb3b57-9f41-49da-a642-86ab6f85caa6
# ╟─8617df23-4406-42ff-8318-b7d283446fba
# ╠═b6fa6512-4972-433f-81b1-3f6e15db6fbb
# ╟─7447e0de-118d-4fd1-8ad6-dc5bf9879c35
# ╠═2e0cee41-451e-4f5c-af75-0edcd90f297c
# ╟─189b5a36-b473-44db-a0dc-89a73781210f
# ╟─8530f0f9-4940-4dcd-9d1f-d962944e85f4
# ╠═ddb7e971-cefb-4a96-8b8c-afb4740616a2
# ╠═99445806-058c-4b8f-a187-0cd0528404d2
# ╠═da8263da-2109-417e-8c9c-e401ffcc732c
# ╟─86250c80-1b04-481f-947b-535c608d8d24
# ╠═a1358f85-32d9-46e1-ba11-58dc6c56c5e1
# ╟─6a64d1a6-edd3-4df2-bffb-aa99dd098a67
# ╟─f4a008cc-cade-44a4-9c49-6b5f7dde6715
# ╠═4c256409-1209-4b33-99d8-0fcc513fbb3b
# ╠═e21ec125-ca5e-47f8-a2a6-1e6a8ca01bc2
# ╟─212e75bc-fbf5-4a68-b002-2c36b1b0a76c
# ╠═2d72c241-0c0b-4e2e-abf0-970c96e887a3
# ╠═4515e77c-47c8-44e3-9f42-6e3c817af6bf
# ╠═3bf6615b-12f8-4964-9f3b-cc1a9c02cc21
# ╟─b1623357-908e-4ed6-8bd2-30c331ddfa67
# ╟─3ab38c4c-5b2d-4b65-a722-3a6fce8f2d98
# ╟─9d187ced-6882-4cc9-838d-9da762054859
# ╠═d16d5892-4663-425b-a0f5-d33cad5c1978
# ╟─7cfffecf-52d6-46ba-8ec0-737e755ef7a9
# ╠═1e91de03-23d7-4259-816e-c290aafa53a2
# ╟─236c2e13-05aa-41ec-9cf0-80020280ae2c
# ╠═79af1cac-a79d-457a-9bb7-b29c4476b97e
# ╟─52d694dc-36cd-45a4-aa69-8a3d09c6616e
# ╠═3c989de9-1a37-4404-a7f3-7db2af042d65
# ╟─3623aa31-4e0c-4ada-b485-85e1adcb7d8b
# ╠═88e10c1b-f4c4-416e-af6e-2d603df8a8e4
# ╟─11c9d457-018d-4412-ba63-865937ec5cb9
# ╠═6eabcfc1-a29f-4882-b97d-9c1a516dab22
# ╟─0c2d1371-9d7e-4743-9881-a8df0ab6aa56
# ╠═f6642995-9be6-49b9-b22b-cbc20541ecfe
# ╟─38dcc8b9-9667-45fc-b1fa-9ab2a274bdd3
# ╠═db202368-36de-440d-b717-eb0259f33e2f
# ╟─9a88bc2b-7212-4734-981a-13d1a62f1924
# ╠═298ae140-5bde-4e50-a1aa-758094d1051b
# ╟─187eb015-8537-470e-8b20-e88b700e9f46
# ╠═c3ceb883-46e8-4677-8f25-46b73a5b5c94
# ╟─4042d061-a517-4bd8-a306-e3fd08952ea6
# ╠═b91eb796-ed0d-4e35-993b-2813fb717aec
# ╠═8f7ef2ad-c54d-41b8-a30a-d42061eb02b4
# ╟─4bdab339-67e1-4b58-9384-6034668f1db7
# ╠═156571a7-46e2-488c-ac77-31432d2225d6
# ╟─de9f97f3-dbe3-4a05-90f0-0f22f07ded2c
# ╟─1124812d-b1e7-4a2d-ab4b-953f13765ecc
