### A Pluto.jl notebook ###
# v0.20.9

using Markdown
using InteractiveUtils

# ╔═╡ d2aa248c-8299-11f0-192f-356e92d3eba4
begin
	using Pkg
	Pkg.activate()

	using PlutoUI
	TableOfContents(title="Contenido")
end

# ╔═╡ b755685f-c924-4290-8f44-ffdacb6f1754
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

# ╔═╡ e451db23-aa93-4c48-a04e-402921997371
md"# Introducción a la biblioteca MLJ"

# ╔═╡ e7768c40-6742-43d5-a035-af6e8f9f39bd
md"En este notebook veremos un ejemplo punta a punta usando la biblioteca MLJ para entrenar un modelo partiendo de una muestra de datos almacenada en un archivo XLSX. El objetivo del modelo será el de predecir fallas en equipos y estará basado en nuestro caso de estudio. Al final del notebook, un poco de teoría sobre como medir errores en problemas de [_clasificación binaria_](https://es.wikipedia.org/wiki/Clasificaci%C3%B3n_binaria), como el aquí tratado.

Los paquetes usados en este notebook son:

* PlutoUI
* DataFrames
* MLJ
* MLJLinearModels
* MLJDecisionTreeInterface
* NearestNeighborModels
* ScientificTypes
* CSV
* XLSX
* Statistics
* Distributions
* Plots

"

# ╔═╡ 7a1ecbf6-b9e0-4297-871a-9082848caf46
md"## Setup"

# ╔═╡ 5dc41d8c-e02d-48fe-9f50-d949244b549c
md"El setup inicial consta de dos partes. La primera, es el setup del notebook para tener la funcionalidad usual:"

# ╔═╡ 661c6ebe-4ab7-463f-97cc-af8b1fa0aa36
md"El segundo, es la importación de todos los paquetes necesarios para realizar el ajuste y las predicciones. Si la siguiente celda arroja un error, lo mas probable es que algunos de los paquetes no los tengan instalados:"

# ╔═╡ 74e7b7d7-c086-4bbb-8b80-f0270a9cb01e
md"## Lectura del archivo y preparación de los datos"

# ╔═╡ 00abf82f-6818-46a6-b5b1-4b32978d6514
md"Los datos que vamos a utilizar en el entrenamiento están en un archivo _XLSX_. Para poder trabajar con ellos, debemos cargarlos en memoria. Una estructura cómoda para manejar datos tabulares (como los que tenemos en el _XLSX_) es el _DataFrame_. A continuación, leemos el archivo. De ser necesario, cambiá la ruta:"

# ╔═╡ 374b249b-38a6-4184-af43-03907ed38131
begin
	ruta_y_nombre_del_archivo = "../Caso de estudio/datasets/Dataset_Mantenimiento_Predictivo_electronicaAvanzada.xlsx"

	nombre_de_la_hoja = "Sheet1"
	
	df = DataFrame(XLSX.readtable(ruta_y_nombre_del_archivo, nombre_de_la_hoja))
end

# ╔═╡ c841abf0-94ed-4842-a9e7-0b7b20518797
md"Si la celda de arriba tiró algún error, lo mas probable es que esté mal sea la ruta al archivo o el nombre de la hoja. Si no tira error, deberías ver la tabla encima de la celda. A los datos los guardamos en una estructura en memoria del tipo _DataFrame_, bajo el nombre _df_. Cada vez que escribamos _df_, nos vamos a estar refiriendo a esta estructura.

Vamos a tratar de usar estos datos para crear un modelo que nos permita predecir si estamos por tener _falla_ en un equipo o no. O sea, nuestra varaible dependiente será la columna _Falla_.

Con los datos cargados en memoria, lo primero que vamos a hacer, para no tener problemas, es borrar todas las filas con datos faltantes, usando para ello la función _dropmissing!_. Observen cómo la función recibe como parámetros a nuestra variable _df_, es decir, la tabla de datos. Dado que la función termina con _!_, los cambios, de haber, se aplican directamente sobre los datos, no se realiza ninguna copia:
"

# ╔═╡ 9233b72a-586c-47d0-877f-cf0dbd479359
dropmissing!(df)

# ╔═╡ 640d85a0-9ed2-4d2c-b771-4a587230e210
md"¿Notaron que no tenemos el _begin_ ni el _end_? Esto es porque ejecutamos una sola sentencia en la celda. Si queremos ejecutar mas de una sentencia, si es obligatorio su uso (o de alguna otra _palabra clave_ para definir bloques de código). Si todo salió bien, arriba de la celda anterior deberían ver nuevamente la tabla. 

Ahora, vamos a chequear los tipos de datos, para ver si es necesario transformarlos a algo más amistoso. Usemos para eso _ScientificTypes.schema_:
"

# ╔═╡ de202332-e1dc-4a66-8be5-f6ad1526d71f
ScientificTypes.schema(df)

# ╔═╡ 7b21fa4a-809d-4c61-b6db-b8130d84db08
md"Lo que vemos arriba de la celda anterior es el _esquema_ de la tabla _df_. Esto es, la lista de columnas de la tabla con indicación del _tipo científico de datos_ (_scitypes_) y el _tipo computacional_ (_types_). Para poder trabajar con los datos, necesitamos el tipo correcto. En los caso que vamos a trabajar nosotros, esto es:

* scitype _Continuous_ para todos los datos que representen magnitudes reales (o sea, cuya naturaleza sea ser números).
* scitype _Multiclass_ para todos los datos que representen categorías en las cuales todas las categorías sean igual de importantes (tipos de cosas, básicamente).
* scitype _OrderedFactor_ para todos los datos que representen categorías en las que el orden sea importante. Típicamente se usa en clasificación dicotómica, cuando el foco está en un valor puntual (verdadero/falso, sano/enfermo, normal/fraude, etc.).

También, para evitar problemas, suele ser conveniente transformar todos los datos que tengan que ser magnitudes reales al _tipo computacional Float64_, y los categóricos a _String_ o a _Int_. Veamos como hacerlo:

"

# ╔═╡ b35a819c-7ae8-4f03-bc6a-60ebfb5461bf
begin
	coerce!(df, :Temperatura => ScientificTypes.Continuous)
	coerce!(df, :Vibracion => ScientificTypes.Continuous)
	coerce!(df, ScientificTypes.Count => ScientificTypes.Continuous)
	coerce!(df, ScientificTypes.Textual => ScientificTypes.Multiclass)
	df.Corriente = Float64.(df.Corriente)
	df.Presion_Aire = Float64.(df.Presion_Aire)
	df.Falla = Int.(df.Falla)
	coerce!(df, :Falla => ScientificTypes.OrderedFactor)
end

# ╔═╡ c4994ff0-8fb1-4566-81dd-61ac316c3bca
md"El bloque anterior hace varias cosas. Primero, se hace uso de la función _coerce!_, la cual cambia el _scitype_ de una o mas columnas. Como termina con _!_, implica que los cambios se hacen en el objeto, directamente. Veamos lo que hacemos:

* _coerce!(df, :Temperatura => ScientificTypes.Continuous)_: cambiamos el _scitype_ de la columna _Temperatura_ a _Continuous_.
* _coerce!(df, :Vibracion => ScientificTypes.Continuous)_: ídem al anterior, pero con la columna _Vibración_
* _coerce!(df, ScientificTypes.Count => ScientificTypes.Continuous)_: acá ya nos cansamos, así que convertimos _todas_ las columnas que tienen el _scitype Count_ (que sirve para contar cantidades discretas) a columnas del _scitype Continuous_.
* _coerce!(df, ScientificTypes.Textual => ScientificTypes.Continuous)_: ídem a la anterior, pero queremos cambiar acá todas las columnas con el _scitype Textual_ (que sirve para textos y anotaciones) a _scitype Multiclass_.
* _df.Corriente = Float64.(df.Corriente)_: cambio el _type_ de la columna _Corriente_ a _Float64_.
* _df.Presion_Aire = Float64.(df.Presion_Aire)_: ídem a la anterior,
* _df.Falla = Int.(df.Falla)_: cambio el _type_ de la columna _Falla_ a _Int_.
* _coerce!(df, :Falla => ScientificTypes.OrderedFactor)_: cambio el _scitype_ de la columna _Falla_ a _OrderedFactor_

Volvamos a chequear el _esquema_ y veamos como quedó:
"

# ╔═╡ 9cb3d01c-faff-46a8-b9d7-57964171bd7b
ScientificTypes.schema(df)

# ╔═╡ 8f1d71b8-7963-4787-9144-4a6e288a47bf
md"A priori, todas las columnas que queramos utilizar en nuestros modelos deberían tener algunos de los tipos mencionados anteriormente."

# ╔═╡ a2bba9d3-be9e-434f-a1f0-bd7e25249ec6
md"## Armando conjuntos de entrenamiento y testeo"

# ╔═╡ 9a07afff-f555-45f5-9cfd-a73bf874cea5
md"La separación entre entrenamiento y test vamos a realizarla en dos partes. En la primera, vamos a guardar los datos asociados a la variable a predecir dentro de la variable _target_, y los datos asociados a nuestros predictores dentro de la variable _features_. O sea, _target_ contiene nuestros _ys_, _features_ nuestras _Xs_. Luego de eso, vamos a separar a _target_ y a _features_ en dos conjuntos cada uno, uno de _entrenamiento_ (denominado _train_) y otro de _testeo_ (denominado _test_).


Para hacer copias de nuestros datos, generando las variables _target_ y _features_, utilizamos la función _unpack_. Como no termina en _!_, esto nos dice que nuestra variable _df_ va a permanecer intacta. _unpack_ tiene la siguiente estructura:


> _target, feature, resto = unpack(dataframe, condición para ser target, condición para ser feature, mezclar_filas; rng=semilla aleatoria)_

Esto es, toma a _dataframe_, chequea qué columnas cumplen la _condición para ser target_ y copia esos valores a una nueva variable llamada _target_. Luego, chequea qué columnas cumplen la _condición para ser feature_ y copia esos valores a una nueva variable denominada _feature_. Las columnas no usadas, lo que sobre, se guarda dentro de _resto_. Por último, si mezclar_filas vale true o declaramos _rng=semilla aleatoria_, le estamos diciendo que mezcle las filas de ambas variables de la misma forma (si cambia la fila 2 por la 15, por ejemplo, lo hace en las dos variables en simultáneo). Si no queremos mezclar las filas, ponemos _shuffle=false_ y no declaramos _rng_. Tengan encuenta que _target_ va a ser un vector columna en nuestro caso, y _features_ una matriz:

"

# ╔═╡ 8f9df81f-1f09-450a-a2bb-d2b8cf76366f
target, features, resto = unpack(
	df,
	==(:Falla),
	∈([:Temperatura, :Vibracion, :Corriente, :Equipo, :Ultimo_Mantenimiento_dias, :Turno, :Horas_Operacion]),
	shuffle=true;
	rng=123
)

# ╔═╡ c74b3d81-41dc-49b9-ab78-14902b537e65
md"¿Que hicimos? Bueno, primero, como es una sola sentencia (aunque separada en varios renglones por claridad) no necesitamos utilizar el _begin-end_. Aclarado esto, tomamos el _dataframe_ de nombre _df_ (es decir, nuestros datos) y a la columna llamada _Falla_ la guardamos dentro de target. Después, columna por columna (a excepción de _Falla_, obviamente) chequeamos si su nombre es alguno de los que pusimos en siguiente vector:
> [:Temperatura, :Vibracion, :Corriente, :Equipo, :Ultimo_Mantenimiento_dias, :Turno, :Horas_Operacion]

A las que sí aparecían, las guardamos en _features_, y al resto en _resto_.

Si se fijan arriba de la celda anterior y hacen clic en el _triangulito_, pueden ver el contenido de cada una de las variables generadas.

Ahora separamos en conjuntos de entrenamiento y testeo. El objetivo es seleccionar al azar el $80\%$ de las filas, y usar los datos en esas filas (tanto de _target_ como de _feature_) como _datos de entrenamiento_, y al $20\%$ restante usarlo como _datos de testeo_. Para ello hacemos uso de la función _partition_, de la siguiente forma:
"

# ╔═╡ f00e7064-8ee1-4103-b8bb-8de59d8260c3
(Xtrain, Xtest), (ytrain, ytest) = partition((features, target), 0.8, stratify=target, multi=true, shuffle=true, rng=123)

# ╔═╡ 0e28b90c-f2f5-4f4c-a258-e8d0070c1062
md"¿Que hicimos? Bueno, le dijimos que separamos _features_ en dos grupos, _Xtrain_ y _Xtest_, y que a _target_ lo separara también en _ytrain_ e _ytest_. El $80\%$ de los datos quedó en los _train_, el restante en los _test_. Muy importante, los $X$ y los $y$ se corresponden fila a fila entre sí. Con _stratify=target_ le estamos diciendo que la separación la realice al azar mediante muestreo estratificado, usando como base la variable _target_ (es decir, va a intentar dejar la misma proporción de cada categoría de la variable _target_ tanto en _train_ como en _test_). _multi=true_ indica que estamos separando _features_ y _target_ al mismo tiempo. Y _shuffle=true_ junto a _rng=123_ regula el _azar_ del muestreo (si no queremos mezclar las filas, ponemos _shuffle=false_ y no declaramos _rng_). Si hacemos clic en el triangulito negro arriba de la celda, vemos el contenido de las cuatro nuevas variables que generamos."

# ╔═╡ 09f78b3f-59cb-4b96-95e3-ba2a3cf0247b
md"## Entrenando (por fin)"

# ╔═╡ c5b97c4c-2491-4cc7-8ccf-2bcbfda3c8ae
md"Ya tenemos los datos en condiciones, vamos a empezar a entrenar nuestro modelo. La biblioteca _[MLJ](https://juliaai.github.io/MLJ.jl/stable/)_ tiene el siguiente esquema de trabajo:

* Las transformaciones y modelos a aplicar a los datos se definen como una variable del tipo _modelo_. Por ejemplo, si queremos ajustar nuestros datos según una regresión logística, usamos esta última como modelo. Un modelo no está vinculado a los datos, es solo la definición formal.
* La vinculación de modelos a datos (de entrenamiento) se hace por medio de _machines_. 
* Para entrenar un modelo, aplico la función _fit!_ sobre la _machine_. Esta función, que termina en _!_ calcula los parámetros y los escribe en la misma variable.
* Para predecir, uso la función _predict_ aplicada a la _machine_ ya entrenada y a muestras de _features_ para las cuales quiero calcular la predicción.

Vamos a ajustar nuestros datos a un clasificador basado en _KNN_, con un $K=5$. El modelo clasificador de _KNN_ se llama _KNNClassifier_. Pero, así, en su forma pura, no nos va a servir, por dos cosas:

* Tenemos datos del tipo de categoría, ¿y cómo calculo la distancia entre dos categorías?
* Los datos numéricos no están normalizados, es decir, tienen distintas escalas, por lo cual los de mayor magnitud van a influir en la distancia mucho más que los de menor magnitud.

Para evitar estos dos problemas voy a construirme un _pipeline_ a modo de _modelo_. Un _pipeline_ (tubería) es una secuencia de operaciones que se realizan una atrás de otras (claro, si fuera de otra forma no sería una secuencia, ¿no?). Entran los datos, salen los datos transformados. En nuestro caso, vamos a crear un modelo que nos diga cómo transformar y ajustar los datos. Nuestra _pipeline_ sera:

* Volver las categorías a variables numéricas, mediante la técnica [_One Hot Encoder_](https://interactivechaos.com/es/manual/tutorial-de-machine-learning/one-hot-encoding) usando la función _ContinuousEncoder_.
* Después de eso, ajustar cada columna a una distribución normal con $\mu=0$ y $\sigma=1$, usando la función _Standardizer_.
* Por último, ajustar un modelo de clasificación de KNN con $K=5$, usando la función _KNNClassifier_.

El _pipeline_ lo guardo en la variable _model_, la cual posteriormente vamos a vincular con los datos a través de una _machine_:


"

# ╔═╡ c3956f99-b625-4145-b56b-d940cc71fcac
model = ContinuousEncoder() |> Standardizer() |> KNNClassifier(K=5)

# ╔═╡ f37e447b-ef28-474b-9c2a-99ce315eb2f8
md"Listo, ya tenemos nuestro _modelo_ (nuestra _pipeline_ en realidad). Ahora usemos la función _machine_ para vincularlo con nuestros datos, guardando la vinculación en la variable _mach_:"

# ╔═╡ 6dfd4c56-37e1-472a-b6c0-548bb26c2214
mach = machine(model,Xtrain, ytrain)

# ╔═╡ ff81490f-a2f2-4409-9df2-f3aeb008ea1c
md"Vemos que dice claramente que la _machine_ no está entrenada todavía, así que entrenémosla con _fit!_. Según los datos y el modelo, puede llegar a demorar bastante este proceso. Pero aquí no va a ser el caso, todavía estamos trabajando con cosas simples:"

# ╔═╡ 005e7cc5-20f5-447b-bce9-3fb02249d328
fit!(mach)

# ╔═╡ 8bbdd13f-7c1c-4b74-a4dc-d86b311b09b0
md"Listo, tenemos una _machine_ ya entrenada. Es decir, ajustamos nuestro modelo a los datos. Si queremos chusmear el ajuste, podemos usar *fitted_params*:"

# ╔═╡ 88d68d19-f96b-4065-9a35-35636acbd088
fitted_params(mach)

# ╔═╡ eee6c64b-8128-48d1-b2f8-1633d267e5e0
md"## Prediciendo (para estimar el error de testeo)"

# ╔═╡ 714174e9-1fdd-4a07-a077-5038b783293e
md"Para predecir, uso la función _MLJ.predict_ pasando como parámetros la _machine_ entrandas y las _features_ para las cuales quiero calcular los valores de variable dependiente. Nuestro conjunto de test tiene el $20\%$ de los datos originales, para saber cuantos son podemos usar _size_:"

# ╔═╡ c59c5cb9-514f-4e82-9393-792f1c12a964
size(Xtest)

# ╔═╡ 5d00f253-66de-4b30-ba9a-6e1baa058da0
md"Vemos que _Xtest_ tiene 1000 filas y 7 columnas. Es decir, queremos predecir la posibilidad de falla para 1000 muestras. Usemos _MLJ.predict_:"

# ╔═╡ cb5767a7-465c-4fbf-b16c-e246ae00b3ad
y_preddist = MLJ.predict(mach, Xtest)

# ╔═╡ 6c52be5a-7628-473e-8492-67dca30e2c75
md"Listo, nos guardamos la predicción en una variable llamada *y_preddist*. El clasificador KNN no predice un valor puntual, predice la distribución de probabilidad correspondiente a cada valor de $y$ que queremos calcular. Veamos el resultado de la predicción del primer elemento:"

# ╔═╡ a13662cd-df68-44fd-a4da-5794fac9f356
y_preddist[1]

# ╔═╡ 2503a2b3-a6b2-4775-88a2-6524b8b0ac18
md"La celda anterior nos está diciendo la probabilidad que tiene el primer elemento del conjunto de test de valor $0$ o $1$, es decir, de _no fallar_ o de _fallar_, respectivamente. Y cual fue la frecuencia de $0$ (_no falla_) en nuestro conjunto de testeo:"

# ╔═╡ 55a98aa9-8f3f-409c-8849-03f266a55a2d
histogram(pdf.(y_preddist, 0), title="Frecuencias de probabilidades de ser cero")

# ╔═╡ dab4a18c-135e-4e9e-be68-00d527c656fe
md"Vemos que alrededor de la mitad, o menos, de las predicciones fueron fuertes, con valores cercanos a $1$. Pero hay un número significativo de predicciones con valores entre $0.5$ y $0.9$. Igualmente, solemos querer reportar estimaciones puntuales, así que en este caso, que predecimos categorías, podemos calcular la _moda_:"

# ╔═╡ 1944bc57-5cf6-4320-81f7-aa068ed5e50a
y_pred =  mode.(y_preddist)

# ╔═╡ ef2e3dcb-9122-4ebc-946e-c2a5886758fd
md"Y si inspeccionamos el primer elemento:"

# ╔═╡ 831b9184-aaa6-458e-89e1-3d5e4d64c55d
y_pred[1]

# ╔═╡ c57665d3-4d37-43a4-9866-073ed92bf860
md"Vemos que nos devuelve la predicción puntual.

Tenemos que evaluar qué tan bueno es nuestro modelo. Para ello, vamos a comparar, respecto del conjunto de test, los valores predichos para $y$ contra sus valores reales. Recordemos que nos guardamos parte de nuestros históricos en el conjunto de test, que no fueron utilizados en el entrenamiento.. Por lo cual, son una buena herramienta para evaluar qué tan bien ajustado está el modelo a los datos reales. Calculemos, entonces, algunas métricas:"

# ╔═╡ a811aecf-4f56-4350-8eaa-5781c64c039a
MLJ.accuracy(y_pred, ytest)

# ╔═╡ 784965a0-379e-4fb9-94ef-0c9a12945904
MLJ.recall(y_pred, ytest)

# ╔═╡ ba49563a-02b7-4834-a207-3d6807c1c0cb
MLJ.ppv(y_pred, ytest)

# ╔═╡ 6fb43d57-a791-464c-b391-0bc4005490b1
MLJ.f1score(y_pred, ytest)

# ╔═╡ cd4874f1-51af-4df3-b463-d484d4287f8d
MLJ.confusion_matrix(y_pred, ytest)

# ╔═╡ 810531fd-2f0f-4726-b386-94b86ea5ece5
md"""
!!! info "Actividad para el alumno"
	Vuelvan a la celda donde se crea el _pipeline_ y reemplacen _KNNClassifier(K=5)_ por _LogisticClassifier()_ o _DecisionTreeClassifier()_, para probar entrenar una regresión logística o un árbol. ¿Qué cambia? Chequeen los parámetros, las predicciones y las métricas de error. Prueben cambiar también el valor de $K$ en el _KNNClassifier_.
"""

# ╔═╡ 544b2fa7-3107-48f9-a3ca-b9701f7096d9
md"### Una disgresión sobre las métricas de testeo utilizadas"

# ╔═╡ c43339d5-8bd6-4cb9-a7b0-7748762f23a2
md"Usamos tres métricas a las que no estamos acostumbrados, más la matriz de confusión, para evaluar en el conjunto de test nuestros datos. Estas son herramientas para evaluar el ajuste de modelos en problemas de clasificación.

Empecemos por la matriz de confusión. Esta tiene en sus filas las predicciones sobre el conjunto de test y en sus columnas los valores reales en el conjunto de test. Esta matriz, en cada una de sus celdas, nos aporta los siguientes datos:

* _TP (True Positives)_: positivos bien clasificados, es decir, valores positivos predichos como positivos.
* _TN (True Negatives)_: negativos bien clasificados, es decir, valores negativos predichos como negativos.
* _FP (False Positives)_: negativos clasificados como positivos
* _FN (False Negatives)_: positivos clasificados como negativos

Para cada uno contabilizamos su frecuencia, o bien en términos absolutos como en relativos.

"

# ╔═╡ 2f9d6114-54a4-4941-95d4-ef6da81ea74b
md"""
!!! info "Actividad para el alumno"
	Vuelvan al cálculo de la matriz de confusión. Identifiquen cuando valen los TP, TN, FP y FN.
"""

# ╔═╡ 5808751b-285c-4060-bec4-637f7e411f7f
md"La _Accuracy_ o exactitud mide el porcentaje de aciertos globales. Es útil cuando las clases están balanceadas, pero engañosa con desbalance: si el 99% es negativo, un clasificador que siempre diga “negativo” tendrá 99% de accuracy pero no detecta positivos. Se calcula como:

$Accuracy=\frac{TP+TN}{TP+TN+FP+FN}$

Bajo _MLJ_ la llamamos mediante la función _MLJ.accuracy_.

La _Precision_ (precisión, tanbién conocida como _positive predictive value_) responde la pregunta: de lo que el modelo predijo como positivo, ¿cuánto realmente lo era? Se calcula como:

$Precision = \frac{TP}{TP+FP}$

Y en _MLJ_ mediante la función _MLJ.ppv_. Alta precisión implica pocos falsos positivos. Es clave cuando el costo de un FP es alto (p. ej., alertas de fraude que disparan acciones costosas).

El _Recall_ (sensibilidad) nos indica cuántos positivos reales fueron detectados. Se calcula como:

$Recall=\frac{TP}{TP+FN}$

En _MLJ_ se calcula usando la función _recall_. Un _recall_ alto implica pocos falsos negativos, lo cual es crucial cuando perder un positivo es grave (p. ej., detectar una enfermedad).

_Precision_ y _recall_ suelen competir al mover el umbral de decisión (por ejemplo, la probabilidad mínima para predecir “positivo”):

* Umbral alto: sube _precision_, baja _recall_.
* Umbral bajo: sube _MLJ.recall_, baja _precision_.

Para equilibrarlos, se usa el _F1_:

$F1=\frac{precision \cdot recall}{precision + recall}$

En _MLJ_ se calcula con la función *MLJ.f1score*.

Estás métricas son solo una muestra, las mas comunes, usadas para problemas de _clasificación binaria_. Si quieren ver todas las métricas que ofrece _MLJ_, visiten el siguiente link: [https://juliaai.github.io/StatisticalMeasures.jl/dev/auto_generated_list_of_measures/](https://juliaai.github.io/StatisticalMeasures.jl/dev/auto_generated_list_of_measures/).

"

# ╔═╡ e391a068-ab13-4e45-b6e4-346052d1c908
md"""
!!! tip "Para tener en cuenta"
	Con desbalance, mirá _precision/recall_ y _F1_, y usa métricas por clase. Elegí la métrica según el costo de errores en tu dominio y fijá el umbral acorde.

"""

# ╔═╡ 7337f915-7377-477a-a31a-ab2308f7f1e2
md"
Observen que los supuestos anteriores aplican a cuando tenemos problemas de _clasificación binaria_, que permiten modelar la variable dependiente como un _OrderedFactor_ de solo dos categorías (asumiendo que $1$ es la clase positiva, la que nos importa detectar). En el caso de multiclase, se promedian _precision/recall_: _macro_ (promedio por clase), _micro_ (agrega TP/FP/FN globales), o _weighted_ (pesa por soporte)."

# ╔═╡ Cell order:
# ╟─e451db23-aa93-4c48-a04e-402921997371
# ╟─e7768c40-6742-43d5-a035-af6e8f9f39bd
# ╟─7a1ecbf6-b9e0-4297-871a-9082848caf46
# ╟─5dc41d8c-e02d-48fe-9f50-d949244b549c
# ╠═d2aa248c-8299-11f0-192f-356e92d3eba4
# ╟─661c6ebe-4ab7-463f-97cc-af8b1fa0aa36
# ╠═b755685f-c924-4290-8f44-ffdacb6f1754
# ╟─74e7b7d7-c086-4bbb-8b80-f0270a9cb01e
# ╟─00abf82f-6818-46a6-b5b1-4b32978d6514
# ╠═374b249b-38a6-4184-af43-03907ed38131
# ╟─c841abf0-94ed-4842-a9e7-0b7b20518797
# ╠═9233b72a-586c-47d0-877f-cf0dbd479359
# ╟─640d85a0-9ed2-4d2c-b771-4a587230e210
# ╠═de202332-e1dc-4a66-8be5-f6ad1526d71f
# ╟─7b21fa4a-809d-4c61-b6db-b8130d84db08
# ╠═b35a819c-7ae8-4f03-bc6a-60ebfb5461bf
# ╟─c4994ff0-8fb1-4566-81dd-61ac316c3bca
# ╠═9cb3d01c-faff-46a8-b9d7-57964171bd7b
# ╟─8f1d71b8-7963-4787-9144-4a6e288a47bf
# ╟─a2bba9d3-be9e-434f-a1f0-bd7e25249ec6
# ╟─9a07afff-f555-45f5-9cfd-a73bf874cea5
# ╠═8f9df81f-1f09-450a-a2bb-d2b8cf76366f
# ╟─c74b3d81-41dc-49b9-ab78-14902b537e65
# ╠═f00e7064-8ee1-4103-b8bb-8de59d8260c3
# ╟─0e28b90c-f2f5-4f4c-a258-e8d0070c1062
# ╟─09f78b3f-59cb-4b96-95e3-ba2a3cf0247b
# ╟─c5b97c4c-2491-4cc7-8ccf-2bcbfda3c8ae
# ╠═c3956f99-b625-4145-b56b-d940cc71fcac
# ╟─f37e447b-ef28-474b-9c2a-99ce315eb2f8
# ╠═6dfd4c56-37e1-472a-b6c0-548bb26c2214
# ╟─ff81490f-a2f2-4409-9df2-f3aeb008ea1c
# ╠═005e7cc5-20f5-447b-bce9-3fb02249d328
# ╟─8bbdd13f-7c1c-4b74-a4dc-d86b311b09b0
# ╠═88d68d19-f96b-4065-9a35-35636acbd088
# ╟─eee6c64b-8128-48d1-b2f8-1633d267e5e0
# ╟─714174e9-1fdd-4a07-a077-5038b783293e
# ╠═c59c5cb9-514f-4e82-9393-792f1c12a964
# ╟─5d00f253-66de-4b30-ba9a-6e1baa058da0
# ╠═cb5767a7-465c-4fbf-b16c-e246ae00b3ad
# ╟─6c52be5a-7628-473e-8492-67dca30e2c75
# ╠═a13662cd-df68-44fd-a4da-5794fac9f356
# ╟─2503a2b3-a6b2-4775-88a2-6524b8b0ac18
# ╠═55a98aa9-8f3f-409c-8849-03f266a55a2d
# ╟─dab4a18c-135e-4e9e-be68-00d527c656fe
# ╠═1944bc57-5cf6-4320-81f7-aa068ed5e50a
# ╟─ef2e3dcb-9122-4ebc-946e-c2a5886758fd
# ╠═831b9184-aaa6-458e-89e1-3d5e4d64c55d
# ╟─c57665d3-4d37-43a4-9866-073ed92bf860
# ╠═a811aecf-4f56-4350-8eaa-5781c64c039a
# ╠═784965a0-379e-4fb9-94ef-0c9a12945904
# ╠═ba49563a-02b7-4834-a207-3d6807c1c0cb
# ╠═6fb43d57-a791-464c-b391-0bc4005490b1
# ╠═cd4874f1-51af-4df3-b463-d484d4287f8d
# ╟─810531fd-2f0f-4726-b386-94b86ea5ece5
# ╟─544b2fa7-3107-48f9-a3ca-b9701f7096d9
# ╟─c43339d5-8bd6-4cb9-a7b0-7748762f23a2
# ╟─2f9d6114-54a4-4941-95d4-ef6da81ea74b
# ╟─5808751b-285c-4060-bec4-637f7e411f7f
# ╟─e391a068-ab13-4e45-b6e4-346052d1c908
# ╟─7337f915-7377-477a-a31a-ab2308f7f1e2
