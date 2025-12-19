### A Pluto.jl notebook ###
# v0.20.18

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

# ╔═╡ b002c1b8-a2c0-11f0-347c-e98e7815a9f6
begin
	using Pkg
	Pkg.activate()

	using Images
	using PlutoUI
    using CairoMakie
	using StatsBase

	TableOfContents(title="Contenido")
end


# ╔═╡ d8bac896-27db-4130-9e65-cc365b5123dd
md"# Convolución para redes neuronales"

# ╔═╡ 2e6478a5-9c5f-4e43-aedf-63b099dd07d2
md"En este notebooks veremos dos técnicas que nos permiten mejorar la eficiencia en el procesamiento de imágenes en redes neuronales.

Los paquetes a utilizar en este notebook son:

* Images
* PlutoUI
* CairoMakie
* StatsBase
"

# ╔═╡ 76894c82-42d4-488f-ad0b-73a8e75e7a47
md"## Setup"

# ╔═╡ 43b69b09-ee24-4a47-91ae-5c6ca70ba8d5
md"## Acerca de imágenes y matrices"

# ╔═╡ 76a1d3c4-7eda-4e39-9683-6a3e105adec6
md"Recordemos que a las imágenes las vamos a representar como [_mapas de bits_](https://es.wikipedia.org/wiki/Imagen_de_mapa_de_bits), es decir como matrices numéricas. Una sola matriz si trabajamos en [_escalas de grises_](https://es.wikipedia.org/wiki/Escala_de_grises), tres si trabajamos en colores ([_RGB_](https://es.wikipedia.org/wiki/RGB), por ejemplo) y cuatro si usamos transparencias ([_RGBA_](https://es.wikipedia.org/wiki/Espacio_de_color_RGBA), por ejemplo). Cada una de estas matrices son los [_canales de una imagen_](https://es.wikipedia.org/wiki/Canal_(imagen_digital)). O sea, una imagen en escala de grises tiene un solo canal, una imagen en RGB tiene 3 canales y una imagen en RGBA tiene 4 canales."

# ╔═╡ 424dce09-39fe-4fc9-a88b-1932894780f6
md"## Convoluciones"

# ╔═╡ 4fd1bbae-0aaa-421e-899d-787ec9684d36
md"En visión por computadora, una convolución es una operación que toma una imagen (estrictamente, un _canal de una imagen_) y le _pasa por encima_ un pequeño _filtro_ (también llamado **kernel**) para producir un nuevo mapa de características (**feature map**). Ese filtro es una matriz pequeña (por ejemplo, 3×3 o 5×5). Al deslizar el filtro por toda la imagen, en cada posición se hace una suma ponderada de los píxeles vecinos, generando de esta forma el _feature map_. Si cuando pasamos el filtro la suma ponderada es mayor que cierto umbral, el patrón que buscábamos filtrar está presente en esa zona de la imagen. En el contexto de redes neuronales, los valores de los elementos de la matriz filtro son pesos que la red aprende.



"

# ╔═╡ 8451bb76-a967-4dd8-bc2f-bcf9b7d0ae96
md"""$(LocalResource("imagenes/2D_Convolution_Animation.gif")) _Por Michael Plotke - Trabajo propio, CC BY-SA 3.0, [https://commons.wikimedia.org/w/index.php?curid=24288958](https://commons.wikimedia.org/w/index.php?curid=24288958)_"""

# ╔═╡ c0a08cb6-fbd0-4046-b6cf-8495132a17c5
md"Cuando tratamos el dataset _MNIST_ con métodos clásicos de aprendizaje automático, una de las primeras cosas que hacíamos era aplanar la imagen (la matriz) a un vector. En esa operación se pierde información crítica de la imagen en sí: su distribución espacial. Efectuar una o más operaciones de convolución antes de aplanar una imagen nos permite obtener información de dicha distribución espacial, ya que el _output_ de una convolución, su _feature map_, es una nueva imagen en la que cada pixel _promedia_ (usando una especie de promedio ponderado) los píxeles cercanos de la imagen original.

La convolución nos permite:

* _Detectar patrones locales_: bordes, texturas, esquinas, líneas, etc.
* _Compartir pesos_: el mismo filtro se aplica en todas las posiciones, reduciendo drásticamente la cantidad de parámetros frente a una capa densa.
* _Mantener equivarianza traslacional_: si el objeto se desplaza en la imagen, la activación del filtro se desplaza igual; con _pooling_ o _stride_ ganamos cierta invariancia (el “qué” importa más que el “dónde exacto”).

¿Y cómo aplicamos la convolución?

1. Elegimos un kernel (p. ej., 3×3).
2. Lo colocamos sobre una ventana de la imagen del mismo tamaño.
3. Multiplicamos elemento a elemento y sumamos (más un sesgo).
4. Desplazamos el kernel a la derecha/abajo y repetimos.
5. (Opcional) Aplicamos una función de activación (_ReLU_, etc.).

Veamos el siguiente ejemplo. Dada la matriz:

$input=\begin{pmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9
\end{pmatrix}$

Y el kernel:

$kernel=\begin{pmatrix}
1 & 0 \\
0 & -1
\end{pmatrix}$

La convolución del _kernel_ sobre el _input_ (aplicada sin _padding_ y con _stride_ igual a 1) daría lugar a una matriz de $2x2$ que se calcularía como:

* Arriba-izq: $1 \cdot 1+2 \cdot 0+4 \cdot 0+5 \cdot (−1)=−4$
* Arriba-der: $2 \cdot 1+3 \cdot 0+5 \cdot 0+6 \cdot (−1)=−4$
* Abajo-izq: $4 \cdot 1+5 \cdot 0+7 \cdot 0+8 \cdot (−1)=−4$
* Abajo-der: $5 \cdot 1+6 \cdot 0+8 \cdot 0+9 \cdot (−1)=−4$

$output=\begin{pmatrix}
-4 & -4 \\
-4 & -4
\end{pmatrix}$

Fíjense que la matriz _output_ es más chica que el _input_. Para que esto no pase y se mantenga al tamaño se suele agregar un **padding**, es decir, bordes de _ceros_. Para nuestro kernel de $2x2$, un  _padding adecuado sería_:

$input=\begin{pmatrix}
0 & 0 & 0 & 0\\
0 & 1 & 2 & 3\\
0 & 4 & 5 & 6\\
0 & 7 & 8 & 9
\end{pmatrix}$

Por otro lado, fijensé que movemos el kernel de a una celda a la vez. Eso es el **stride** de 1. Podríamos movernos de a dos celdas (_stride 2_) o, en general, un _stride n_.

"

# ╔═╡ d29f4df6-1806-4469-98f0-3e9e536df47e
md"### Un ejemplo interactivo (sin padding)"

# ╔═╡ 086b5efb-fa17-4927-9578-51b0d570e64c
md"Probemos usar distintos kerneles sobre una imagen con una gradiente de color"

# ╔═╡ 0037258a-e2dc-4aba-aa58-8328b1948f3a
md"""
|Parámetro|Valor|
|---------|-----|
|Patrón del input|$(@bind pattern Select(["Gradiente", "Cuadriculado", "Aleatorio"]; default="Gradiente"))|
|Kernel|$(@bind kernel_name Select(["Sobel X","Sobel Y","Borde (Laplace)","Desenfoque (box)","Sharpen","Identidad"]; default="Sobel X"))|
|Stride|$(@bind stride PlutoUI.Slider(1:3; default=2, show_value=true))|

"""

# ╔═╡ 79afdf25-ddd4-4978-9fd2-8bbef9d39aa3
begin
	# Imagen de demo (H×W)
	function demo_image(H::Int, W::Int, pattern::String)
	    if pattern == "Gradiente"
	        A = [ (x + y) for y in 1:H, x in 1:W ]
	    elseif pattern == "Cuadriculado"
	        A = [ ((x ÷ 2 + y ÷ 2) % 2) for y in 1:H, x in 1:W ]
	    else
	        A = rand(H, W)
	    end
	    A = Float32.(A)
	    A .-= minimum(A)
	    M = maximum(A)
	    M > 0 && (A ./= M)
	    return A
	end
	
	# Catálogo de kernels 3×3 (básicos para docencia)
	function preset_kernel(name::String)
	    if name == "Sobel X"
	        Float32.([-1 0 1; -2 0 2; -1 0 1])
	    elseif name == "Sobel Y"
	        Float32.([-1 -2 -1; 0 0 0; 1 2 1])
	    elseif name == "Borde (Laplace)"
	        Float32.([0 -1 0; -1 4 -1; 0 -1 0])
	    elseif name == "Desenfoque (box)"
	        fill(Float32(1/9), 3, 3)
	    elseif name == "Sharpen"
	        Float32.([0 -1 0; -1 5 -1; 0 -1 0])
	    elseif name == "Identidad"
	        Float32.([0 0 0; 0 1 0; 0 0 0])
	    else
	        error("Kernel no reconocido")
	    end
	end
	
	# Convolución 2D "valid" con stride (sin padding)
	function conv_valid(I::Matrix{T}, K::Matrix{T}; stride::Int=1) where {T<:Real}
	    H, W = size(I)
	    kh, kw = size(K)
	    oh = 1 + (H - kh) ÷ stride
	    ow = 1 + (W - kw) ÷ stride
	    O = zeros(Float32, oh, ow)
	    for r in 1:oh, c in 1:ow
	        i = 1 + (r-1)*stride
	        j = 1 + (c-1)*stride
	        @inbounds O[r, c] = sum(@view(I[i:i+kh-1, j:j+kw-1]) .* K)
	    end
	    return O
	end
	H=11
	W=11
	kscale = 1.0

	I = Float64.(demo_image(H, W, pattern))
	K = Float64.(kscale .* preset_kernel(kernel_name))
	kh, kw = size(K)
	
	# Validación mínima para que exista al menos 1 posición:
	@assert H >= kh && W >= kw "La imagen debe ser al menos tan grande como el kernel."
	
	O = conv_valid(I, K; stride=stride)
	oh, ow = size(O)
	total_steps = max(1, oh*ow)

	md"<-- Cálculos internos -->"
end

# ╔═╡ 73b64fa8-bfb9-4c2f-8e66-709ca7b52c05
md"""
Paso de la convolución: $(@bind step PlutoUI.Slider(1:total_steps; default=1, show_value=true))
"""

# ╔═╡ 572e185b-6975-4619-b15f-a92f5d766dd1
begin
	# (r,c) en el feature map
	r = Int(fld(step-1, ow)) + 1
	c = Int(mod(step-1, ow)) + 1
	
	# Top-left (i,j) en la imagen original para la ventana actual
	i = 1 + (r-1)*stride
	j = 1 + (c-1)*stride
	
	patch = @view I[i:i+kh-1, j:j+kw-1]
	valor = sum(patch .* K)
	salida_prevista = O[r, c]

	md"Salida del paso actual: $(salida_prevista)"
end

# ╔═╡ 545ede3e-03fa-4661-b5a2-5bd0be95b6ef
begin
	fig = Figure(resolution = (1050, 360))
	
	# Imagen
	ax1 = CairoMakie.Axis(fig[1,1], title = "Imagen (entrada)", yreversed = true)
	heatmap!(ax1, 1:W, 1:H, I; interpolate=false, colormap=:viridis)
	xlims!(ax1, 0.5, W+0.5); ylims!(ax1, 0.5, H+0.5)
	poly!(ax1, Rect(j-0.5, i-0.5, kw, kh); color = (:transparent), strokecolor = :red, strokewidth = 2)
	
	# Kernel
	ax2 = CairoMakie.Axis(fig[1,2], title = "Kernel (filtro)", yreversed = true)
	heatmap!(ax2, 1:kw, 1:kh, K; interpolate=false, colormap=:viridis)
	xlims!(ax2, 0.5, kw+0.5); ylims!(ax2, 0.5, kh+0.5)
	
	# Feature map
	ax3 = CairoMakie.Axis(fig[1,3], title = "Feature map (salida)", yreversed = true)
	heatmap!(ax3, 1:ow, 1:oh, O; interpolate=false, colormap=:viridis)
	xlims!(ax3, 0.5, ow+0.5); ylims!(ax3, 0.5, oh+0.5)
	poly!(ax3, Rect(c-0.5, r-0.5, 1, 1); color = (:transparent), strokecolor = :orange, strokewidth = 2)
	
	fig
end

# ╔═╡ 01fe8cea-ffea-4e2a-b727-d1bdebe11a93
md"### Creando convoluciones _a mano_"

# ╔═╡ 325a2f62-09c5-4cb1-bbc2-e3176df2e008
md"Vamos a implementar una convolución básica _a mano_. Para ello nos vamos a basar en [este tutorial](https://medium.com/data-science/understanding-convolution-by-implementing-in-julia-3ed744e2e933). 

Empecemos creando una matriz _input_ y una matriz _filter_ (el kernel). Noten como en Julia las matrices se definen _por columnas_:"

# ╔═╡ fc2775fb-1401-4074-950c-6b48263761dd
input = [[1,2,3,4,5,6] [7,8,9,10,11,12] [2,4,6,8,10,12] [1,3,5,7,9,11] [1,1,1,2,2,2] [1,0,1,0,1,0]]

# ╔═╡ 6a145ce8-bb01-4c52-8004-4498555f43fd
filter = [[1,1,1] [0,0,0] [0.1,0.1,0.1]]

# ╔═╡ 9083eb03-c220-4ea4-99ed-e1d26909d007
md"Creamos ahora una función para aplicar convolución, sin padding y con stride igual a 1, para mantenerla simple:"

# ╔═╡ eaf1bdac-a5a7-48b8-ad4e-ec2aaf23dcd7
# Función para calcular la convolución, sin padding y con stride igual a 1. Toma como input la matriz a procesar (input) y el kernel (filter). Asume que el kernel es de 2x2 o mayor
function convolucion(input, filter)
    input_r, input_c = size(input) # Extrae la cantidad de filas y columnas del input
    filter_r, filter_c = size(filter) # Extrae la cantidad de filas y columnas del kernel

	# Me aseguro que el kernel sea cuadrado
    if filter_r != filter_c
        throw(DomainError(filter, "La cantidad de filas y columnas del filtro debe ser la misma"))
    end

	# Inicializo una matriz de resultados con zeros, y calcula la cantidad de filas y columnas. Como no estoy aplicando padding, el resultado es una matriz con dos filas y dos columnas menos que la original.
    #result = zeros(input_r-2, input_c-2)
	result = zeros(input_r-filter_r+1, input_c-filter_c+1)
    result_r, result_c = size(result)

	# Me desplazo por columnas (j) y filas (i), aplicando el filtro en cada iteración.
    for i in 1:result_r
        for j in 1:result_c
			# Aplico el kernel, celda por celda
			for k in 1:filter_r
				for l in 1:filter_c
					result[i,j] += input[i+k-1,j+l-1] * filter[k,l]
				end
			end
        end
    end

    return result
end

# ╔═╡ d3040ef6-6695-471b-88d4-ed634cb382be
md"Y aplicamos la convolución:"

# ╔═╡ c60ae5c9-9719-4599-9e18-39bbb8e007a6
convolucion(input, filter)

# ╔═╡ 3be77870-3ab9-4b3c-ac8f-8de02d9513e3
md"### Aplicando la convolución _a mano_ a una imagen"

# ╔═╡ 09cdedfe-075d-497a-9a01-5bb38ec29be5
md"Carguemos una imagen cualquiera. Puede ser una imagen de nuestro disco rígido, o una imagen de internet. Como ejemplo voy a utilizar la una imagen extraide de *Freepik* descargada desde [este link](https://www.freepik.es/foto-gratis/disparo-vertical-leopardo-su-habitat-safari-delta-okavanga-botswana_24345392.htm#position=4):,"

# ╔═╡ 1a1d0ebb-8318-4c92-b30f-b76df137ff50
imagen = load("imagenes/disparo-vertical-de-un-leopardo-en-su-habitat-en-un-safari-en-el-delta-del-okavanga-en-botswana.jpg")

# ╔═╡ 7c53749d-3126-40ce-adec-5baf8ea1fa37
md"Veamos el tamaño de esta imagen:"

# ╔═╡ 1df24e76-0334-421e-9e12-1aaaa5666f07
size(imagen)

# ╔═╡ 01ab82d1-47f0-4497-a01b-931f5770f4a9
md"Julia procesa guarda las imágenes a color como una matriz de objetos _RGB_, los cuales encapsulan toda la información de color del pixel. Como son matrices, se pueden extraer sus píxeles mediante indexación:"

# ╔═╡ 2cf3e20a-3eb9-4f99-93f6-9dc8164439cc
imagen[1,1]

# ╔═╡ 99d3e9e5-ef6f-4dbc-b110-e5fd9f3d7ab4
println(imagen[1,1])

# ╔═╡ c130e2b4-688c-4509-8ef5-12e9090fd794
imagen[1000:3000,300:2500]

# ╔═╡ 83c9455a-59a8-43de-844c-8968cc286ab5
md"La imagen tiene 3 canales, para que nuestro algoritmo _a mano_ de convolución funcione, pasame a una canal, convirtiendo la imagen a escala de grises:"

# ╔═╡ 1e5547fd-5a34-4816-8dbd-5cf310ca98c0
imagen_gris = Gray.(imagen)

# ╔═╡ f860aef8-3851-4c30-9906-ee8ed1362414
md"Podemos ver el valor del primer pixel, en escala de gris normalizado entre 0 y 1:"

# ╔═╡ 90d1aaec-a125-40f0-85cc-5e4bd83e7ea1
println(Float64(imagen_gris[1,1]))

# ╔═╡ cd1243bd-55b7-4477-bb53-3bbef3a1cd30
md"De hecho, podemos ver toda la matríz:"

# ╔═╡ 2235ff92-255b-4b68-a7e1-bf57b0d40201
Float64.(imagen_gris)

# ╔═╡ 36efa1cf-5807-45fe-9a7d-03f214748f10
md"La matriz anterior, convolucionada, es:"

# ╔═╡ b5158c9c-a1c1-41d8-8507-a41f5437e61f
convolucion(imagen_gris, filter)

# ╔═╡ 2a07994c-3022-403a-a68d-021b155c4142
md"Y, si la queremos ver como imagen, podemos aplicar la función Gray para convertir la matriz en imagen (si, la misma función que usamos para convertir de RGB a escala de grises):"

# ╔═╡ 31ef15a9-8fc9-433d-b646-5ca4572a2491
Gray.(convolucion(imagen_gris, filter))

# ╔═╡ 86823881-bddf-4d20-8bb3-2b11be2019be
md"Probemos usar otro kernel:"

# ╔═╡ 9ecfe533-2c56-4038-8ae1-5b9e412c1bd5
Gray.(
	convolucion(
		imagen_gris, 
		[[0,0.5,0.1, 1] [0.1,0.5,0.5 ,1] [-2,-0.5,-1 , 1] [-2,-0.5,-1 , 1]]))

# ╔═╡ 354a53e4-941a-41a0-b07e-05cb0915e6a2
md"Probemos una doble convolución:"

# ╔═╡ d4411f18-b329-4f3e-a055-d53381fa524f
Gray.(convolucion(convolucion(imagen_gris, filter), filter))

# ╔═╡ a6bed5db-26b0-494f-a8e6-da71ae070509
md"Generemos un kernel al azar:"

# ╔═╡ 0f1b0ad0-c04a-4f14-8de2-3d1168999dc1
Gray.(convolucion(imagen_gris, rand(5,5)))

# ╔═╡ 3f7ae3a5-ab4d-44c0-aecc-d01c4d1b0ee0
md"Probemos aplicar filtros de Sobel:"

# ╔═╡ 390f9467-f1f1-40ef-a11b-73a52d898171
sobel_filter_x = [[1, 2, 1] [0, 0, 0] [-1, -2, -1]]

# ╔═╡ e578a95a-32c6-4acc-834f-929278990960
sobel_filter_y = [[1, 0, -1] [2, 0, -2] [1, 0, -1]]

# ╔═╡ 7082118c-ae9e-4f39-94a4-bc05bd9adda8
sobel_filter = sqrt.(sobel_filter_x.^2 + sobel_filter_y.^2)

# ╔═╡ 641d02e3-559e-4d15-a059-2fd291f9e3da
# Cambiá el tipo de filtro de Sobel
Gray.(convolucion(imagen_gris, sobel_filter_y ))

# ╔═╡ 20c1679d-bc87-487a-a3d5-49e0d75ea22f
md"Y ahora, prendamos la webcam y convolucionemos tu foto:"

# ╔═╡ f9aed4a3-7bf6-46c8-9492-7a772376c1aa
begin

	function camera_input(;max_size=200, default_url="https://i.imgur.com/SUmi94P.png")
	"""
	<span class="pl-image waiting-for-permission">
	<style>
		
		.pl-image.popped-out {
			position: fixed;
			top: 0;
			right: 0;
			z-index: 5;
		}
	
		.pl-image #video-container {
			width: 250px;
		}
	
		.pl-image video {
			border-radius: 1rem 1rem 0 0;
		}
		.pl-image.waiting-for-permission #video-container {
			display: none;
		}
		.pl-image #prompt {
			display: none;
		}
		.pl-image.waiting-for-permission #prompt {
			width: 250px;
			height: 200px;
			display: grid;
			place-items: center;
			font-family: monospace;
			font-weight: bold;
			text-decoration: underline;
			cursor: pointer;
			border: 5px dashed rgba(0,0,0,.5);
		}
	
		.pl-image video {
			display: block;
		}
		.pl-image .bar {
			width: inherit;
			display: flex;
			z-index: 6;
		}
		.pl-image .bar#top {
			position: absolute;
			flex-direction: column;
		}
		
		.pl-image .bar#bottom {
			background: black;
			border-radius: 0 0 1rem 1rem;
		}
		.pl-image .bar button {
			flex: 0 0 auto;
			background: rgba(255,255,255,.8);
			border: none;
			width: 2rem;
			height: 2rem;
			border-radius: 100%;
			cursor: pointer;
			z-index: 7;
		}
		.pl-image .bar button#shutter {
			width: 3rem;
			height: 3rem;
			margin: -1.5rem auto .2rem auto;
		}
	
		.pl-image video.takepicture {
			animation: pictureflash 200ms linear;
		}
	
		@keyframes pictureflash {
			0% {
				filter: grayscale(1.0) contrast(2.0);
			}
	
			100% {
				filter: grayscale(0.0) contrast(1.0);
			}
		}
	</style>
	
		<div id="video-container">
			<div id="top" class="bar">
				<button id="stop" title="Stop video">✖</button>
				<button id="pop-out" title="Pop out/pop in">⏏</button>
			</div>
			<video playsinline autoplay></video>
			<div id="bottom" class="bar">
			<button id="shutter" title="Click to take a picture">📷</button>
			</div>
		</div>
			
		<div id="prompt">
			<span>
			Enable webcam
			</span>
		</div>
	
	<script>
		// based on https://github.com/fonsp/printi-static (by the same author)
	
		const span = currentScript.parentElement
		const video = span.querySelector("video")
		const popout = span.querySelector("button#pop-out")
		const stop = span.querySelector("button#stop")
		const shutter = span.querySelector("button#shutter")
		const prompt = span.querySelector(".pl-image #prompt")
	
		const maxsize = $(max_size)
	
		const send_source = (source, src_width, src_height) => {
			const scale = Math.min(1.0, maxsize / src_width, maxsize / src_height)
	
			const width = Math.floor(src_width * scale)
			const height = Math.floor(src_height * scale)
	
			const canvas = html`<canvas width=\${width} height=\${height}>`
			const ctx = canvas.getContext("2d")
			ctx.drawImage(source, 0, 0, width, height)
	
			span.value = {
				width: width,
				height: height,
				data: ctx.getImageData(0, 0, width, height).data,
			}
			span.dispatchEvent(new CustomEvent("input"))
		}
		
		const clear_camera = () => {
			window.stream.getTracks().forEach(s => s.stop());
			video.srcObject = null;
	
			span.classList.add("waiting-for-permission");
		}
	
		prompt.onclick = () => {
			navigator.mediaDevices.getUserMedia({
				audio: false,
				video: {
					facingMode: "environment",
				},
			}).then(function(stream) {
	
				stream.onend = console.log
	
				window.stream = stream
				video.srcObject = stream
				window.cameraConnected = true
				video.controls = false
				video.play()
				video.controls = false
	
				span.classList.remove("waiting-for-permission");
	
			}).catch(function(error) {
				console.log(error)
			});
		}
		stop.onclick = () => {
			clear_camera()
		}
		popout.onclick = () => {
			span.classList.toggle("popped-out")
		}
	
		shutter.onclick = () => {
			const cl = video.classList
			cl.remove("takepicture")
			void video.offsetHeight
			cl.add("takepicture")
			video.play()
			video.controls = false
			console.log(video)
			send_source(video, video.videoWidth, video.videoHeight)
		}
		
		
		document.addEventListener("visibilitychange", () => {
			if (document.visibilityState != "visible") {
				clear_camera()
			}
		})
	
	
		// Set a default image
	
		const img = html`<img crossOrigin="anonymous">`
	
		img.onload = () => {
		console.log("helloo")
			send_source(img, img.width, img.height)
		}
		img.src = "$(default_url)"
		console.log(img)
	</script>
	</span>
	""" |> HTML
	end

	function process_raw_camera_data(raw_camera_data)
		# the raw image data is a long byte array, we need to transform it into something
		# more "Julian" - something with more _structure_.
		
		# The encoding of the raw byte stream is:
		# every 4 bytes is a single pixel
		# every pixel has 4 values: Red, Green, Blue, Alpha
		# (we ignore alpha for this notebook)
		
		# So to get the red values for each pixel, we take every 4th value, starting at 
		# the 1st:
		reds_flat = UInt8.(raw_camera_data["data"][1:4:end])
		greens_flat = UInt8.(raw_camera_data["data"][2:4:end])
		blues_flat = UInt8.(raw_camera_data["data"][3:4:end])
		
		# but these are still 1-dimensional arrays, nicknamed 'flat' arrays
		# We will 'reshape' this into 2D arrays:
		
		width = raw_camera_data["width"]
		height = raw_camera_data["height"]
		
		# shuffle and flip to get it in the right shape
		reds = reshape(reds_flat, (width, height))' / 255.0
		greens = reshape(greens_flat, (width, height))' / 255.0
		blues = reshape(blues_flat, (width, height))' / 255.0
		
		# we have our 2D array for each color
		# Let's create a single 2D array, where each value contains the R, G and B value of 
		# that pixel
		
		RGB.(reds, greens, blues)
	end

	md"<-- Cálculos internos -->"

end

# ╔═╡ e75af744-0d2c-4b9c-80be-df923f24b19c
@bind cam_data camera_input()

# ╔═╡ 023fafc4-bc64-4d8f-958c-9bd0019232c3
cam_image = Gray.(process_raw_camera_data(cam_data))

# ╔═╡ 32c9ffce-035e-42b3-82c0-38f3738833fc
Gray.(convolucion(cam_image, [[-1,0,0] [0,0,0] [0,0,1]]))

# ╔═╡ ca9b7992-ed68-460d-900d-82ec567c00b4
Gray.(convolucion(cam_image, [[-1,0,1] [-1,0,1] [-1,0,1]]))

# ╔═╡ afdb842a-2e08-4ea4-9bac-59f494bf4a57
Gray.(convolucion(cam_image, [[1,-1,1] [0,-1,0] [1,-1,1]]))

# ╔═╡ 5289e2f0-8412-49df-8cb4-bdb54b2df895
md"## Submuestreo"

# ╔═╡ 3c8e4b5f-6253-4228-b315-efd5bfd88619
md"El submuestreo ([_pooling_](https://en.wikipedia.org/wiki/Pooling_layer) en la bibliografía) es una operación que se utiliza para reducir el tamaño de una imagen. Esta reducción del tamaño, bien hecha, tiene muchos beneficios:

* Reduce el tamaño del modelo, achicando el costo de las operaciones de entrenamiento y predicción, tanto en consumo de CPU como de memoria.
* Insensibiliza el modelo ante pequeñas variaciones en la imagen.
* Reduce la cantidad de pesos en cero en etapas posteriores de la red.

El _pooling_ consiste, básicamente, en tomar bloques de píxeles y generar una nueva imagen en la cual cada bloque es reemplazado por un único píxel. Para realizar el reemplazo, usualmente se utilizan dos tipos de _pooling_:

* **Max-pooling**: aplicar la función _máximo_ a los píxeles del bloque y usar dicho valor.
* **Average-pooling**: aplicar la función _promedio_ a los píxeles del bloque y usar dicho valor.

En la siguiente imagen vemos un ejemplo de _Max-pooling_ sobre una imagen de un canal (en escala de grises, por ejemplo):

"

# ╔═╡ 2bada8d4-ad31-43a2-9f19-8472b385d0e9
md"""$(LocalResource("imagenes/max_pooling.png")) _Por Daniel Voigt Godoy - [https://github.com/dvgodoy/dl-visuals/, CC BY 4.0, https://commons.wikimedia.org/w/index.php?curid=150823502](https://github.com/dvgodoy/dl-visuals/, CC BY 4.0, https://commons.wikimedia.org/w/index.php?curid=150823502)_

"""

# ╔═╡ d9636e73-0c00-4ec1-9e89-92afd75e78af
md"A diferencia de la convolución, en la imagen anterior no vamos _moviendo_ un filtro sobre la imagen, sino que los bloques están fijos. En este caso se aplicó un _max-pooling_ de $2x2$ sobre una imagen de tamaño $6x6$, resultando en una nueva imagen de tamaño $3x3$. Podríamos tener un comportamiento de desplazamiento similar a la convolución cambiando el _stride_, pero en general se hace que el _stride_ sea del mismo tamaño que la matriz de _pooling_. Llegado el caso, podríamos agregar _padding_, pero no suelen ser operaciones utilizadas.
"

# ╔═╡ 820e420f-df7a-42df-9f99-6aa8fda6bf7b
md"### Haciendo un pooling a mano"

# ╔═╡ 370cdf22-58af-42ab-94a9-0bcf5130dc19
md"Vamos a implementar un _pooling a mano_ ahora, de manera similar a como hicimos con la convolución."

# ╔═╡ eb5552bd-3ca4-4ce5-bb8a-ae04fa0210b6
# Función para calcular el pooling, con padding y con stride. Como valores por defecto, asume que la matriz de pooling es de 2x2, el padding horizontal y el vertical son de 2, el stride es 2 y que la función a aplicar es el máximo (:max). Se puede cambiar la función a promedio también (_avg). El tamaño de la matriz es igual al stride en esta función.
function aplicar_pooling(input; ph::Int=2, pw::Int=2, stride::Int=2, operacion::Symbol=:max)
	
    H, W = size(input) # Calculo el tamaño de la imagen (altura y ancho)

	# Calculo el tamaño que va a tener la imagen de salida e inicializo la matriz.
    oh = 1 + (H - ph) ÷ stride
    ow = 1 + (W - pw) ÷ stride
    output = similar(input, Float32, oh, ow)

	# Recorro la imagen de salida, actualizando los valores aplicando la función elegida al bloque correspondiente de la imagen original.
    for r in 1:oh, c in 1:ow
        i = 1 + (r-1)*stride
        j = 1 + (c-1)*stride
        patch = @view input[i:i+ph-1, j:j+pw-1]
        if operacion == :max
            output[r,c] = maximum(patch)
        elseif operacion == :avg
            output[r,c] = mean(patch)
        else
            error("Operación no soportada: $operacion")
        end
    end
	
    return output
end

# ╔═╡ 6300425e-87d5-4758-ac99-101440919fff
md"Vamos a aplicarla a nuestra matriz _input_ original (si, la que está bien arriba del notebook):"

# ╔═╡ e1cf1fb5-43f6-47cc-9d73-60d982de8e00
aplicar_pooling(input)

# ╔═╡ ee3f79b7-4927-4a8c-aab7-51507a89d231
md"Como a la función le definimos valores por defecto, estos se pueden omitir y solo pasarle el input (la imagen a aplicar el _pooling_). Vamos a cambiar de _max-pooling_, la operación por defecto, a _average-pooling_:"

# ╔═╡ 469180f5-65f7-47fd-9326-34afde659992
aplicar_pooling(input, operacion=:avg)

# ╔═╡ 05c343a9-4bf6-48be-83ba-c0b07fc9359a
md"Con _stride_ de 1:"

# ╔═╡ 84e9ba9d-807f-4af1-b18e-acca54967596
aplicar_pooling(input, operacion=:avg, stride=1)

# ╔═╡ faed6e7e-c0b1-4aa6-8d99-884fe07cf9eb
md"¿Que le pasará a nuestro leopardo?:"

# ╔═╡ e1d7e5cb-a8be-4490-bc20-da125f8f6554
aplicar_pooling(imagen_gris)

# ╔═╡ 5ae0ffeb-4c3b-438f-a28f-539b93724b44
md"Al aplicar un _pooling_ por defecto de $2x2$ con _stride_ de 2, el tamaño de la imagen se reduce a la mitad. Cierto, hay que retransformarlo en imagen:"

# ╔═╡ 19f49f7d-6ccb-4420-8fad-83d9f84a3ff0
Gray.(aplicar_pooling(imagen_gris))

# ╔═╡ 45ac2c11-2b8c-4969-ba68-83a1fb06d1b2
md"Como es esperable, en imágenes de mucha resolución, la pérdida de información es muy poca si hacemos un _pooling_ de $2x2$. Y, sin embargo, reducimos la cantidad de píxeles en un factor de 4 ($6000 \cdot 4000=24000000$ a $3000 \cdot 2000=6000000$). Esto implica una cantidad muuuucho más chica de pesos para entrenar. Probemos aplicar un _pooling_ mas grande (matríz de $10x10$ con _stride_ de $10$):"

# ╔═╡ dae6d2e6-2437-4730-818b-95ba007ac98e
aplicar_pooling(imagen_gris, stride=10)

# ╔═╡ b9101d9b-0393-42a9-bf07-119a87985d2e
Gray.(aplicar_pooling(imagen_gris, stride=10))


# ╔═╡ 7c469aeb-d3e6-4a5d-8748-a2131a984899
md"La imagen es todavía reconocible, ¡y tuvimos una reducción de los pesos en un factor de 100!"

# ╔═╡ 144b1602-bd60-4c0b-bca2-7a8977414f5f
md"Si promediamos:"

# ╔═╡ c635cc2d-0b9d-40f3-a7b6-a6ca8f2427dc
Gray.(aplicar_pooling(imagen_gris, stride=10, operacion=:avg))

# ╔═╡ a3e8f233-5d89-45a5-b159-3f080871cce9
md"¿Que pasará con un tamaño de bloque igual a $50$?:"

# ╔═╡ 50ab3d42-ef12-4e87-b51f-cd36be95f64f
Gray.(aplicar_pooling(imagen_gris, stride=50, operacion=:avg))

# ╔═╡ 7bb493df-8c40-441d-8e50-c4c5464784f6
md"¿Y que pasa con la imagen de la webcam?. Probemos con un bloque de tamaño $10$:"

# ╔═╡ cbcae0c5-5f95-4b16-8c44-e3e520c1dce5
Gray.(aplicar_pooling(cam_image, stride=10, operacion=:avg))

# ╔═╡ 94775cf6-2781-4bd0-a0e7-36f673a3b776
md"¿Y un bloque de tamaño $2$?:"

# ╔═╡ c1f10d60-4500-42c1-a605-6f6b817baaf4
Gray.(aplicar_pooling(cam_image, stride=2, operacion=:avg))

# ╔═╡ 208b38fc-ee11-49ae-b941-29a40cfc7b38
md"Hagamos un _max-pooling_ en vez de un _average-pooling_:"

# ╔═╡ 41d7ea42-c7f0-40b3-8273-08c485dac4d4
Gray.(aplicar_pooling(cam_image, stride=2, operacion=:max))

# ╔═╡ 47a14a77-8e72-4939-ad76-1b20775fb482
md"## Convolución + submuestreo"

# ╔═╡ 1e0536bf-72d7-4186-a5de-5986d7577cfa
md"Apliquemos ahora una convolución y luego un submuestreo:"

# ╔═╡ e5164e18-1f80-448f-87a8-353cf673c5d6
Gray.(aplicar_pooling(convolucion(imagen_gris, filter), stride=10))

# ╔═╡ a83284d4-5bd4-4964-a1df-6440264360d0
md"Al revés ahora, submuestreo y convolución:"

# ╔═╡ 818e84b0-00d9-4a80-815e-9df2b4517a77
Gray.(convolucion(aplicar_pooling(imagen_gris, stride=10), filter))

# ╔═╡ ff1d4fb8-196a-486a-8fae-dbb310147760
md"Encadenemos convolución, submuestreo, convolución y submuestreo:"

# ╔═╡ 092cb977-99fd-4638-af3c-9391f3307d1c
Gray.(
	aplicar_pooling(
		convolucion(
			aplicar_pooling(
				convolucion(imagen_gris, filter), 
				stride=10), 
			filter), 
		stride=10))

# ╔═╡ 3c1c393a-6c03-492c-9453-b2f534d4d24f
md"Lo mismo, con tamaño de bloque $4x4$:"

# ╔═╡ 892d0d17-1d39-49cb-a485-d5b4e23d3c60
Gray.(
	aplicar_pooling(
		convolucion(
			aplicar_pooling(
				convolucion(imagen_gris, filter), 
				stride=4), 
			filter), 
		stride=4))

# ╔═╡ Cell order:
# ╟─d8bac896-27db-4130-9e65-cc365b5123dd
# ╟─2e6478a5-9c5f-4e43-aedf-63b099dd07d2
# ╟─76894c82-42d4-488f-ad0b-73a8e75e7a47
# ╠═b002c1b8-a2c0-11f0-347c-e98e7815a9f6
# ╟─43b69b09-ee24-4a47-91ae-5c6ca70ba8d5
# ╟─76a1d3c4-7eda-4e39-9683-6a3e105adec6
# ╟─424dce09-39fe-4fc9-a88b-1932894780f6
# ╟─4fd1bbae-0aaa-421e-899d-787ec9684d36
# ╟─8451bb76-a967-4dd8-bc2f-bcf9b7d0ae96
# ╟─c0a08cb6-fbd0-4046-b6cf-8495132a17c5
# ╟─d29f4df6-1806-4469-98f0-3e9e536df47e
# ╟─086b5efb-fa17-4927-9578-51b0d570e64c
# ╟─0037258a-e2dc-4aba-aa58-8328b1948f3a
# ╟─79afdf25-ddd4-4978-9fd2-8bbef9d39aa3
# ╟─73b64fa8-bfb9-4c2f-8e66-709ca7b52c05
# ╟─572e185b-6975-4619-b15f-a92f5d766dd1
# ╟─545ede3e-03fa-4661-b5a2-5bd0be95b6ef
# ╟─01fe8cea-ffea-4e2a-b727-d1bdebe11a93
# ╟─325a2f62-09c5-4cb1-bbc2-e3176df2e008
# ╠═fc2775fb-1401-4074-950c-6b48263761dd
# ╠═6a145ce8-bb01-4c52-8004-4498555f43fd
# ╟─9083eb03-c220-4ea4-99ed-e1d26909d007
# ╠═eaf1bdac-a5a7-48b8-ad4e-ec2aaf23dcd7
# ╟─d3040ef6-6695-471b-88d4-ed634cb382be
# ╠═c60ae5c9-9719-4599-9e18-39bbb8e007a6
# ╟─3be77870-3ab9-4b3c-ac8f-8de02d9513e3
# ╟─09cdedfe-075d-497a-9a01-5bb38ec29be5
# ╠═1a1d0ebb-8318-4c92-b30f-b76df137ff50
# ╟─7c53749d-3126-40ce-adec-5baf8ea1fa37
# ╠═1df24e76-0334-421e-9e12-1aaaa5666f07
# ╟─01ab82d1-47f0-4497-a01b-931f5770f4a9
# ╠═2cf3e20a-3eb9-4f99-93f6-9dc8164439cc
# ╠═99d3e9e5-ef6f-4dbc-b110-e5fd9f3d7ab4
# ╠═c130e2b4-688c-4509-8ef5-12e9090fd794
# ╟─83c9455a-59a8-43de-844c-8968cc286ab5
# ╠═1e5547fd-5a34-4816-8dbd-5cf310ca98c0
# ╟─f860aef8-3851-4c30-9906-ee8ed1362414
# ╠═90d1aaec-a125-40f0-85cc-5e4bd83e7ea1
# ╟─cd1243bd-55b7-4477-bb53-3bbef3a1cd30
# ╠═2235ff92-255b-4b68-a7e1-bf57b0d40201
# ╟─36efa1cf-5807-45fe-9a7d-03f214748f10
# ╠═b5158c9c-a1c1-41d8-8507-a41f5437e61f
# ╟─2a07994c-3022-403a-a68d-021b155c4142
# ╠═31ef15a9-8fc9-433d-b646-5ca4572a2491
# ╟─86823881-bddf-4d20-8bb3-2b11be2019be
# ╠═9ecfe533-2c56-4038-8ae1-5b9e412c1bd5
# ╟─354a53e4-941a-41a0-b07e-05cb0915e6a2
# ╠═d4411f18-b329-4f3e-a055-d53381fa524f
# ╟─a6bed5db-26b0-494f-a8e6-da71ae070509
# ╠═0f1b0ad0-c04a-4f14-8de2-3d1168999dc1
# ╟─3f7ae3a5-ab4d-44c0-aecc-d01c4d1b0ee0
# ╠═390f9467-f1f1-40ef-a11b-73a52d898171
# ╠═e578a95a-32c6-4acc-834f-929278990960
# ╠═7082118c-ae9e-4f39-94a4-bc05bd9adda8
# ╠═641d02e3-559e-4d15-a059-2fd291f9e3da
# ╟─20c1679d-bc87-487a-a3d5-49e0d75ea22f
# ╟─f9aed4a3-7bf6-46c8-9492-7a772376c1aa
# ╠═e75af744-0d2c-4b9c-80be-df923f24b19c
# ╠═023fafc4-bc64-4d8f-958c-9bd0019232c3
# ╠═32c9ffce-035e-42b3-82c0-38f3738833fc
# ╠═ca9b7992-ed68-460d-900d-82ec567c00b4
# ╠═afdb842a-2e08-4ea4-9bac-59f494bf4a57
# ╟─5289e2f0-8412-49df-8cb4-bdb54b2df895
# ╟─3c8e4b5f-6253-4228-b315-efd5bfd88619
# ╟─2bada8d4-ad31-43a2-9f19-8472b385d0e9
# ╟─d9636e73-0c00-4ec1-9e89-92afd75e78af
# ╟─820e420f-df7a-42df-9f99-6aa8fda6bf7b
# ╟─370cdf22-58af-42ab-94a9-0bcf5130dc19
# ╠═eb5552bd-3ca4-4ce5-bb8a-ae04fa0210b6
# ╟─6300425e-87d5-4758-ac99-101440919fff
# ╠═e1cf1fb5-43f6-47cc-9d73-60d982de8e00
# ╟─ee3f79b7-4927-4a8c-aab7-51507a89d231
# ╠═469180f5-65f7-47fd-9326-34afde659992
# ╟─05c343a9-4bf6-48be-83ba-c0b07fc9359a
# ╠═84e9ba9d-807f-4af1-b18e-acca54967596
# ╟─faed6e7e-c0b1-4aa6-8d99-884fe07cf9eb
# ╠═e1d7e5cb-a8be-4490-bc20-da125f8f6554
# ╟─5ae0ffeb-4c3b-438f-a28f-539b93724b44
# ╠═19f49f7d-6ccb-4420-8fad-83d9f84a3ff0
# ╟─45ac2c11-2b8c-4969-ba68-83a1fb06d1b2
# ╠═dae6d2e6-2437-4730-818b-95ba007ac98e
# ╠═b9101d9b-0393-42a9-bf07-119a87985d2e
# ╟─7c469aeb-d3e6-4a5d-8748-a2131a984899
# ╟─144b1602-bd60-4c0b-bca2-7a8977414f5f
# ╠═c635cc2d-0b9d-40f3-a7b6-a6ca8f2427dc
# ╠═a3e8f233-5d89-45a5-b159-3f080871cce9
# ╠═50ab3d42-ef12-4e87-b51f-cd36be95f64f
# ╟─7bb493df-8c40-441d-8e50-c4c5464784f6
# ╠═cbcae0c5-5f95-4b16-8c44-e3e520c1dce5
# ╟─94775cf6-2781-4bd0-a0e7-36f673a3b776
# ╠═c1f10d60-4500-42c1-a605-6f6b817baaf4
# ╟─208b38fc-ee11-49ae-b941-29a40cfc7b38
# ╠═41d7ea42-c7f0-40b3-8273-08c485dac4d4
# ╟─47a14a77-8e72-4939-ad76-1b20775fb482
# ╟─1e0536bf-72d7-4186-a5de-5986d7577cfa
# ╠═e5164e18-1f80-448f-87a8-353cf673c5d6
# ╟─a83284d4-5bd4-4964-a1df-6440264360d0
# ╠═818e84b0-00d9-4a80-815e-9df2b4517a77
# ╟─ff1d4fb8-196a-486a-8fae-dbb310147760
# ╠═092cb977-99fd-4638-af3c-9391f3307d1c
# ╟─3c1c393a-6c03-492c-9453-b2f534d4d24f
# ╠═892d0d17-1d39-49cb-a485-d5b4e23d3c60
