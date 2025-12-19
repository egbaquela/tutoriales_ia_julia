### A Pluto.jl notebook ###
# v0.20.18

using Markdown
using InteractiveUtils

# ╔═╡ ca00dede-dded-48b2-b64f-28da8d959031
begin
	using Pkg
	Pkg.activate()

	using PlutoUI
	
	using HTTP, JSON3, Base64

	TableOfContents(title="Contenido")
end

# ╔═╡ 37fe2e7e-addc-11f0-034c-21c569f7ea8a
md"# Usando Roboflow para clasificar imágenes"

# ╔═╡ b17053bd-6015-4be0-b969-7ceeb9fc1a4c
md"## Setup"

# ╔═╡ 3199ff54-bfe1-4c16-98c1-c82b68e3df9d
md"## Interface con Roboflow"

# ╔═╡ fa95faf7-6e58-42bf-bcf5-0475f52ef5b7
	"""
	roboflow_classify(image_path; api_key, project, version) -> JSON3.Object

	Envía una imagen para clasificar al proyecto y versión del clasificador de Roboflow pasado en los parámetros, y devuelve un JSON con el resultado de la inferencia.
	"""
	function roboflow_classify(image_path::AbstractString; 
							   api_key::AbstractString,
	                           project::AbstractString, 
							   version::Integer)
	    url = "https://classify.roboflow.com/$(project)/$(version)?api_key=$(api_key)"
	    # Leemos bytes y codificamos en base64 (sin saltos de línea)
	    img_b64 = base64encode(open(read, image_path))
	
	    # TBRL: Roboflow acepta el body crudo en base64 (equivalente a `-d @-` en CURL).
	    # Si hiciera falta, se podría probar "text/plain" en vez de x-www-form-urlencoded.
	    headers = ["Content-Type" => "application/x-www-form-urlencoded"]
	
	    resp = HTTP.post(url, headers, img_b64)
	    body = String(resp.body)
	
	    if resp.status != 200
	        error("Roboflow devolvió $(resp.status): $body")
	    end
	    return JSON3.read(body)
	end

# ╔═╡ bcf401dc-5e43-4693-b66e-8cdccbc6f960
md"## Probando la interface"

# ╔═╡ 07db17af-971c-44ff-868c-fc9f6c29128c
begin
	apikey = "acá_va_el_apikey"
	project="acá_va_el_proyecto"
	version=3 # Poner la versión correcta
end

# ╔═╡ 941fb00d-f005-4358-b41c-497154451a04
result = roboflow_classify("imagenes/mate_01.png";
						   api_key=api_key,
	                       project=project,
	                       version=version)


# ╔═╡ 039bbe94-570b-4c3f-84b2-b13430ef33b4
result.top

# ╔═╡ Cell order:
# ╟─37fe2e7e-addc-11f0-034c-21c569f7ea8a
# ╟─b17053bd-6015-4be0-b969-7ceeb9fc1a4c
# ╠═ca00dede-dded-48b2-b64f-28da8d959031
# ╟─3199ff54-bfe1-4c16-98c1-c82b68e3df9d
# ╠═fa95faf7-6e58-42bf-bcf5-0475f52ef5b7
# ╟─bcf401dc-5e43-4693-b66e-8cdccbc6f960
# ╠═07db17af-971c-44ff-868c-fc9f6c29128c
# ╠═941fb00d-f005-4358-b41c-497154451a04
# ╠═039bbe94-570b-4c3f-84b2-b13430ef33b4
