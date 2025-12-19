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

# ╔═╡ ab290805-55ef-4d8b-8cd1-7e14ee9dbffb
begin
	using Pkg
	Pkg.activate()
	
	using Random, Statistics, LinearAlgebra, Distributions
	using PlutoUI, Plots
	Plots.default(; legend=:topright)

	TableOfContents(title="Contenido")
end

# ╔═╡ 31c0dfa4-8288-11f0-189a-bff3372c8817
md"""
# Subajuste vs Sobreajuste — Notebook interactivo

Este notebook permite explorar **subajuste** (modelo muy simple que no captura la estructura) y **sobreajuste** (modelo demasiado flexible que memoriza el ruido), y ver por qué conviene separar los datos en **entrenamiento** y **test**.

- Elegí una **función real** (cúbica o senoidal), el **ruido** y el **tamaño muestral**.
- Ajustá una **regresión polinómica** de grado variable (y, opcionalmente, con **Ridge**).
- Observá la **curva de error** (train/test) al variar el grado: típicamente el error de test forma una curva en “U”.
"""

# ╔═╡ dc8a09ec-b6ef-4a6f-b75b-9e7383ee7b70
md"## Setup"

# ╔═╡ 984b3fd8-f8a2-4611-a02b-913e9e01a307
md"## Modelo Interactivo"

# ╔═╡ adeedac8-fc6b-4867-ab49-3cef8fcb8046
md"""
|Parámetro|Valor|
|---------|-----|
|Semilla: |$(@bind seed Slider(1:1000, default=42, show_value=true))|
|Función verdadera: |$(@bind truth Select([ :cubic =>"Cúbica suave",  :sine=> "Senoidal"]))|
|N: |$(@bind N Slider(20:5:300, default=80, show_value=true))|
|Ruido: |$(@bind σ Slider(0.0:0.05:2.0, default=0.4, show_value=true))|
|Fracción de entrenamiento: |$(@bind train_frac Slider(0.5:0.05:0.9, default=0.7, show_value=true))|
|Dominio: |$(@bind domain Select([(-3.0, 3.0) => "[-3, 3]", (-1.0, 1.0) => "[-1, 1]", (0.0, 2π) => "[0, 2π]"]))|
"""


# ╔═╡ e944a431-229f-4040-82a1-9302ce0399b9
begin
	# Funciones "reales" para generar datos (sin y con ruido)
	function f_real(x, truth::Symbol)
	    if truth == :cubic
	        return 0.5 .* x.^3 .- 1.5 .* x .+ 1.0  # cúbica suave
	    else
	        return 2.0 .* sin.(x) .+ 0.3 .* x      # senoidal con tendencia
	    end
	end
	
	# Generador de muestra
	function make_dataset(N, σ, domain::Tuple{<:Real,<:Real}, truth::Symbol, seed::Int)
	    rng = MersenneTwister(seed)
	    x = rand(rng, Uniform(domain[1], domain[2]), N)
	    y_clean = f_real(x, truth)
	    y = y_clean .+ σ .* randn(rng, N)
	    x, y, y_clean
	end
	
	# Train/test split
	function train_test_split(x, y, frac::Real, seed::Int)
	    rng = MersenneTwister(seed + 12345) # offset para que el split no cambie con el ruido
	    n = length(x)
	    idx = shuffle(rng, collect(1:n))
	    ntr = round(Int, frac*n)
	    itr = idx[1:ntr]; ite = idx[ntr+1:end]
	    x[itr], y[itr], x[ite], y[ite]
	end
	
	x, y, y_clean = make_dataset(N, σ, domain, truth, seed)
	xtr, ytr, xte, yte = train_test_split(x, y, train_frac, seed)

	# Matriz de diseño de grado d: [1 x x^2 ... x^d]
	design_matrix(x::AbstractVector, d::Int) = reduce(hcat, (x.^k for k in 0:d))
	
	# Ajuste por mínimos cuadrados con Ridge (sin regularizar el sesgo por defecto)
	function fit_ridge(X, y; λ=0.0, reg_bias=false)
	    p = size(X, 2)
	    # Regularizador: no penalizamos el sesgo (columna 1) salvo que reg_bias=true
	    R = λ .* I(p)
	    if !reg_bias && p ≥ 1
	        R = copy(R)
	        R[1,1] = 0.0
	    end
	    β = (X'X .+ R) \ (X'y)
	    β
	end
	
	# Predicción
	predict(X, β) = X * β

	md"<-- Cálculos internos -->"
end

# ╔═╡ dfc974b1-dbee-4090-9dfa-49fd256ea34b
md"""
|Parámetro|Valor|
|---------|-----|
|Grado: |$(@bind degree Slider(1:30, default=3, show_value=true))|
|Usar Ridge: |$(@bind use_ridge CheckBox(default=false))|
|Log(λ): |$(@bind log10λ Slider(-6.0:0.5:2.0, default=-6.0, show_value=true))|
"""

# ╔═╡ e7d2d5f2-69f3-4ef6-8232-eac06df422f6
λ = use_ridge ? 10.0^log10λ : 0.0

# ╔═╡ feec553a-3ed4-4180-8bc4-e491baa61ff7
begin
	# Entrenamiento
	Xtr = design_matrix(xtr, degree)
	β = fit_ridge(Xtr, ytr; λ=λ, reg_bias=false)
	
	# Métricas (MSE)
	mse(ŷ, y) = mean((ŷ .- y).^2)
	
	# En train:
	ŷtr = predict(Xtr, β)
	mse_train = mse(ŷtr, ytr)
	
	# En test:
	Xte = design_matrix(xte, degree)
	ŷte = predict(Xte, β)
	mse_test = mse(ŷte, yte)
	
	# Heurística simple de diagnóstico:
	# - Subajuste: error grande y similar en train/test
	# - Sobreajuste: train muy bajo y test mucho más alto
	# - Adecuado: intermedio
	ratio = mse_test / max(mse_train, 1e-12)
	underfit = (mse_train > var(y) * 0.6) && (abs(mse_test - mse_train) / max(mse_test, mse_train) < 0.35)
	overfit  = (mse_train < var(y) * 0.3) && (ratio > 1.5)
	diagnóstico = underfit ? "Subajuste" : overfit ? "Sobreajuste" : "Ajuste adecuado"
end

# ╔═╡ 1e0bdf02-d8c5-464d-9aa3-ad42b6444aee
begin
	# Grid para visualizar curvas
	xmin, xmax = domain
	xgrid = range(xmin, xmax; length=400)
	
	# Curva real y predicción del modelo
	Yreal = f_real(xgrid, truth)
	Xgrid = design_matrix(collect(xgrid), degree)
	Yhat  = predict(Xgrid, β)
	
	p1 = plot(xgrid, Yreal; lw=3, label="f(x) real")
	scatter!(xtr, ytr; ms=4, alpha=0.75, label="Train")
	plot!(xgrid, Yhat; lw=3, label="Modelo (grado = $(degree), λ = $(round(λ,sigdigits=3)))")
	title!(p1, "Ajuste — Diagnóstico: **$(diagnóstico)**")
	xlabel!(p1, "x"); ylabel!(p1, "y")
	annotate!(xmin + 0.02*(xmax-xmin), maximum(Yreal), text("MSE train = $(round(mse_train, sigdigits=4))", 8))
	p1
end

# ╔═╡ 994fff6f-ee34-40ca-889b-e16adb8b8a97
begin

	p2 = plot(xgrid, Yreal; lw=3, label="f(x) real")
	scatter!(xte, yte; ms=4, alpha=0.75, label="Test")
	plot!(xgrid, Yhat; lw=3, label="Modelo (grado = $(degree), λ = $(round(λ,sigdigits=3)))")
	title!(p2, "Partición test")
	xlabel!(p2, "x"); ylabel!(p1, "y")
	annotate!(xmin + 0.02*(xmax-xmin), maximum(Yreal) - 0.07*(maximum(Yreal)-minimum(Yreal)),text("MSE test  = $(round(mse_test, sigdigits=4))", 8))
	p2
end

# ╔═╡ f1caecbc-7be1-4f5c-b8c0-f573c40ef49a
md"""Grado máximo: $(@bind maxD Slider(1:1:30, default=10, show_value=true))"""

# ╔═╡ 64fabd28-5adb-48d4-9d4e-5e7d72ba632a
begin
	degs = 1:maxD
	mse_tr = similar(collect(degs), Float64)
	mse_te = similar(collect(degs), Float64)
	for (i,d) in enumerate(degs)
	    Xtr_d = design_matrix(xtr, d)
	    βd = fit_ridge(Xtr_d, ytr; λ=λ, reg_bias=false)
	    mse_tr[i] = mse(predict(Xtr_d, βd), ytr)
	    Xte_d = design_matrix(xte, d)
	    mse_te[i] = mse(predict(Xte_d, βd), yte)
	end
	
	p3 = plot(degs, mse_tr; lw=2, marker=:circle, label="MSE Train")
	plot!(p3, degs, mse_te; lw=2, marker=:square, label="MSE Test")
	vline!(p3, [degree]; lw=2, linestyle=:dash, label="grado actual")
	title!(p3, "Error vs. Grado del Polinomio (λ = $(round(λ,sigdigits=3)))")
	xlabel!(p3, "Grado"); ylabel!(p3, "MSE")
	p3
end

# ╔═╡ c04d2581-7e42-40e2-a405-023cc527503e
md"""
## ¿Qué mirar?

- **Subajuste**: el modelo no sigue la estructura de la función real → errores altos en train y test, similares.  
- **Sobreajuste**: el modelo se pega a puntos de train (error bajísimo en train) pero generaliza mal → **test** grande.  
- **Ajuste adecuado**: errores moderados y cercanos; la curva de test suele ser mínima a un cierto **grado** y luego empeora.

### Consejos de uso
- Aumentá el **grado** y observá cómo baja el error de train y qué le pasa al de test.
- Agregá **regularización (Ridge)** para domar el sobreajuste con altos grados.
- Probá distintos **tamaños muestrales (N)** y **niveles de ruido (σ)**.
- Cambiá la **función real** y el **dominio** para ver comportamientos distintos.
"""

# ╔═╡ Cell order:
# ╟─31c0dfa4-8288-11f0-189a-bff3372c8817
# ╟─dc8a09ec-b6ef-4a6f-b75b-9e7383ee7b70
# ╟─ab290805-55ef-4d8b-8cd1-7e14ee9dbffb
# ╟─984b3fd8-f8a2-4611-a02b-913e9e01a307
# ╟─adeedac8-fc6b-4867-ab49-3cef8fcb8046
# ╟─e944a431-229f-4040-82a1-9302ce0399b9
# ╟─dfc974b1-dbee-4090-9dfa-49fd256ea34b
# ╟─e7d2d5f2-69f3-4ef6-8232-eac06df422f6
# ╟─feec553a-3ed4-4180-8bc4-e491baa61ff7
# ╟─1e0bdf02-d8c5-464d-9aa3-ad42b6444aee
# ╟─994fff6f-ee34-40ca-889b-e16adb8b8a97
# ╟─f1caecbc-7be1-4f5c-b8c0-f573c40ef49a
# ╟─64fabd28-5adb-48d4-9d4e-5e7d72ba632a
# ╟─c04d2581-7e42-40e2-a405-023cc527503e
