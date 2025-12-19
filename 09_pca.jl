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

# ╔═╡ 322e0a56-3e75-4e79-92a1-261071acd38a
begin
	using Pkg
	Pkg.activate()
end

# ╔═╡ 7f899147-644b-4b58-b2da-f9f9f16a5174
begin
    using PlutoUI, LinearAlgebra, Statistics, Random, Distributions, DataFrames
    using Plots
    using MLJ
    using MLJMultivariateStatsInterface  # PCA base
	using NearestNeighborModels
	using MLJDecisionTreeInterface
	using MLJLinearModels
		
    TableOfContents(title="Contenido")
end

# ╔═╡ 7a74fcfd-345e-487e-8fb5-b8cae545e8fc
begin
    # Acepta siempre los términos/licencias sin pedir confirmación
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    # (Opcional) elegí dónde guardar los datasets
    # ENV["DATADEPS_DIR"] = "/ruta/a/mis/datasets"
    using DataDeps
    using MLDatasets
end

# ╔═╡ 9b9825b6-92f7-11f0-0b64-330dc66d8bc8
md"# Introducción al método de componentes principales (PCA)"

# ╔═╡ 3d3658cf-7d1f-4ab3-8f22-dcf8dd0fc0bb
md"## Setup"

# ╔═╡ 5c39d5c9-9992-4e12-a35d-4613c1dee449
md"## PCA"

# ╔═╡ 349cae87-b4c4-42fa-8f6a-f3693311a91c
md"""
- Sea $X \in \mathbb{R}^{n\times p}$ la matriz de datos, con filas como observaciones y columnas como variables.  
- Centramos: $\tilde X = X - \mathbf{1}\mu^\top$, con $\mu$ el vector de medias columna.  
- **Matriz de covarianza**: $\Sigma = \frac{1}{n-1}\tilde X^\top \tilde X$ (simétrica, semidefinida positiva).
- **Autovector y autovalor**: Para $A \in \mathbb{R}^{p\times p}$, un vector no nulo $v$ es **autovector** si  
  $$A v = \lambda v,$$  
  donde $\lambda$ es el **autovalor** asociado.  
  Para $\Sigma$, sus autovectores (columnas de $V$) forman una base ortonormal, y los autovalores $\lambda_1\ge\cdots\ge\lambda_p\ge 0$ miden la **varianza** capturada por cada dirección principal.

- **Componentes principales**: $Z = \tilde X V$; si tomamos las primeras $k$ columnas de $V$ como $V_k$, entonces $Z_k=\tilde X V_k$ maximiza la varianza explicada con $k$ dimensiones.

- **Reconstrucción** (con $k\le p$):  
  $$\hat X^{(k)} = Z_k V_k^\top + \mathbf{1}\mu^\top.$$

**Limitaciones y notas**  
- PCA es **lineal**: capta relaciones lineales (no descubre estructuras altamente no lineales sin extensiones, p.ej., kernel PCA).  
- Sensible a **escala**: conviene estandarizar variables si tienen unidades/escala distintas.  
- Los signos de los autovectores son **arbitrarios** (si $v$ es autovector, también lo es $-v$).  
- Es **no supervisado**: no usa etiquetas; puede mejorar o no el desempeño predictivo, según el problema.  
"""

# ╔═╡ 66d5a755-e1f9-46cd-866d-b3634fbf0409
md"## Caso 2D"

# ╔═╡ 01231e45-4422-4d17-954a-8cff1faea385
md"""
|Parámetro|Valor|
|---------|-----|
|Número de muestras|$(@bind n2d Slider(50:10:1000, default=300, show_value=true))|
|Varianza 1|$(@bind var1 Slider(0.9:0.1:3.0, default=1.0, show_value=true))|
|Varianza 2|$(@bind var2 Slider(0.9:0.1:3.0, default=2.0, show_value=true))|
"""

# ╔═╡ 06a861a6-2330-4d83-95f4-2dee347dc6db
begin
	Random.seed!(1234)
    # Datos 2D correlacionados
    μ = [0.0, 0.0]
    Σ = [var1 0.8; 0.8 var2]  # simétrica, p.s.d.
    dist = MvNormal(μ, Σ)
    X2 = rand(dist, n2d)'  # n x 2
    X2_mean = mean(X2, dims=1)
    X2c = X2 .- X2_mean
    Σhat = cov(X2c; corrected=true)
    ev = eigen(Σhat)  # ev.values (asc), ev.vectors (columns = eigenvectors)
    # Ordenar de mayor a menor autovalor
    idx = sortperm(ev.values, rev=true)
    λ = ev.values[idx]
    V = ev.vectors[:, idx]
    V1, V2 = V[:,1], V[:,2]
    λ1, λ2 = λ[1], λ[2]
    X2_proj1 = X2c * (V1*V1') .+ X2_mean  # reconstrucción k=1
	ev
end

# ╔═╡ 3c57a9fa-2558-4b70-906a-881a41abeec8
begin
    p = scatter(X2[:,1], X2[:,2], label="Datos", aspect_ratio=:equal, ms=3, alpha=0.6)
    scatter!(p, [X2_mean[1]], [X2_mean[2]], label="Media", ms=5, color=:black)
    # Ejes principales (desde la media)
    scale = 3*sqrt.(λ) # longitud proporcional a varianza
    plot!(p, [X2_mean[1], X2_mean[1] + scale[1]*V1[1]],
             [X2_mean[2], X2_mean[2] + scale[1]*V1[2]], lw=3, label="PC1")
    plot!(p, [X2_mean[1], X2_mean[1] + scale[2]*V2[1]],
             [X2_mean[2], X2_mean[2] + scale[2]*V2[2]], lw=3, label="PC2")
	title!(p, "Componentes principales")
    p
end

# ╔═╡ e38cab34-6f39-4ed8-9f63-051fe369dd7b
md"""
Cantidad de componentes a utilizar: $(@bind use_k Select(["k = 1","k = 2"]; default="k = 1"))
"""

# ╔═╡ 998d8b78-f011-4606-adca-0838a8e25cb8
begin
    if use_k == "k = 1"
        p2 = scatter(X2[:,1], X2[:,2], label="Original", ms=3, alpha=0.35, aspect_ratio=:equal)
        scatter!(p2, X2_proj1[:,1], X2_proj1[:,2], label="Recon (k=1)", ms=3)
        # Error de reconstrucción MSE
        mse = mean(sum(((X2 - X2_proj1).^2), dims=2))
        title!(p2, "Reconstrucción k=1 — MSE = $(round(mse, digits=4))")
        p2
    else
        # k=2 reconstruye perfecto (redondeo numérico)
        X2_proj2 = X2c * (V*V') .+ X2_mean
        mse2 = mean(sum(((X2 - X2_proj2).^2), dims=2))
        p2 = scatter(X2[:,1], X2[:,2], label="Original (k=2 ~ exacto)", ms=3, alpha=0.8, aspect_ratio=:equal)
        title!(p2, "Reconstrucción k=2 — MSE ≈ $(round(mse2, digits=6))")
        p2
    end
end

# ╔═╡ d4decac7-a54c-4dd0-a9ad-c31d27296292
begin
    varexp = λ ./ sum(λ)
    md"""
    **Autovalor 1:** $(round(λ[1], digits=4))  
    **Autovalor 2:** $(round(λ[2], digits=4))  
    **Varianza explicada:** $(round.(varexp, digits=4))  
    **Acumulada:** $(round.(cumsum(varexp), digits=4))
    """
end

# ╔═╡ 85f64ad3-9ac1-426c-b560-48774dbf11b7
begin
    p_scree = bar(1:2, λ, xlabel="Componente", ylabel="Autovalor", legend=false, title="Scree plot (2D)")
    p_scree
end

# ╔═╡ 78e07d2b-5f7a-4fe4-9572-f7923ac91dee
md"*Matriz Original*"

# ╔═╡ 6fdc915e-9c83-4452-a8dd-5af0d341e9fd
X2

# ╔═╡ 65bc37ab-1726-4dda-b76d-ef83e712f0b5
md"*Matriz proyectada en la primer componente*"

# ╔═╡ 91a80120-7d88-4195-a85f-9e7fb90e7840
X2_proj1

# ╔═╡ 03bf9016-9d33-4d90-87cb-b755b8172541
md"""## Caso supervisado: ¿PCA mejora la performance?"""

# ╔═╡ 0d8e9792-5c50-4487-b999-4af95ce4390a
md"""
Generaremos un dataset **alta dimensión** con 2 características informativas y muchas de ruido; luego comparamos:
- **Sin PCA**: Standardizer → Clasificador  
- **Con PCA**: Standardizer → PCA (manteniendo una fracción de varianza) → Clasificador

Usaremos **MLJ** con validación cruzada
"""

# ╔═╡ 9531a831-cdf2-448f-8476-4da01053d952
begin
	md"""
|Parámetro|Valor|
|---------|-----|
|Número de muestras|$(@bind n_samples Slider(200:50:3000, default=800, show_value=true))|
|Número de variables de ruido|$(@bind n_noise   Slider(0:5:200, default=80, show_value=true))|
|Separación de clases|$(@bind sep       Slider(0.5:0.1:3.0, default=2.3, show_value=true))|
|Ratio PCA|$(@bind pratio    Slider(0.50:0.05:0.999, default=0.55, show_value=true))|
|k (para KNN)|$(@bind k_nn      Slider(1:1:25, default=7, show_value=true))|
	"""
end

# ╔═╡ cec72b70-73f6-4440-bcc5-c9191cf08a00
begin
    # 2D informativo (clases separadas por traslación), y resto ruido gaussiano
    function make_dataset(n; n_noise=80, sep=1.5)
        n1 = n ÷ 2; n2 = n - n1
        μ1 = [sep, sep]; μ2 = [-sep, -sep]
        Σ = [1.0 0.6; 0.6 1.5]
        X1 = rand(MvNormal(μ1, Σ), n1)'
        X2 = rand(MvNormal(μ2, Σ), n2)'
        Xbase = vcat(X1, X2)                          # n x 2
        y = vcat(fill("A", n1), fill("B", n2))
        # Ruido (correlacionado débilmente entre sí y con base)
        R = randn(n, n_noise)
        X = hcat(Xbase, R)                            # n x (2 + n_noise)
        X, y
    end
    X, y = make_dataset(n_samples; n_noise=n_noise, sep=sep)
    #size(X), length(y)

	Xtbl = DataFrame(MLJ.table(X))  # convierte a tabla MLJ
    yv = categorical(y)
    sch = schema(Xtbl)
    md"Dimensiones de la muestra: **filas=$(size(Xtbl)[1])**, **columnas=$(size(Xtbl)[2])**"
end

# ╔═╡ 3e0305c4-1147-4fcb-8105-04d91ab4d3c8
sch

# ╔═╡ 9999248c-7c74-488c-b919-2e6a177a60fc
begin
    # Pipelines
    pipe_no_pca = Standardizer() |> KNNClassifier(K=k_nn)
    pipe_with_pca = Standardizer() |> PCA(variance_ratio=pratio) |> KNNClassifier(K=k_nn)
    pipe_no_pca, pipe_with_pca
end

# ╔═╡ 3b5ef2c4-e0d3-4294-b7ab-195ba452bb8b
begin
    # Validación cruzada estratificada
	Random.seed!(1234)
    cv = CV(nfolds=6, shuffle=true, rng=Random.GLOBAL_RNG)
    # Sin PCA
    machA = machine(pipe_no_pca, Xtbl, yv)
    resA = evaluate!(machA, resampling=cv, measure=accuracy, verbosity=0)
    accA = round(resA.measurement[1], digits=4)
    # Con PCA
    machB = machine(pipe_with_pca, Xtbl, yv)
    resB = evaluate!(machB, resampling=cv, measure=accuracy, verbosity=0)
    accB = round(resB.measurement[1], digits=4)
	nothing
end

# ╔═╡ 51ec6325-3ed2-45f7-860b-29fbeb47ad5d
    md"""
    **Accuracy (KNN, CV=6):**  
    • Sin PCA: **$(accA)**  
    • Con PCA: **$(accB)** 

    > En datasets con muchas variables de ruido y correlaciones, **PCA** suele **mejorar KNN** (y a veces LR/SVM) al des-ruidar y des-correlacionar.
    """

# ╔═╡ 1131220c-660e-448d-b9a3-87d3c09eb7ce
begin
    # Ajustamos un PCA “solo para ver” varianza explicada real
    Xstd = MLJ.transform(machine(Standardizer(), Xtbl) |> fit!, Xtbl)
    pca_only = machine(PCA(), Xstd) |> fit!
    W = report(pca_only).principalvars              # varianzas por componente
    varexp_all = W ./ sum(W)
    p_expl = bar(1:length(varexp_all), varexp_all,
        xlabel="Componente", ylabel="Frac. var. explicada", legend=false,
        title="Scree / Varianza explicada (post estandarización)")
    p_expl
end

# ╔═╡ f88dcefb-caeb-4cc8-a590-00426456a453
begin
    cumexp = cumsum(varexp_all)
    k95 = findfirst(>=(0.95), cumexp)
    k99 = findfirst(>=(0.99), cumexp)
    md"""
    Con estandarización, **k para 95% var** ≈ $(k95), **k para 99%** ≈ $(k99).  
    """
end

# ╔═╡ 664542fc-ed2a-41ce-98c9-065bb4bcb02f
md"## Reconstrucción general y error (multivariado)"

# ╔═╡ 3c575785-d98e-4a0b-8350-2f133564c54d
md"""
Dado $V_k$ (autovectores de las $k$ PCs) y $\mu$, la reconstrucción es  
$\hat X^{(k)} = (X-\mathbf{1}\mu^\top) V_k V_k^\top + \mathbf{1}\mu^\top$.  
El **error de reconstrucción** decae al aumentar $k$ y su traza promedio se relaciona con la suma de autovalores descartados.
"""

# ╔═╡ 22ef786b-2e62-42d5-bb2b-27cb5116e7ec
begin
    # Demostración rápida en los datos (estandarizados, para que escalar no afecte)
    Xstd_mat = MLJ.matrix(Xstd)
    μstd = vec(mean(Xstd_mat, dims=1))
    Σstd = cov(Xstd_mat; corrected=true)
    ev_full = eigen(Σstd)
    idxf = sortperm(ev_full.values, rev=true)
    Λf = ev_full.values[idxf]
    Vf = ev_full.vectors[:, idxf]
    @bind k_recon Slider(1:1:size(Xstd_mat,2), default=min(10,size(Xstd_mat,2)), show_value=true)
end

# ╔═╡ 627ba4df-d7d9-4de8-8b41-56d1cf7a47a7
begin
    Vk = Vf[:, 1:k_recon]
    Xhat = (Xstd_mat .- μstd') * (Vk*Vk') .+ μstd'
    mse_recon = mean(sum(((Xstd_mat - Xhat).^2), dims=2))
    p_rec = plot(1:length(Λf), cumsum(Λf)./sum(Λf), ylabel="Var. explicada acumulada",
                 xlabel="k", label="Cumul var", legend=:bottomright, lw=2)
    vline!(p_rec, [k_recon], label="k actual")
    title!(p_rec, "k=$(k_recon) — MSE recon ≈ $(round(mse_recon, digits=4))")
    p_rec
end

# ╔═╡ 1b92b166-a1b4-4bd1-8802-3ae4f892db93
md"## Caso supervisado: PCA con Fashion-MNIST"

# ╔═╡ 34704bce-2eca-4098-baca-7933396460f0
begin
	n_fash = 3000
    # Cargar split de entrenamiento y submuestrear
    Xraw = FashionMNIST.traintensor(Float32)[:, :, 1:n_fash]  # 28×28×n_fash
    yraw = FashionMNIST.trainlabels()[1:n_fash]               # Vector{UInt8} (0..9)
    # Aplanar a n×784 y escalar a [0,1]
    Xf = reshape(permutedims(Xraw, (3,1,2)), n_fash, 28*28) |> x -> Float64.(x) ./ 255.0
    yf = Int.(yraw)  # 0..9
    size(Xf), length(yf)
end

# ╔═╡ 4230b99e-25aa-40ac-88ad-dcad2ec165ff
plot(Gray.(Xraw[:,:,1]'))

# ╔═╡ 63b47022-2cf9-4e07-a4ee-774b4db1be2d
begin
    # Centrado por pixel
    μpix_f = vec(mean(Xf, dims=1))
    Xc_f = Xf .- μpix_f'
    # PCA por SVD; limitar #componentes para velocidad
	pca_f = machine(PCA(variance_ratio=1.0), Xc_f)
    MLJ.fit!(pca_f)
    Vars_f = report(pca_f).principalvars
    varexp_f = Vars_f ./ sum(Vars_f)
    cumexp_f = cumsum(varexp_f)
    md"**n=$(size(Xf,1))**, **p=784** — **#PCs** = $(length(Vars_f))"
end

# ╔═╡ dd49ef55-3d76-4017-8c15-1b9eb5312b13
begin
    p_scree_f = bar(1:length(varexp_f), varexp_f, legend=false, xlabel="Componente",
                    ylabel="Frac. var. explicada", title="Fashion-MNIST — Scree")
    p_cum_f = plot(1:length(cumexp_f), cumexp_f, lw=2, ylim=(0,1.01), xlabel="k",
                   ylabel="Acumulada", legend=false, title="Varianza acumulada")
    plot(p_scree_f, p_cum_f, layout=(1,2))
end

# ╔═╡ 46c40fa6-be59-4f48-888c-0311eae6d84a
begin
    # Eigen-prendas: columnas de V (loadings) como imágenes
    V_f = fitted_params(pca_f).projection  # 784×r
    kshow_f = min(16, size(V_f,2))
    tiles = []
    for i in 1:kshow_f
        img = reshape(V_f[:,i], 28, 28)
        push!(tiles, heatmap(img', c=:grays, colorbar=false, axis=nothing,
                             aspect_ratio=1, title="PC$(i)"))
    end
    plot(tiles..., layout=(4,4))
end

# ╔═╡ 1081ff61-77b2-4393-a802-7a0964fc7f48
md"""
|Parámetro|Valor|
|---------|-----|
|Cant. componentes|$(@bind k_f Slider(10:10:784, default=40, show_value=true))|
|Elemento a visualizar|$(@bind idx_f Slider(1:1:30, default=40, show_value=true))|
"""

# ╔═╡ 3e30042f-44a9-4609-9d95-492cc1fcd146
begin
    # Vectorizar la imagen (784,)
    x0f = vec(Xf[idx_f, :])               # Vector{Float64}(784)
    xc  = x0f .- μpix_f                   # μpix_f debe ser Vector{Float64}(784)

    # Tomar las primeras k PCs; columnas de V_f son los loadings (784×k)
    Vkf = V_f[:, 1:k_f]                    # 784×k

    # scores (k,), reconstrucción (784,)
    z    = Vkf' * xc                       # k
    xhat = μpix_f .+ Vkf * z               # 784

    p_orig = heatmap(reshape(x0f, 28, 28)', c=:grays, axis=nothing,
                     aspect_ratio=1, title="Original")
    p_rec2 = heatmap(reshape(xhat, 28, 28)', c=:grays, axis=nothing,
                     aspect_ratio=1, title="Recon k=$(k_f)")
    plot(p_orig, p_rec2)
end

# ╔═╡ Cell order:
# ╟─9b9825b6-92f7-11f0-0b64-330dc66d8bc8
# ╟─3d3658cf-7d1f-4ab3-8f22-dcf8dd0fc0bb
# ╠═322e0a56-3e75-4e79-92a1-261071acd38a
# ╠═7f899147-644b-4b58-b2da-f9f9f16a5174
# ╠═7a74fcfd-345e-487e-8fb5-b8cae545e8fc
# ╟─5c39d5c9-9992-4e12-a35d-4613c1dee449
# ╟─349cae87-b4c4-42fa-8f6a-f3693311a91c
# ╟─66d5a755-e1f9-46cd-866d-b3634fbf0409
# ╟─01231e45-4422-4d17-954a-8cff1faea385
# ╟─06a861a6-2330-4d83-95f4-2dee347dc6db
# ╟─3c57a9fa-2558-4b70-906a-881a41abeec8
# ╟─e38cab34-6f39-4ed8-9f63-051fe369dd7b
# ╟─998d8b78-f011-4606-adca-0838a8e25cb8
# ╟─d4decac7-a54c-4dd0-a9ad-c31d27296292
# ╟─85f64ad3-9ac1-426c-b560-48774dbf11b7
# ╟─78e07d2b-5f7a-4fe4-9572-f7923ac91dee
# ╟─6fdc915e-9c83-4452-a8dd-5af0d341e9fd
# ╟─65bc37ab-1726-4dda-b76d-ef83e712f0b5
# ╟─91a80120-7d88-4195-a85f-9e7fb90e7840
# ╟─03bf9016-9d33-4d90-87cb-b755b8172541
# ╟─0d8e9792-5c50-4487-b999-4af95ce4390a
# ╟─9531a831-cdf2-448f-8476-4da01053d952
# ╟─cec72b70-73f6-4440-bcc5-c9191cf08a00
# ╟─3e0305c4-1147-4fcb-8105-04d91ab4d3c8
# ╠═9999248c-7c74-488c-b919-2e6a177a60fc
# ╠═3b5ef2c4-e0d3-4294-b7ab-195ba452bb8b
# ╟─51ec6325-3ed2-45f7-860b-29fbeb47ad5d
# ╟─1131220c-660e-448d-b9a3-87d3c09eb7ce
# ╟─f88dcefb-caeb-4cc8-a590-00426456a453
# ╟─664542fc-ed2a-41ce-98c9-065bb4bcb02f
# ╟─3c575785-d98e-4a0b-8350-2f133564c54d
# ╟─22ef786b-2e62-42d5-bb2b-27cb5116e7ec
# ╟─627ba4df-d7d9-4de8-8b41-56d1cf7a47a7
# ╟─1b92b166-a1b4-4bd1-8802-3ae4f892db93
# ╠═34704bce-2eca-4098-baca-7933396460f0
# ╠═4230b99e-25aa-40ac-88ad-dcad2ec165ff
# ╠═63b47022-2cf9-4e07-a4ee-774b4db1be2d
# ╟─dd49ef55-3d76-4017-8c15-1b9eb5312b13
# ╟─46c40fa6-be59-4f48-888c-0311eae6d84a
# ╟─1081ff61-77b2-4393-a802-7a0964fc7f48
# ╠═3e30042f-44a9-4609-9d95-492cc1fcd146
