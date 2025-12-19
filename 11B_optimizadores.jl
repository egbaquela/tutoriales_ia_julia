### A Pluto.jl notebook ###
# v0.20.18

using Markdown
using InteractiveUtils

# ╔═╡ 522998c1-9a4b-45b9-90ea-50a21a04f27f
begin
	using Pkg
	Pkg.activate()

	using PlutoUI
	using LinearAlgebra
	using Random
	using Plots
	
	TableOfContents(title="Contenido")
end

# ╔═╡ 9ee2f0e6-a929-11f0-2931-e7a30c29d077
md"# Optimizadores"

# ╔═╡ 526bd944-0b90-40f3-bfd1-0632f645ac39
md"## Setup"

# ╔═╡ b583a461-caed-4968-a6cc-5870f20bdc13
md"## Minibatch"

# ╔═╡ d194ef30-6f15-4751-b361-cbe06ba23d87
md"""
- Parámetros: \( \theta_t \in \mathbb{R}^n \)
- Mini-batch en el paso \(t\): \(B_t\)
- Gradiente promedio del mini-batch:
$g_t \;=\; \nabla_\theta \Bigg(\frac{1}{|B_t|}\sum_{i\in B_t}\ell\big(f_\theta(x_i),y_i\big)\Bigg)\Bigg|_{\theta=\theta_t}$

"""

# ╔═╡ ca6c11b9-1f67-41be-bf94-a0a14df7f710
md"# Gradiente por descenso estocástico (SGD)"

# ╔═╡ d8d45be2-7c5e-4e48-abdf-bb0b55b437d6
md"""
$\theta_{t+1} \;=\; \theta_t \;-\; \eta_t\, g_t$


* Pros: simple, robusto, buena generalización con buen schedule.
* Contras: alta varianza, sensible a escalas de los parámetros.
"""

# ╔═╡ 7c53a3c7-79e4-4529-a92e-fc7b296677c2
md"## Momentum"

# ╔═╡ 4e09b6a6-e375-4b67-ab7c-a9f7b4debe58
md"""
$\begin{aligned}
v_t &= \beta\, v_{t-1} \;+\; g_t \\
\theta_{t+1} &= \theta_t \;-\; \eta_t\, v_t
\end{aligned}$

Hiperparámetro: $\beta \in [0.8, 0.99]$.

* Pros respecto a SGD: Suaviza el ruido estocástico y acelera en valles alargados (porque hace un promedio móvil)
"""

# ╔═╡ a391e69e-d439-4788-9aad-dfb1c13f9a35
md"## ADAM"

# ╔═╡ b72d692b-8c5d-49c2-adb2-fbeb79e396dd
md"""
$\begin{aligned}
m_t &= \beta_1\, m_{t-1} \;+\; (1-\beta_1)\, g_t \\
v_t &= \beta_2\, v_{t-1} \;+\; (1-\beta_2)\, g_t^{\odot 2} \\
\hat m_t &= \frac{m_t}{1-\beta_1^{\,t}}, \qquad
\hat v_t = \frac{v_t}{1-\beta_2^{\,t}} \\
\theta_{t+1} &= \theta_t \;-\; \alpha \;\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}
\end{aligned}$

Típicos: $\beta_1=0.9,\; \beta_2=0.999,\; \varepsilon=10^{-8}$.

* Pros respecto a momentum y SGD: Escalado por coordenada (como RMSProp) + promedio móvil del gradiente (como Momentum) + corrección de sesgo.
"""

# ╔═╡ 0ac4a8bb-9fba-410e-893c-191e84b148ec
md"## Un ejemplo"

# ╔═╡ d0ec111c-2282-4a3b-8bf9-bc962125d7d3
md"problema y función para graficación:"

# ╔═╡ 1a1255b4-c843-4ba6-8cf1-0c6a2776d269
begin
	# ----------------------------
	# 1) Problema: Rosenbrock 2D
	# ----------------------------
	rosen(x::AbstractVector{<:Real}) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
	
	function grad_rosen(x::AbstractVector{<:Real})
	    x1, x2 = x[1], x[2]
	    gx1 = 2*(x1 - 1) - 400*x1*(x2 - x1^2)
	    gx2 = 200*(x2 - x1^2)
	    return [gx1, gx2]
	end
	
	# ---------------------------------------
	# 2) Utilidades de trayectorias y parada
	# ---------------------------------------
	mutable struct Trajectory
	    xs::Vector{Vector{Float64}}   # puntos (x) visitados
	    fs::Vector{Float64}           # valores de f(x)
	    gnorms::Vector{Float64}       # normas de gradiente
	end
	Trajectory() = Trajectory(Vector{Vector{Float64}}(), Float64[], Float64[])
	
	function record!(tr::Trajectory, x::Vector{Float64}, f::Float64, g::Vector{Float64})
	    push!(tr.xs, copy(x))
	    push!(tr.fs, f)
	    push!(tr.gnorms, norm(g))
	end

	nothing
end

# ╔═╡ 37fbe201-e902-4cd3-985a-e70c28d1352e
function optimize_sgd(f, g!, x0::AbstractVector{<:Real};
                      η::Float64=1e-3, iters::Int=5000, tol::Float64=1e-8, seed::Int=0)
    Random.seed!(seed)
    x = collect(float.(x0))
    tr = Trajectory()
    for t in 1:iters
        g = g!(x)
        fx = f(x)
        record!(tr, x, fx, g)
        if norm(g) < tol
            break
        end
        @. x = x - η * g
    end
    return x, tr
end

# ╔═╡ b80e3507-4810-459a-ad67-f96ba2ed30b1
function optimize_momentum(f, g!, x0::AbstractVector{<:Real};
                           η::Float64=1e-3, β::Float64=0.9, iters::Int=5000, tol::Float64=1e-8, seed::Int=0)
    Random.seed!(seed)
    x = collect(float.(x0))
    v = zeros(length(x))
    tr = Trajectory()
    for t in 1:iters
        g = g!(x)
        fx = f(x)
        record!(tr, x, fx, g)
        if norm(g) < tol
            break
        end
        @. v = β*v + g
        @. x = x - η*v
    end
    return x, tr
end

# ╔═╡ 1a909a93-52bf-4098-a7c1-b0d950f16dce
function optimize_adam(f, g!, x0::AbstractVector{<:Real};
                       α::Float64=1e-2, β1::Float64=0.9, β2::Float64=0.999, ϵ::Float64=1e-8,
                       weight_decay::Float64=0.0, iters::Int=4000, tol::Float64=1e-8, seed::Int=0)
    Random.seed!(seed)
    x = collect(float.(x0))
    m = zeros(length(x))
    v = zeros(length(x))
    tr = Trajectory()
    t = 0
    for k in 1:iters
        t += 1
        g = g!(x)
        fx = f(x)
        record!(tr, x, fx, g)
        if norm(g) < tol
            break
        end
        @. m = β1*m + (1 - β1)*g
        @. v = β2*v + (1 - β2)*(g*g)
        m̂ = m ./ (1 - β1^t)
        v̂ = v ./ (1 - β2^t)
        # AdamW (decoupled weight decay)
        @. x = (1 - α*weight_decay)*x - α * (m̂ / (sqrt(v̂) + ϵ))
    end
    return x, tr
end

# ╔═╡ 0b633ae4-0863-469c-b60e-d2a2844792f2
begin
	# Punto inicial clásico para Rosenbrock:
	x0 = [-1.2, 1.0]
	
	# Ejecutar los tres métodos
	x_sgd,  tr_sgd  = optimize_sgd(rosen, grad_rosen, x0; η=1e-3, iters=6000)
	x_mom,  tr_mom  = optimize_momentum(rosen, grad_rosen, x0; η=1e-3, β=0.9, iters=6000)
	x_adam, tr_adam = optimize_adam(rosen, grad_rosen, x0; α=1e-2, β1=0.9, β2=0.999, iters=4000, weight_decay=0.0)
	
	@info "Resultados:"
	@info "SGD   → x*=$(x_sgd),  f*=$(rosen(x_sgd))"
	@info "Mom   → x*=$(x_mom),  f*=$(rosen(x_mom))"
	@info "Adam  → x*=$(x_adam), f*=$(rosen(x_adam))"
end

# ╔═╡ 348c0d51-ac6c-40ae-a239-063dc0ebe065
begin
	# ---- 4.1 Curvas de pérdida (log-escala) ----
	plt1 = plot(title="Convergencia de la función (log)",
	            xlabel="Iteración", ylabel="f(x)", yscale=:log10, legend=:topright)
	plot!(plt1, tr_sgd.fs,  label="SGD",   lw=2)
	plot!(plt1, tr_mom.fs,  label="Momentum", lw=2)
	plot!(plt1, tr_adam.fs, label="Adam",  lw=2)
	
	# ---- 4.2 Trayectorias sobre contornos ----
	# Mapa de contornos de Rosenbrock
	xr = range(-2.0, 2.0, length=350)
	yr = range(-1.0, 3.0, length=350)
	Z = [rosen([x,y]) for y in yr, x in xr]  # filas: y, columnas: x
	
	function coords(tr::Trajectory)
	    xs = first.(tr.xs)
	    ys = last.(tr.xs)
	    return xs, ys
	end
	
	xs_sgd, ys_sgd   = coords(tr_sgd)
	xs_mom, ys_mom   = coords(tr_mom)
	xs_adam, ys_adam = coords(tr_adam)
	
	plt2 = contour(xr, yr, Z, levels=40, linewidth=1.0, title="Trayectoria en Rosenbrock",
	               xlabel="x₁", ylabel="x₂", legend=:topleft)
	plot!(plt2, xs_sgd,  ys_sgd,  label="SGD",     lw=2, marker=:circle, ms=2)
	plot!(plt2, xs_mom,  ys_mom,  label="Momentum", lw=2, marker=:utriangle, ms=2)
	plot!(plt2, xs_adam, ys_adam, label="Adam",     lw=2, marker=:diamond, ms=2)
	
	# marcar el mínimo (1,1)
	scatter!(plt2, [1.0], [1.0], label="Mínimo", marker=:star5, ms=8)
	
	display(plt1)
	display(plt2)
	
	# (Opcional) imprimir algunas métricas finales
	println("\nResumen final:")
	println("SGD:    f*=$(round(last(tr_sgd.fs), digits=6)),   ||g||=$(round(last(tr_sgd.gnorms), digits=6)), iters=$(length(tr_sgd.fs))")
	println("Mom:    f*=$(round(last(tr_mom.fs), digits=6)),   ||g||=$(round(last(tr_mom.gnorms), digits=6)), iters=$(length(tr_mom.fs))")
	println("Adam:   f*=$(round(last(tr_adam.fs), digits=6)),  ||g||=$(round(last(tr_adam.gnorms), digits=6)), iters=$(length(tr_adam.fs))")

end

# ╔═╡ 1c23d488-c996-4dd2-a07c-23c6fd90aa56
plt1

# ╔═╡ 3e8bc118-669d-4fc3-b9fb-184b589cdc0a
plt2

# ╔═╡ Cell order:
# ╟─9ee2f0e6-a929-11f0-2931-e7a30c29d077
# ╟─526bd944-0b90-40f3-bfd1-0632f645ac39
# ╠═522998c1-9a4b-45b9-90ea-50a21a04f27f
# ╟─b583a461-caed-4968-a6cc-5870f20bdc13
# ╟─d194ef30-6f15-4751-b361-cbe06ba23d87
# ╟─ca6c11b9-1f67-41be-bf94-a0a14df7f710
# ╟─d8d45be2-7c5e-4e48-abdf-bb0b55b437d6
# ╟─7c53a3c7-79e4-4529-a92e-fc7b296677c2
# ╟─4e09b6a6-e375-4b67-ab7c-a9f7b4debe58
# ╟─a391e69e-d439-4788-9aad-dfb1c13f9a35
# ╟─b72d692b-8c5d-49c2-adb2-fbeb79e396dd
# ╟─0ac4a8bb-9fba-410e-893c-191e84b148ec
# ╟─d0ec111c-2282-4a3b-8bf9-bc962125d7d3
# ╠═1a1255b4-c843-4ba6-8cf1-0c6a2776d269
# ╠═37fbe201-e902-4cd3-985a-e70c28d1352e
# ╠═b80e3507-4810-459a-ad67-f96ba2ed30b1
# ╠═1a909a93-52bf-4098-a7c1-b0d950f16dce
# ╠═0b633ae4-0863-469c-b60e-d2a2844792f2
# ╠═348c0d51-ac6c-40ae-a239-063dc0ebe065
# ╠═1c23d488-c996-4dd2-a07c-23c6fd90aa56
# ╠═3e8bc118-669d-4fc3-b9fb-184b589cdc0a
