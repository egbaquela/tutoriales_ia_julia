### A Pluto.jl notebook ###
# v0.20.18

using Markdown
using InteractiveUtils

# ╔═╡ d76c771b-737d-4777-b839-fb00703e54fc
begin
	using Pkg
	Pkg.activate()

	using PlutoUI
	using LinearAlgebra
	using Random
	using Plots
	
	TableOfContents(title="Contenido")
end

# ╔═╡ b9567cda-a92c-11f0-05b7-e1c658744d65
md"# Diferenciación automática"

# ╔═╡ ba3cf62d-3834-4c82-a0ac-e11e849590fb
md"## Setup"

# ╔═╡ 9e49abab-bbea-479a-8171-5e37513e332f
md"## Forward-mode con números duales"

# ╔═╡ 6e3d4166-7132-4483-a8cf-25cab495d21d
md"### Numeros duales"

# ╔═╡ 9240dda3-b620-46bf-85cf-6dd045ed12df
md"Un número dual es un par $(a,b)$, equivalente a:

$a+bε,con \ ε^2=0, ε \neq 0$

Donde $b$ (la parte dual) es la derivada de $a$ (la parte real).

Si defino correctamente el álgebra de los números duales, el algebra de los duales imita las reglas del cálculo diferencial:

* Suma: $(a+bε)+(c+dε)=(a+c)+(b+d)ε$
* Producto: $(a+bε)(c+dε)=ac+(ad+bc)ε$

Puedo hacer uso de esto para evaluar el número real en una función. Calculando la serie de Taylor en el entorno de $a$:

$f(a+bε)=f(a)+f′(a)(bε)+0.5f′′(a)(bε)^2$

Pero como $(bε)^2=0$, nos queda:

* Parte real: $f(a)$
* Parte dual: $f′(a)(bε)$

Es decir, al evaluar $f$ en un número dual, obtenemos su propia derivada.
"

# ╔═╡ c2649822-1982-4995-9992-2b11bba93f49
md"### Implementación básica"

# ╔═╡ c240b004-b3bb-45b1-9578-9f623f7c4e5d
# --- Tipo Dual (a + bε, con ε^2 = 0, ε != 0 ) ---
struct Dual{T} <: Number
    val::T   # valor primario
    der::T   # derivada (tangente) respecto a la dirección sembrada
end

# ╔═╡ cba8261d-f447-4e45-90ad-24097a9df910
begin
	# Operaciones básicas
	import Base: +, -, *, /, ^, zero, one, abs, sign, inv
	
	zero(x::Dual) = Dual(zero(x.val), zero(x.der))
	one(x::Dual)  = Dual(one(x.val),  zero(x.der))
	
	+(x::Dual, y::Dual) = Dual(x.val + y.val, x.der + y.der)
	+(x::Dual, r::Real) = Dual(x.val + r,     x.der)
	+(r::Real, x::Dual) = x + r
	
	-(x::Dual, y::Dual) = Dual(x.val - y.val, x.der - y.der)
	-(x::Dual, r::Real) = Dual(x.val - r,     x.der)
	-(r::Real, x::Dual) = Dual(r - x.val,    -x.der)
	-(x::Dual)          = Dual(-x.val,       -x.der)
	
	*(x::Dual, y::Dual) = Dual(x.val*y.val, x.val*y.der + x.der*y.val)
	*(x::Dual, r::Real) = Dual(x.val*r,     x.der*r)
	*(r::Real, x::Dual) = x * r
	
	/(x::Dual, y::Dual) = Dual(x.val/y.val, (x.der*y.val - x.val*y.der)/(y.val*y.val))
	/(x::Dual, r::Real) = Dual(x.val/r,     x.der/r)
	/(r::Real, x::Dual) = Dual(r/x.val,    -r*x.der/(x.val*x.val))
	
	^(x::Dual, n::Integer) = begin
	    # d/dx x^n = n*x^(n-1)
	    Dual(x.val^n, n*(x.val^(n-1))*x.der)
	end
	
	inv(x::Dual) = one(x)/x
	abs(x::Dual) = Dual(abs(x.val), x.val >= 0 ? x.der : -x.der)
	sign(x::Dual) = Dual(sign(x.val), zero(x.der))

	nothing
end

# ╔═╡ b897ea33-b60a-4c3c-8b04-8af9c0e9290a
begin
	# Constructores
	dual(x::T, dx::T=zero(T)) where {T} = Dual{T}(x, dx)
	value(x::Dual)   = x.val
	tangent(x::Dual) = x.der
	
	# Promoción / conversión (para operar Dual con Reales)
	Base.promote_rule(::Type{Dual{T}}, ::Type{S}) where {T,S<:Real} = Dual{promote_type(T,S)}
	Base.convert(::Type{Dual{T}}, x::Real) where {T} = Dual{T}(convert(T,x), zero(T))
	
	# Mostrar
	Base.show(io::IO, x::Dual) = print(io, "Dual(", x.val, " ± ", x.der, "ε)")
end

# ╔═╡ 3c0b0d77-c48c-41e3-8b31-8d5172e9e98f
begin
	# Funciones elementales
	import Base: exp, log, sqrt, sin, cos, tan, sinh, cosh, tanh
	
	exp(x::Dual)  = (e = exp(x.val); Dual(e, e*x.der))
	log(x::Dual)  = Dual(log(x.val), x.der/x.val)
	sqrt(x::Dual) = Dual(sqrt(x.val), x.der/(2sqrt(x.val)))
	
	sin(x::Dual)  = Dual(sin(x.val),  cos(x.val)*x.der)
	cos(x::Dual)  = Dual(cos(x.val), -sin(x.val)*x.der)
	tan(x::Dual)  = Dual(tan(x.val),  x.der/(cos(x.val)^2))
	
	sinh(x::Dual) = Dual(sinh(x.val), cosh(x.val)*x.der)
	cosh(x::Dual) = Dual(cosh(x.val), sinh(x.val)*x.der)
	tanh(x::Dual) = Dual(tanh(x.val), (1 - tanh(x.val)^2)*x.der)

	nothing
end

# ╔═╡ 6d9f7a48-63cd-4325-82e5-2dddd5d4acd9
begin
	# Activaciones útiles
	relu(x::Dual) = Dual(max(x.val, zero(x.val)), x.val > 0 ? x.der : zero(x.der))
	σ(x) = one(x)/(one(x) + exp(-x))  # funciona para Dual y Real
	nothing
end

# ╔═╡ c8259c0c-0391-45b9-8540-ea33a0174354
md"### Probando nuestro forward autodiff"

# ╔═╡ 682942e8-3545-465f-bef1-85aabdb7499d
function multiplicacion(x::Number,n::Int)
    if n==0 return 0 end

	resultado = x
	for i in 2:n
       resultado=resultado+x
    end
    
	return resultado
end


# ╔═╡ 70f349a6-785b-4f32-8984-3c8e79d35cb5
function potencia(x::Number,n::Int)
    if n==0 return 1 end

	resultado = x
	for i in 2:n
       resultado=resultado*x
    end
    
	return resultado
end

# ╔═╡ a2e4a9cb-5099-4083-8d01-e96b73e4a28d
function raiz_babilonica(n::Number, max_iteraciones::Int = 100)
	# La estimación en la iteración (i+1) es x_{i+1} = (x_{i} + n/x_{i})/2

    # Estimación inicial (puede ser n/2 o cualquier otra aproximación)
    x = n / 2 
    
    for _ in 1:max_iteraciones
        x_nuevo = (x + n / x) / 2
        x = x_nuevo
    end
    
    return x
end


# ╔═╡ 53cd202e-4b27-483e-b221-5a972dac0a0a
begin
	x=5
	my_dual = dual(x,1)
end

# ╔═╡ Cell order:
# ╟─b9567cda-a92c-11f0-05b7-e1c658744d65
# ╟─ba3cf62d-3834-4c82-a0ac-e11e849590fb
# ╠═d76c771b-737d-4777-b839-fb00703e54fc
# ╟─9e49abab-bbea-479a-8171-5e37513e332f
# ╟─6e3d4166-7132-4483-a8cf-25cab495d21d
# ╟─9240dda3-b620-46bf-85cf-6dd045ed12df
# ╟─c2649822-1982-4995-9992-2b11bba93f49
# ╠═c240b004-b3bb-45b1-9578-9f623f7c4e5d
# ╠═b897ea33-b60a-4c3c-8b04-8af9c0e9290a
# ╠═cba8261d-f447-4e45-90ad-24097a9df910
# ╠═3c0b0d77-c48c-41e3-8b31-8d5172e9e98f
# ╠═6d9f7a48-63cd-4325-82e5-2dddd5d4acd9
# ╟─c8259c0c-0391-45b9-8540-ea33a0174354
# ╠═682942e8-3545-465f-bef1-85aabdb7499d
# ╠═70f349a6-785b-4f32-8984-3c8e79d35cb5
# ╠═a2e4a9cb-5099-4083-8d01-e96b73e4a28d
# ╠═53cd202e-4b27-483e-b221-5a972dac0a0a
