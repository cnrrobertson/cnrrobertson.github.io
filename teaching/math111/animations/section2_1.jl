cd(@__DIR__)
using Pkg
Pkg.activate(".")

using CairoMakie

# Fruit fly example Figure 2.6 logistic curve
f(x) = 350 / (1 + exp(-.18 * (x - 25)))
xs = range(0,50,length=100)
fs = f.(xs)

# Animation
fig = Figure(resolution=(600,500))
ax = Axis(fig[1,1])
ax.ylabel = "Number of flies"
ax.xlabel = "Time (days)"
ax.xlabelsize = 20
ax.ylabelsize = 20
ax.xticksize=18
ax.yticksize=18
ax.limits = (-1, 51, -1, 351)

lines!(ax,xs,fs)

# Moving secant
start_h = 22.0
h = Observable(start_h)
t1 = 23.0
f1 = f(23.0)
point_slope(x,s) = s*(x - t1) + f1

sec_ts = @lift([t1, t1 + $h])
sec_fs = @lift([f1, f(t1 + $h)])
slope = @lift(($sec_fs[2] - $sec_fs[1]) / ($sec_ts[2] - $sec_ts[1]))
tan_ts = @lift([(-f1/$slope) + t1, (350-f1/$slope) + t1])
tan_fs = @lift([point_slope($tan_ts[1],$slope), point_slope($tan_ts[2],$slope)])
lines!(ax,tan_ts,tan_fs,color=:red)
scatter!(ax,sec_ts,sec_fs)

steps = 100
step_path = vcat(1:steps-1, fill(steps-1,steps÷2), steps-1:-1:1)
record(fig, "section2_1.gif", step_path) do i
  h[] = start_h*(steps-i)/steps
end
