using Flux, DiffEqFlux, DifferentialEquations,Random, Plots,LinearAlgebra

u0 = Float64[2.; 0.]
datasize = 30
tspan = (0.0,1.5)


function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end
t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(trueODEfunc,u0,tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))

Random.seed!(1234);

# glorot_uniform weight and zeros bias initialization
dense1 = Dense(2,50,tanh)
dense2 = Dense(50,2)

dudt = Chain(x -> x.^3,
             dense1,
             dense2)

ps = Flux.params(dudt)

struct RK4Step
end

stepper = RK4Step()
h = (tspan[2]-tspan[1])/datasize

function step(stepper::RK4Step,F,u,tk,tkp1)
    h    = tkp1 - tk

    du1  = F(u)
    du2  = F(u+(h/2)*du1)
    du3  = F(u+(h/2)*du2)
    du4  = F(u+h*du3)

    u  += (h/6) * (du1 + 2 .* du2 + 2 .* du3 + du4)
    return u
end


function predict_rk4()
    un = copy(u0)
    pred = zeros(2,datasize)
    tk = tspan[1]
    pred = un
    for k=1:datasize-1
        un = step(stepper,dudt,un,tk,tk+h)
        pred = [pred un]
        tk+=h
    end
    return pred
end

pred = predict_rk4() # Get the prediction using the correct initial condition
scatter(t,ode_data[1,:],label="data")
scatter!(t,Flux.data(pred[1,:]),label="prediction")

loss_n_ode() = sum(abs2,ode_data .- predict_rk4())

J0,back = Tracker.forward(() -> loss_n_ode(), ps)
gc = back(1)
dJ = []
for p in ps
    println(size(gc[p]))
    gp = gc[p]
    push!(dJ,gp)
end

J0 = loss_n_ode()
W10  = copy(dudt[2].W.data)
b10  = copy(dudt[2].b.data)
W20  = copy(dudt[3].W.data)
b20  = copy(dudt[3].b.data)

dW1 = randn(Float32,size(W10))
db1 = randn(Float32,size(b10))
dW2 = randn(Float32,size(W20))
db2 = randn(Float32,size(b20))

dJdW1 = dot(dJ[1],dW1)
dJdb1 = dot(dJ[2],db1)
dJdW2 = dot(dJ[3],dW2)
dJdb2 = dot(dJ[4],db2)
dJdth = dJdW1 + dJdb1 + dJdW2 + dJdb2
his = zeros(20,3)
for k=1:20
    h = 2.0f0^(-k)

    dudt[2].W.data .= W10 + h*dW1
    dudt[2].b.data .= b10 + h*db1
    dudt[3].W.data .= W20 + h*dW2
    dudt[3].b.data .=  b20 + h.*db2

    Jt = loss_n_ode()
    err0 = abs(Jt.data-J0.data)
    err1 = abs(Jt.data-J0.data-h*dJdth)
    his[k,:] = [h err0 err1.data]

    println("h=$(h)\terr0=$(err0)\terr1=$(err1)")
end

for k=1:20
    println("$(his[k,1])\t$(his[k,3])\\\\")
end


using Plots
p = plot(his[:,1],his[:,2],linewidth=5,title="Derivative Check RK4",
     xscale=:log10,yscale=:log10,
     xaxis="step size (h)",yaxis="Error",label="E0")
plot!(p, his[:,1],his[:,3],linewidth=5,title="Derivative Check RK4",
      label="E1",legend=:topleft)


savefig(p,"image/derivCheckRK4.pdf")
