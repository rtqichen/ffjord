using Flux, DiffEqFlux, DifferentialEquations, Plots,LinearAlgebra,Random,
Printf, LaTeXStrings

u0 = Float64[2.; 0.] # optimize TODO: make @SVector static array
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
# Random normal initialization
# W10  = param(randn(50,2))
# b10  = param(randn(50))
# W20  = param(randn(2,50))
# b20  = param(randn(2))

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


tpode = []
function predict_rk4()
    tt = @elapsed begin
    un = copy(u0)
    pred = zeros(2,datasize)
    tk = tspan[1]
    pred = un
    for k=1:datasize-1
        un = step(stepper,dudt,un,tk,tk+h)
        pred = [pred un]
        tk+=h
    end
    end
    push!(tpode,tt)
    return pred
end

pred = predict_rk4() # Get the prediction using the correct initial condition
scatter(t,ode_data[1,:],label="data")
scatter!(t,Flux.data(pred[1,:]),label="prediction")

loss_n_ode() = sum(abs2,ode_data .- predict_rk4())


data = Iterators.repeated((), 300)
opt = ADAM(0.1)
his = []
times = []
global bestLoss = 1000000
global pl

cb = function () # callback function to observe training
    global bestLoss, pl
    lc = loss_n_ode().data;
    push!(his,lc) # append
    push!(times,time())


    sTime = @sprintf("%5.2f", (times[end]-times[max(1,end-1)]) )

    if lc < bestLoss
        bestLoss = lc

        # plot current prediction against data
        cur_pred = Flux.data(predict_n_ode())
        pl = scatter(t,ode_data[1,:],label="data",
            xlabel=L"t",ylabel=L"u_1",legend=:bottomright, margin=5Plots.mm)
        scatter!(pl,t,cur_pred[1,:],label="prediction")
    end

    sLoss = @sprintf("%7.3f", bestLoss)
    str = "Disc-Opt.  iter=$(length(his)-1)  timeIter=" * sTime *
          " sec. \n Best Iterate, with loss=" * sLoss

    println(str)
    title!(str)
    display(plot(pl))


    if length(his) % 50 == 1 # just save every 10 iterations
        fn = "image/tmp/rk4-iter-$(length(his)-1).pdf"
        savefig(pl,fn)
    end
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_n_ode, ps, data, opt, cb = cb)


using JLD # TODO: fix conflict warning
save("DiscOptResults.jld","his",his,"times", times)

hisOD = load("OptDiscResults.jld","his")
timesOD = load("OptDiscResults.jld","times")

lossPlot = plot(his, linewidth=1, linecolor=:blue,
                ylabel="Loss", xlabel="Iteration", label="", legend=:topright) # actual
plot!(lossPlot, accumulate(min, his), linewidth=4, linecolor=:blue, label="Disc-Opt. loss") # accumulated
plot!(lossPlot, hisOD, linewidth=1, linecolor=:red,label="") # actual
plot!(lossPlot, accumulate(min, hisOD), linewidth=4, linecolor=:red, label="Opt-Disc. loss") # accumulated

savefig(lossPlot,"image/lossPlot.pdf")

timePlot = plot(diff(times), linewidth=4, linecolor=:blue,
                ylabel="Time", xlabel="Iteration", label="Disc-Opt. Timing")
plot!(timePlot, diff(timesOD), linewidth=4, linecolor=:red, label="Opt-Disc. Timing")

savefig(timePlot,"image/timePlot.pdf")
