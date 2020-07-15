using Flux, DiffEqFlux, DifferentialEquations, Plots, Random, LaTeXStrings, Printf
# from https://github.com/FluxML/model-zoo/blob/da4156b4a9fb0d5907dcb6e21d0e78c72b6122e0/other/diffeq/neural_ode.jl
# changes made for reducability and ability to compare with other approaches

u0 = [2.; 0.]
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
tpode = []
reltol = 1e-7;
abstol = 1e-9;
function n_ode(x)
    tt = @elapsed begin
        pred = neural_ode(dudt,x,tspan,Tsit5(),saveat=t,reltol=reltol,abstol=abstol)
    end
    push!(tpode,tt)
    return pred
end

pred = n_ode(u0) # Get the prediction using the correct initial condition
scatter(t,ode_data[1,:],label="data")
scatter!(t,Flux.data(pred[1,:]),label="prediction")

function predict_n_ode()
  n_ode(u0)
end
loss_n_ode() = sum(abs2,ode_data .- predict_n_ode())

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
    str = "Opt-Disc.  iter=$(length(his)-1)  timeIter=" * sTime *
          " sec. \n Best Iterate, with loss=" * sLoss

    println(str)
    title!(str)
    display(plot(pl))


    if length(his) % 50 == 1 # just save every 10 iterations
        fn = "image/tmp/NODE-f64-reltol-1e-7-iter-$(length(his)-1).pdf"
        savefig(pl,fn)
    end
end

# Display the ODE with the initial parameter values.
cb()
Flux.train!(loss_n_ode, ps, data, opt, cb = cb)

using JLD
save("OptDiscResults.jld","his",his,"times", times) # save results for a joint graph


# lossPlot = plot(his, linewidth=1, linecolor=:red,
#                 ylabel="Loss", xlabel="Time", label="", legend=:topright)
# plot!(lossPlot, accumulate(min, his), linewidth=4, linecolor=:red, label="Opt-Disc. loss")
#
# savefig(lossPlot,"lossPlotOD.pdf")
#
#
# timePlot = plot(diff(times), linewidth=4, linecolor=:red,
#                 ylabel="Time", xlabel="Iteration", label="Opt-Disc Timing")
# savefig(timePlot,"timePlotOD.pdf")
