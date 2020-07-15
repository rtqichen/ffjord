# compareBoth.jl
#
using Flux
using DiffEqFlux
using DifferentialEquations
using Plots
using LinearAlgebra
using Random
using Printf
using LaTeXStrings

## functions
# shared
#--------
function initDUDT(seed)
    Random.seed!(seed);
    # glorot_uniform weight and zeros bias initialization
    dense1 = Dense(2,50,tanh)
    dense2 = Dense(50,2)
    dudt = Chain(x -> x.^3,
                 dense1,
                 dense2)
end

function trueODEfunc(du,u,p,t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u.^3)'true_A)'
end



## setup
u0 = Float64[2.; 0.]
datasize = 30
tspan = (0.0,1.5)

t = range(tspan[1],tspan[2],length=datasize)
prob = ODEProblem(trueODEfunc,u0,tspan)
ode_data = Array(solve(prob,Tsit5(),saveat=t))

#---------------
seed = 1234
numIterations = 500
doDerivCheck=false
#---------------

dudt = initDUDT(seed)
ps = Flux.params(dudt)

# OD #

tpode = []
reltol = 1e-7;
abstol = 1e-9;
function n_ode(x,model=dudt)
    tt = @elapsed begin
        pred = neural_ode(model,x,tspan,Tsit5(),saveat=t,reltol=reltol,abstol=abstol) # neural_ode uses diffeq_adjoint
    end
    push!(tpode,tt)
    return pred
end

pred = n_ode(u0) # Get the prediction using the correct initial condition
scatter(t,ode_data[1,:],label="data")
scatter!(t,Flux.data(pred[1,:]),label="prediction")

function predict_n_ode(model=dudt)
  n_ode(u0,model)
end
loss_n_ode() = sum(abs2,ode_data .- predict_n_ode(dudt))


data = Iterators.repeated((), numIterations)
opt = ADAM(0.1)
his = []
times = []
global bestLoss = 1000000
global pl

cbOD = function () # callback function to observe/ training
    global bestLoss, pl
    start = time()
    lc = loss_n_ode().data;
    push!(times,time()-start)
    push!(his,lc) # append

    sTime = @sprintf("%.2e", times[end] )

    if lc < bestLoss
        bestLoss = lc

        # plot current prediction against data
        cur_pred = Flux.data(predict_n_ode())
        pl = scatter(t,ode_data[1,:],label="data", # titlefont=fnt,
            xlabel=L"\mathbf{t}",ylabel=L"\mathbf{u_1}",legend=:bottomright, margin=5Plots.mm)
        scatter!(pl,t,cur_pred[1,:],label="prediction")
    end

    sLoss = @sprintf("%7.3f", bestLoss)
    str = "Opt-Disc.  iter=$(length(his)-1)  timeIter=" * sTime *
          " sec. \n Best Iterate, with loss=" * sLoss

    println(str)
    title!(str)
    display(plot(pl))

    fn = "image/tmp/seed$(seed)-NODE-f64-reltol-1e-7-iter-$(length(his)-1).png"
    savefig(pl,fn)

    if length(his) % 50 == 1 # make a nice pdf every 50 iterations
        fn = "image/tmp/seed$(seed)-NODE-f64-reltol-1e-7-iter-$(length(his)-1).pdf"
        savefig(pl,fn)
    end

    # plot the zero-th and first-order derivative differences
    if doDerivCheck

        # deepcopy the model
        weight1  = param(copy(dudt[2].W.data))
        bias1    = param(copy(dudt[2].b.data))
        weight2  = param(copy(dudt[3].W.data))
        bias2    = param(copy(dudt[3].b.data))

        dudtCopy = Chain(x -> x.^3,
             Dense(weight1,bias1,tanh),
             Dense(weight2,bias2))

        psCopy = Flux.params(dudtCopy)

        J0,backp = Tracker.forward(() -> sum(abs2,ode_data .- predict_n_ode(dudtCopy)), psCopy)
        gradInfo = backp(1)
        dJ = []
        for p in psCopy
            # println(size(gradInfo[p]))
            g = gradInfo[p]
            push!(dJ,g)
        end

        Jac = Flux.jacobian(x -> dudtCopy(x), rand(2)) # u0 or rand(2)?
        # eye = Matrix{Float64}(I,2,2) # 2x2 identity
        # eye = Diagonal(ones(2,2))
        eye = [0 1; 1 0]
        eigVals = eigen(Jac.data,eye).values # need the eye to get the imaginary part
        println(eigVals)
        if eigVals[1] isa Float64
            sEigs = @sprintf("%5.3f + 0i , %5.3f + 0i", eigVals[1], eigVals[2])
        else # if complex
            sEigs = @sprintf("%5.3f + %5.3fi , %5.3f + %5.3fi", eigVals[1].re, eigVals[1].im,
                            eigVals[2].re, eigVals[2].im)
        end

        J0 = sum(abs2,ode_data .- predict_n_ode(dudtCopy))
        W10  = copy(dudtCopy[2].W.data)
        b10  = copy(dudtCopy[2].b.data)
        W20  = copy(dudtCopy[3].W.data)
        b20  = copy(dudtCopy[3].b.data)

        dW1 = randn(Float32,size(W10))
        db1 = randn(Float32,size(b10))
        dW2 = randn(Float32,size(W20))
        db2 = randn(Float32,size(b20))

        dJdW1 = dot(dJ[1],dW1)
        dJdb1 = dot(dJ[2],db1)
        dJdW2 = dot(dJ[3],dW2)
        dJdb2 = dot(dJ[4],db2)
        dJdth = dJdW1 + dJdb1 + dJdW2 + dJdb2
        hisErr = zeros(20,3)
        for k=1:20
            h = 2.0f0^(-k)

            dudtCopy[2].W.data .= W10 + h*dW1
            dudtCopy[2].b.data .= b10 + h*db1
            dudtCopy[3].W.data .= W20 + h*dW2
            dudtCopy[3].b.data .= b20 + h*db2

            Jt = sum(abs2,ode_data .- predict_n_ode(dudtCopy))
            err0 = abs(Jt.data-J0.data)
            err1 = abs(Jt.data-J0.data-h*dJdth)
            if dJ isa TrackedArray
                hisErr[k,:] = [h err0 err1.data]
            else # if Float64
                hisErr[k,:] = [h err0 err1]
            end
        end

        p = plot(hisErr[:,1],hisErr[:,2],linewidth=5,
            title="Opt.-Disc. Derivative Check iter=$(length(his)-1)\neigs: " * sEigs,
            xscale=:log10,yscale=:log10,
            xaxis="Step Size (h)",yaxis="Error",label="E0")
        plot!(p, hisErr[:,1],hisErr[:,3],linewidth=5,
              label="E1",legend=:topleft)

        savefig(p,"image/tmp/seed$(seed)-derivCheckNODE-iter-$(length(his)-1).pdf")

    end
end

# Display the ODE with the initial parameter values.
cbOD()

# precompile up to here, then run the next line by itself
Flux.train!(loss_n_ode, ps, data, opt, cb = cbOD)

multT = 4# multiply time end by this multiplier
tspanlong = (0.0,multT*1.5)
tlong = range(tspanlong[1],tspanlong[2],length=multT*datasize)
extra_OD = neural_ode(dudt,u0,tspanlong,Tsit5(),saveat=tlong,reltol=reltol,abstol=abstol)

#----------------------------------


# now do the discretize-Optimize approach

dudt = initDUDT(seed)
ps = Flux.params(dudt)
# Discretize-Optimize #


struct RK4Step
end

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
function predict_rk4(model=dudt)
    tt = @elapsed begin
    un = copy(u0)
    pred = zeros(2,datasize)
    tk = tspan[1]
    pred = un
    for k=1:datasize-1
        un = step(stepper,model,un,tk,tk+h)
        pred = [pred un]
        tk+=h
    end
    end
    push!(tpode,tt)
    return pred
end

stepper = RK4Step()
h = (tspan[2]-tspan[1])/datasize
pred = predict_rk4() # Get the prediction using the correct initial condition
scatter(t,ode_data[1,:],label="data")
scatter!(t,Flux.data(pred[1,:]),label="prediction")

loss_rk() = sum(abs2,ode_data .- predict_rk4(dudt))


# data = Iterators.repeated((), 5)
# opt = ADAM(0.1)
hisRK = []
timesRK = []
bestLoss = 1000000



cbRK = function () # callback function to observe/ training
    global bestLoss, pl
    start = time()
    lc = loss_rk().data;
    push!(timesRK,time()-start)
    push!(hisRK,lc) # append

    sTime = @sprintf("%.2e", timesRK[end] )


    if lc < bestLoss
        bestLoss = lc

        # plot current prediction against data
        cur_pred = Flux.data(predict_rk4())
        pl = scatter(t,ode_data[1,:],label="data",
            xlabel=L"\mathbf{t}",ylabel=L"\mathbf{u_1}",legend=:bottomright, margin=5Plots.mm)
        scatter!(pl,t,cur_pred[1,:],label="prediction")
    end

    sLoss = @sprintf("%7.3f", bestLoss)
    str = "Disc-Opt.  iter=$(length(hisRK)-1)  timeIter=" * sTime *
          " sec. \n Best Iterate, with loss=" * sLoss

    println(str)
    title!(str)
    display(plot(pl))

    fn = "image/tmp/seed$(seed)-rk4-iter-$(length(hisRK)-1).png"
    savefig(pl,fn)

    if length(hisRK) % 50 == 1 # save a nice pdf every 50 iterations
        fn = "image/tmp/seed$(seed)-rk4-iter-$(length(hisRK)-1).pdf"
        savefig(pl,fn)
    end

    if doDerivCheck

        # deepcopy the model
        weight1  = param(copy(dudt[2].W.data))
        bias1    = param(copy(dudt[2].b.data))
        weight2  = param(copy(dudt[3].W.data))
        bias2    = param(copy(dudt[3].b.data))

        dudtCopy = Chain(x -> x.^3,
             Dense(weight1,bias1,tanh),
             Dense(weight2,bias2))

        psCopy = Flux.params(dudtCopy)

        J0,back = Tracker.forward(() -> sum(abs2,ode_data .- predict_rk4(dudtCopy)), psCopy)
        gc = back(1)
        dJ = []
        for p in psCopy
            # println(size(gc[p]))
            gp = gc[p]
            push!(dJ,gp)
        end

        # eigVals = eigen((Flux.jacobian(x -> sin.(x), rand(5))).data).values
        Jac = Flux.jacobian(x -> dudtCopy(x), rand(2)) # u0 or rand(2)?
        # eye = Matrix{Float64}(I,2,2) # 2x2 identity
        # eye = Diagonal(ones(2,2))
        eye = [0 1; 1 0]
        eigVals = eigen(Jac.data,eye).values # need the eye to get the imaginary part
        println(eigVals)
        if eigVals[1] isa Float64
            sEigs = @sprintf("%5.3f + 0i , %5.3f + 0i", eigVals[1], eigVals[2])
        else # if complex
            sEigs = @sprintf("%5.3f + %5.3fi , %5.3f + %5.3fi", eigVals[1].re, eigVals[1].im,
                            eigVals[2].re, eigVals[2].im)
        end


        J0 = sum(abs2,ode_data .- predict_rk4(dudtCopy))
        W10  = copy(dudtCopy[2].W.data)
        b10  = copy(dudtCopy[2].b.data)
        W20  = copy(dudtCopy[3].W.data)
        b20  = copy(dudtCopy[3].b.data)

        dW1 = randn(Float32,size(W10))
        db1 = randn(Float32,size(b10))
        dW2 = randn(Float32,size(W20))
        db2 = randn(Float32,size(b20))

        dJdW1 = dot(dJ[1],dW1)
        dJdb1 = dot(dJ[2],db1)
        dJdW2 = dot(dJ[3],dW2)
        dJdb2 = dot(dJ[4],db2)
        dJdth = dJdW1 + dJdb1 + dJdW2 + dJdb2
        hisErr = zeros(20,3)
        for k=1:20
            h = 2.0f0^(-k)

            dudtCopy[2].W.data .= W10 + h*dW1
            dudtCopy[2].b.data .= b10 + h*db1
            dudtCopy[3].W.data .= W20 + h*dW2
            dudtCopy[3].b.data .= b20 + h*db2

            Jt = sum(abs2,ode_data .- predict_rk4(dudtCopy))
            err0 = abs(Jt.data-J0.data)
            err1 = abs(Jt.data-J0.data-h*dJdth)
            hisErr[k,:] = [h err0 err1.data]

            # println("h=$(h)\terr0=$(err0)\terr1=$(err1)")
        end

        # for k=1:20
        #     println("$(hisErr[k,1])\t$(hisErr[k,3])\\\\")
        # end

        p = plot(hisErr[:,1],hisErr[:,2],linewidth=5,
            title="Disc.-Opt. Derivative Check iter=$(length(hisRK)-1)\neigs: " * sEigs,
            xscale=:log10,yscale=:log10,
            xaxis="Step Size (h)",yaxis="Error",label="E0")
        plot!(p, hisErr[:,1],hisErr[:,3],linewidth=5,
              label="E1",legend=:topleft)


        savefig(p,"image/tmp/seed$(seed)-derivCheckRK4-iter-$(length(hisRK)-1).pdf")

    end

end

# Display the ODE with the initial parameter values.
cbRK()

Flux.train!(loss_rk, ps, data, opt, cb = cbRK)

#--------------------------------------------
# multT = 4# multiply time end by this multiplier
# tspanlong = (0.0,multT*1.5)
# tlong = range(tspanlong[1],tspanlong[2],length=multT*datasize)

function extrapolate_rk4(model=dudt)
    tt = @elapsed begin
    un = copy(u0)
    pred = zeros(2,multT*datasize)
    tk = tspanlong[1] #!!!!!!!!!!!!!!!!!!!!
    pred = un
    for k=1:multT*datasize-1
        un = step(stepper,model,un,tk,tk+h)
        pred = [pred un]
        tk+=h
    end
    end
    push!(tpode,tt)
    return pred
end
extra_RK = extrapolate_rk4()
# extra_OD = neural_ode(model,x,tspanlong,Tsit5(),saveat=tlong,reltol=reltol,abstol=abstol) # neural_ode uses diffeq_adjoint

problong = ODEProblem(trueODEfunc,u0,tspanlong)
ode_datalong = Array(solve(problong,Tsit5(),saveat=tlong))

pl = scatter(tlong,ode_datalong[1,:],label="data",
    xlabel=L"\mathbf{t}",ylabel=L"\mathbf{u_1}",legend=:topright, margin=5Plots.mm)
scatter!(pl,tlong,Flux.data(extra_RK[1,:]),label="pred Disc-Opt")
scatter!(pl,tlong,Flux.data(extra_OD[1,:]),label="pred Opt-Disc")

str = "Trained on t=[0,1.5] , Predicted on t=[0,6]"

println(str)
# title!(L"\text{Trained on } t \in [0,1.5] \text{ , Predicted on } t \in [0,6]")
title!(str)
display(plot(pl))
fn = "image/seed$(seed)-extrapolation.pdf"
savefig(pl,fn)



#--------------------------------------------



####################
fontSize=16
xTicksTuple = (0:25:300, ["0","","","","100","","","","200","","","","300"])

# save data and make plots

# using JLD
# save("seed$(seed)_hisAndTimes.jld","hisOD",his,"timesOD", times,
#             "hisRK",hisRK,"timesRK", timesRK)

using HDF5

h5open("seed$(seed)_hisAndTimes.h5", "w") do file
    write(file, "hisOD",float.(his))
    write(file, "timesOD", float.(times))
    write(file, "hisRK", float.(hisRK))
    write(file, "timesRK", float.(timesRK))
end

lossPlot = plot(his, linewidth=1, linecolor=:red,label="", # yscale=:log10,
                ylabel="Loss", xlabel="Iteration", legend=:topright) # actual
plot!(lossPlot, hisRK, linewidth=1, linecolor=:blue,label="") # actual
plot!(lossPlot, accumulate(min, his), linewidth=5, linecolor=:red,
        label="Opt-Disc.") # accumulated

plot!(lossPlot, accumulate(min, hisRK), linewidth=5, linecolor=:blue,
        label="Disc-Opt.") # accumulated
plot!(lossPlot,xguidefontsize=fontSize,yguidefontsize=fontSize,
        legendfontsize=fontSize, tickfont=fontSize-2, xticks = xTicksTuple,
        ylims=(0,120))

savefig(lossPlot,"image/seed$(seed)_lossPlot.pdf")



timePlot = plot(times, linewidth=4, linecolor=:red,
                ylabel="Time (s)", xlabel="Iteration", label="Opt-Disc.")
plot!(timePlot, timesRK, linewidth=4, linecolor=:blue, label="Disc-Opt.")
plot!(timePlot,xguidefontsize=fontSize,yguidefontsize=fontSize,
        legendfontsize=fontSize, tickfont=fontSize-2, xticks = xTicksTuple,
        yticks=(0:.1:.7, [string(i) for i in 0:.1:.7]))



savefig(timePlot,"image/seed$(seed)_timePlot.pdf")
