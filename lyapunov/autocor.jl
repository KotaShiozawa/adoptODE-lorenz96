using DelayEmbeddings, PredefinedDynamicalSystems, DynamicalSystems
using Distributions
using ProgressMeter
using HDF5

τs = 1:300
Δt = 0.01
Ds = 5:1:120

h5open("../data/02_analysis/autocor_and_mi.h5", "w") do file
    file["time"] = collect(τs) .* Δt
end

@showprogress for D in Ds
    tr, _ = trajectory(PredefinedDynamicalSystems.lorenz96(D; F=8.17), 100; Δt, Ttr=1000)
    acor_sys = []
    mutualinfo_sys = []
    for var in 1:D
        push!(acor_sys, autocor(tr[:, var], τs))
        push!(mutualinfo_sys, selfmutualinfo(tr[:, var], τs))
    end

    h5open("../data/02_analysis/autocor_and_mi.h5", "r+") do file
        file["$D/autocor/mean"] = vec(mean(hcat(acor_sys...), dims=2))
        file["$D/autocor/std"] = vec(std(hcat(acor_sys...), dims=2))
        file["$D/mutualinfo/mean"] = vec(mean(hcat(mutualinfo_sys...), dims=2))
        file["$D/mutualinfo/std"] = vec(std(hcat(mutualinfo_sys...), dims=2))
    end
end

