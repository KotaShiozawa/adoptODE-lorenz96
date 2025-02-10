using ChaosTools, PredefinedDynamicalSystems, FractalDimensions
import OrdinaryDiffEqVerner.Vern9
using HDF5

forcing = 8.17 # system parameter
N = Int(1e6) # total system time to integrate for
Ttr = 1e3 # transient time
Δt = 0.5 # time step between re-orthogonalizations
diffeq = Dict(:reltol=>1e-9, :abstol=>1e-9, :alg=>Vern9())
Ds = [4]#, 5, 10, 20, 30, 40, 50, 60, 80, 100, 120]

for D in Ds
    println("Calculating Lyapunov spectrum for D = $D")
    ds = CoupledODEs(PredefinedDynamicalSystems.lorenz96(D; F=forcing), diffeq)
    λs = lyapunovspectrum(ds, N; Δt, Ttr, show_progress=true)

    h5open("../data/02_analysis/lyapunovs.h5", "r+") do file
        if haskey(file, "$D/spectrum")
            delete_object(file, "$D/spectrum")
            delete_object(file, "$D/ky_dim")
        end
        file["$D/spectrum"] = λs
        file["$D/ky_dim"] = kaplanyorke_dim(λs)
    end
end

