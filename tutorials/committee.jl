# # Committee Potentials

using Plots, ACE1pack

# ### Perform the fit

data = data_params(
    fname=joinpath(ACE1pack.artifact("Si_tiny_dataset"), "Si_tiny.xyz"),
    energy_key="dft_energy", force_key="dft_force", virial_key="dft_virial")
ace_basis = basis_params(
    type="ace",
    species=[:Si],
    N=4,
    maxdeg=12,
    r0=2.35126,
    radial=basis_params(type="radial", pin=2, pcut=2, rcut=5.5, rin=1.65))
pair_basis = basis_params(
    type="pair",
    species=[:Si],
    maxdeg=3,
    r0=2.35126,
    rcut=6.5,
    pcut=1,
    pin=0)
basis = Dict("ace"=>ace_basis, "pair"=>pair_basis)
e0 = Dict("Si"=>-158.54496821)
weights = Dict(
    "default" => Dict("E"=>30.0, "F"=>1.0, "V"=>1.0),
    "liq" => Dict("E"=>10.0, "F"=>0.66, "V"=>0.25))

# Give a nonzero committee_size to the solver

solver = solver_params(type=:blr, committee_size=10)

params = fit_params(
    data = data,
    basis = basis,
    solver = solver,
    e0 = e0,
    weights = weights,
    ACE_fname = "")

results = fit_ace(params)
ip_comm = results["IP_com"]

# ### Inspect the committee energies

atoms = rattle!(bulk(:Si, cubic=true) * 2, 0.2)
energy, energies = ACE1.co_energy(ip_comm, atoms)
plot(1:10, energy*ones(10), label="mean energy")
plot!(1:10, energies, seriestype=:scatter, label="committee energies")

# Committee forces

@show ACE1.co_forces(ip_comm, atoms)

# Committee virial

@show ACE1.co_virial(ip_comm, atoms)
