module ExportMulti

using Interpolations
using OrderedCollections
using YAML
using ACE1
using ACE1: PIBasis, PIBasisFcn, PIPotential
using ACE1.OrthPolys: TransformedPolys
using ACE1: rand_radial, cutoff, numz, ZList
using JuLIP: energy, bulk, i2z, z2i, chemical_symbol, SMatrix

function export_ACE(fname, IP; export_pairpot_as_table=false)
    # supply fname with the .yace extension
    # if export_pairpot_as_table, don't write the pairpot and make .table file instead.
    if !(fname[end-4:end] == ".yace")
        throw(ArgumentError("Potential name must be supplied with .yace extension"))
    end

    # decomposing into V1, V2, V3 (One body, two body and ACE bases)
    # they could be in a different order
    if length(IP.components) != 3
        throw("IP must have three components which are OneBody, pair potential, and ace")
    end

    ordered_components = []

    for target_type in [OneBody, PolyPairPot, PIPotential]
        did_not_find = true
        for i = 1:3
            if typeof(IP.components[i]) <: target_type
                push!(ordered_components, IP.components[i])
                did_not_find = false
            end
        end

        if did_not_find
            throw("IP must have three components which are OneBody, pair potential, and ace")
        end
    end

    V1 = ordered_components[1]
    V2 = ordered_components[2]
    V3 = ordered_components[3]
    
    species = collect(string.(chemical_symbol.(V3.pibasis.zlist.list)))
    species_dict = Dict(zip(collect(0:length(species)-1), species))
    reversed_species_dict = Dict(zip(species, collect(0:length(species)-1)))

    # check for the old pairpotential style: one basis, or the new style: a matrix of bases
    matrix_v2 = false
    if typeof(V2.basis.J) <: SMatrix
        matrix_v2 = true
        if !(export_pairpot_as_table)
            throw(ArgumentError(
                "This potential was made using a recent version of ACE1. Exporting with export_pairpot_as_table=false is only possible for older potential files. See https://acesuit.github.io/ACE1docs.jl/dev/Using_ACE/lammps/"))
        end
    #else
        #warn("This potential has been made with an older version of ACE1, and may not be compatible")
    end

    data = Dict()

    data["deltaSplineBins"] = 0.001 #" none

    elements = Vector(undef, length(species))
    E0 = zeros(length(elements))
    
    for (index, element) in species_dict
        E0[index+1] = V1(Symbol(element))
        elements[index+1] = element
    end

    # grabbing the elements key and E0 key from the onebody (V1)
    data["elements"] = elements

    # 1body
    data["E0"] = E0

    # 2body
    if export_pairpot_as_table
        # I am not handling the hasproperty(V2, :Vin) case, since I don't know what this is
        # this writes a .table file, so for simplicity require that export fname is passed with
        # .yace extension, and we remove this and add the .table extension instead
        fname_stem = fname[1:end-5]
        write_pairpot_table(fname_stem, V2, species_dict)
    else
        if hasproperty(V2, :basis)
            polypairpot = export_polypairpot(V2, reversed_species_dict)
        else hasproperty(V2, :Vin)
            polypairpot = export_polypairpot(V2.Vout, reversed_species_dict)
            reppot = export_reppot(V2, reversed_species_dict)
            data["reppot"] = reppot
        end
        data["polypairpot"] = polypairpot
    end

    # ACE
    embeddings, bonds = export_radial_basis(V3, species_dict)
    data["embeddings"] = embeddings
    data["bonds"] = bonds

    functions, lmax = export_ACE_functions(V3, species, reversed_species_dict)
    data["functions"] = functions
    data["lmax"] = lmax

    YAML.write_file(fname, data)
end

function export_ace(fname, IP)
    # supply fname with the .yace extension
    # if export_pairpot_as_table, don't write the pairpot and make .table file instead.
    if !(fname[end-4:end] == ".yace")
        throw(ArgumentError("Potential name must be supplied with .yace extension"))
    end

    # decomposing into V1, V2, V3 (One body, two body and ACE bases)
    # they could be in a different order
    if length(IP.components) != 3
        throw("IP must have three components which are OneBody, pair potential, and ace")
    end

    ordered_components = []

    for target_type in [OneBody, PolyPairPot, PIPotential]
        did_not_find = true
        for i = 1:3
            if typeof(IP.components[i]) <: target_type
                push!(ordered_components, IP.components[i])
                did_not_find = false
            end
        end

        if did_not_find
            throw("IP must have three components which are OneBody, pair potential, and ace")
        end
    end

    V1 = ordered_components[1]
    V2 = ordered_components[2]
    V3 = ordered_components[3]
    
    species = collect(string.(chemical_symbol.(V3.pibasis.zlist.list)))
    species_dict = Dict(zip(collect(0:length(species)-1), species))
    reversed_species_dict = Dict(zip(species, collect(0:length(species)-1)))

    data = OrderedDict()

    #data["deltaSplineBins"] = 0.001 #" none

    elements = Vector(undef, length(species))
    E0 = zeros(length(elements))
    
    for (index, element) in species_dict
    #    E0[index+1] = V1(Symbol(element))
        elements[index+1] = element
    end

    # grabbing the elements key and E0 key from the onebody (V1)
    data["elements"] = elements

    # 1body
    data["E0"] = E0

    ## 2body
    #if export_pairpot_as_table
    #    # I am not handling the hasproperty(V2, :Vin) case, since I don't know what this is
    #    # this writes a .table file, so for simplicity require that export fname is passed with
    #    # .yace extension, and we remove this and add the .table extension instead
    #    fname_stem = fname[1:end-5]
    #    write_pairpot_table(fname_stem, V2, species_dict)
    #else
    #    if hasproperty(V2, :basis)
    #        polypairpot = export_polypairpot(V2, reversed_species_dict)
    #    else hasproperty(V2, :Vin)
    #        polypairpot = export_polypairpot(V2.Vout, reversed_species_dict)
    #        reppot = export_reppot(V2, reversed_species_dict)
    #        data["reppot"] = reppot
    #    end
    #    data["polypairpot"] = polypairpot
    #end

    # ACE
#    embeddings, bonds = export_radial_basis(V3, species_dict)
    data["embeddings"] = Dict()
    data["embeddings"][0] = Dict(
        "ndensity" => 1,
        "FS_parameters" => [1.0, 1.0],
        "npoti" => "FinnisSinclairShiftedScaled",
        "drho_core_cutoff" => 1.000000000000000000,
        "rho_core_cutoff" => 100000.000000000000000000)

    radialsplines = ACE1.Splines.RadialSplines(V3.pibasis.basis1p.J; nnodes = 10000)
    ranges, nodalvals, zlist = ACE1.Splines.export_splines(radialsplines)

    # TODO: move this elsewhere
    # compute spline derivatives
    nodalderivs = similar(nodalvals)
    for iz1 in size(nodalvals,2), iz2 in size(nodalvals,3)
        range = ranges[iz1,iz2]
        for i in 1:size(radialsplines.splines,1)
            spl = radialsplines.splines[1,iz1,iz2]
            deriv(r) = Interpolations.gradient(spl,r)[1]
            nodalderivs[i,iz1,iz2] = deriv.(range)
        end
    end

    # export splines
    data["bonds"] = OrderedDict()
    for iz1 in size(nodalvals,2), iz2 in size(nodalvals,3)
        data["bonds"][[iz1-1,iz2-1]] = OrderedDict{Any,Any}(
            "radbasename" => "ACE.jl",
            "maxn" => length(V3.pibasis.basis1p.J.J.A),
            "rcut" => ranges[iz1,iz2][end],
            "ntot" => length(ranges[iz1,iz2])-1)
        nodalvals_map = OrderedDict([i-1 => nodalvals[i,iz1,iz2] for i in 1:size(nodalvals,1)])
        data["bonds"][[iz1-1,iz2-1]]["splinenodalvals"] = nodalvals_map
        nodalderivs_map = OrderedDict([i-1 => nodalderivs[i,iz1,iz2] for i in 1:size(nodalvals,1)])
        data["bonds"][[iz1-1,iz2-1]]["splinenodalderivs"] = nodalderivs_map
    end

    functions, lmax = export_ACE_functions(V3, species, reversed_species_dict)
    data["functions"] = functions
    data["lmax"] = lmax

    YAML.write_file(fname, data)
end

function export_reppot(Vrep, reversed_species_dict)
    reppot = Dict("coefficients" => Dict())

    zlist_dict = Dict(zip(1:length(Vrep.Vout.basis.zlist.list), [string(chemical_symbol(z)) for z in Vrep.Vout.basis.zlist.list]))

    for (index1, element1) in zlist_dict
        for (index2, element2) in zlist_dict
            pair = [reversed_species_dict[element1], reversed_species_dict[element2]]
            coefficients = Dict( "A" => Vrep.Vin[index1, index2].A,
                                "B" => Vrep.Vin[index1, index2].B,
                                "e0" => Vrep.Vin[index1, index2].e0,
                                "ri" => Vrep.Vin[index1, index2].ri) 
            reppot["coefficients"][pair] = coefficients
        end
    end

    return reppot
end

function export_polypairpot(V2, reversed_species_dict)
    Pr = V2.basis.J

    p = Pr.trans.p
    r0 = Pr.trans.r0
    xr = Pr.J.tr
    xl = Pr.J.tl
    pr = Pr.J.pr
    pl = Pr.J.pl
    rcut = cutoff(Pr)
    maxn = length(Pr)

    if length(keys(reversed_species_dict)) == 1
        num_coeffs = length(V2.coeffs)
    else
        num_coeffs = vcat(V2.basis.bidx0...)[2]
    end

    zlist_dict = Dict(zip(1:length(V2.basis.zlist.list), [string(chemical_symbol(z)) for z in V2.basis.zlist.list]))

    polypairpot = Dict( "p" => p,
                        "r0" => r0,
                        "xr" => xr,
                        "xl" => xl,
                        "pr" => pr,
                        "pl" => pl,
                        "rcut" => rcut,
                        "maxn"=> maxn,
                        "recursion_coefficients" => Dict("A" => [Pr.J.A[i] for i in 1:maxn],
                                                         "B" => [Pr.J.B[i] for i in 1:maxn],
                                                         "C" => [Pr.J.C[i] for i in 1:maxn],),
                        "coefficients" => Dict())

    for (index1, element1) in zlist_dict
        for (index2, element2) in zlist_dict
            pair = [reversed_species_dict[element1], reversed_species_dict[element2]]
            ind = V2.basis.bidx0[index1, index2]
            polypairpot["coefficients"][pair] = V2.coeffs[ind+1:ind+num_coeffs]
        end
    end
    
    return polypairpot
end


make_dimer(s1, s2, rr) = Atoms(
    [[0.0,0.0,0.0],[rr,0.0,0.0]], 
    [[0.0,0.0,0.0],[0.0,0.0,0.0]],
    [atomic_mass(s1), atomic_mass(s2)],
    [AtomicNumber(s1), AtomicNumber(s2)],
    [100.0,100.0,100.0],
    [false, false, false])

function write_pairpot_table(fname, V2, species_dict)
    # fname is JUST THE STEM
    # write a pair_style table file for LAMMPS
    # the file has a seperate section for each species pair interaction
    # format of table pair_style is described at https://docs.lammps.org/pair_table.html

    # Create filename. Only the stem is specified
    fname = fname * "_pairpot.table"

    # enumerate sections
    species_pairs = []
    for i in 0:length(species_dict) - 1
        for j in i:length(species_dict) - 1
            push!(species_pairs, (species_dict[i], species_dict[j]))
        end
    end

    lines = Vector{String}()

    # make header. date is none since ACE1 current doesnt depend on time/dates package
    push!(lines, "# DATE: none UNITS: metal CONTRIBUTOR: ACE1.jl - https://github.com/ACEsuit/ACE1.jl")
    push!(lines, "# ACE1 pair potential")
    push!(lines, "")

    for spec_pair in species_pairs
        # make dimer
        dimer = make_dimer(Symbol(spec_pair[1]), Symbol(spec_pair[2]), 1.0)

        # get inner and outer cutoffs

        if typeof(V2.basis.J) <: SMatrix
            get_ru(jj) = jj.ru
            rus = get_ru.(V2.basis.J)
            rout = maximum(rus)    
        else
            rout = V2.basis.J.ru
        end

        rin = 0.1
        spacing = 0.001
        rs = rin:spacing:rout

        # section header
        push!(lines, string(spec_pair[1], "_", spec_pair[2]))
        push!(lines, string("N ", length(rs)))
        push!(lines, "")
        
        # values
        for (index, R) in enumerate(rs)
            set_positions!(dimer, AbstractVector{JVec{Float64}}([[R,0.0,0.0], [0.0,0.0,0.0]]))
            E = energy(V2, dimer)
            F = forces(V2, dimer)[1][1]
            push!(lines, string(index, " ", R, " ", E, " ", F))
        end
        push!(lines, "")
    end

    # write
    open(fname, "w+") do io
        for line in lines
            write(io, line * "\n")
        end
    end

    return nothing
end

function export_radial_basis(V3, species_dict)
    #grabbing the transform and basis
    transbasis = V3.pibasis.basis1p.J
    Pr = V3.pibasis.basis1p

    #grabbing all the required params
    p = transbasis.trans.p
    r0 = transbasis.trans.r0
    xr = transbasis.J.tr
    xl = transbasis.J.tl
    pr = transbasis.J.pr
    pl = transbasis.J.pl
    rcut = cutoff(Pr)
    maxn = length(V3.pibasis.basis1p.J.J.A)

    #guessing "radbasname" is that just "polypairpots"
    radbasename = "ACE.jl.base"

    embeddings = Dict()

    for species_ind1 in sort(collect(keys(species_dict)))
        embeddings[species_ind1] = Dict("ndensity" => 1,
                    "FS_parameters" => [1.0, 1.0],
                    "npoti" => "FinnisSinclairShiftedScaled",
                    "drho_core_cutoff" => 1.000000000000000000,
                    "rho_core_cutoff" => 100000.000000000000000000)
    end

    bonds = Dict()
    #this does not respect the coefficient decompositions required per pair
    #need to figure out how to get the right coeffs per pair
    for species_ind1 in sort(collect(keys(species_dict)))
        for species_ind2 in sort(collect(keys(species_dict)))
            pair = [species_ind1, species_ind2]
            bonds[pair] = Dict("p" => p,
                "r0" => r0,
                "xl" => xl,
                "xr" => xr,
                "pr" => pr,
                "pl" => pl,
                "rcut" => rcut,
                "radbasename" => radbasename,
                "maxn" => maxn,
                "recursion_coefficients" => Dict("A" => [Pr.J.J.A[i] for i in 1:maxn],
                                                 "B" => [Pr.J.J.B[i] for i in 1:maxn],
                                                 "C" => [Pr.J.J.C[i] for i in 1:maxn],))
        end
    end

    return embeddings, bonds
end

function export_ACE_functions(V3, species, reversed_species_dict)
    functions = Dict()
    lmax = 0

    for i in 1:length(V3.pibasis.inner)
        sel_bgroups = []
        inner = V3.pibasis.inner[i]
        z0 = V3.pibasis.inner[i].z0
        coeffs = V3.coeffs[i]
        groups = _basis_groups(inner, coeffs)
        for group in groups
            for (m, c) in zip(group["M"], group["C"])
                c_ace = c / (4*π)^(group["ord"]/2)
                #@show length(c_ace)
                ndensity = 1
                push!(sel_bgroups, Dict("rank" => group["ord"],
                            "mu0" => reversed_species_dict[string(chemical_symbol(group["z0"]))],
                            "ndensity" => ndensity,
                            "ns" => group["n"],
                            "ls" => group["l"],
                            "mus" => [reversed_species_dict[i] for i in string.(chemical_symbol.(group["zs"]))],
                            "ctildes" => [c_ace],
                            "ms_combs" => m,
                            "num_ms_combs" => length([c_ace])))
                if maximum(group["l"]) > lmax
                    lmax = maximum(group["l"])
                end
            end
        end
        functions[reversed_species_dict[string(chemical_symbol(z0))]] = sel_bgroups
    end

    return functions, lmax
end

function _basis_groups(inner, coeffs)
    ## grouping the basis functions
    NLZZ = []
    M = []
    C = []
    for b in keys(inner.b2iAA)
       if coeffs[ inner.b2iAA[b] ] != 0
          push!(NLZZ, ( [b1.n for b1 in b.oneps], [b1.l for b1 in b.oneps], [b1.z for b1 in b.oneps], b.z0))
          push!(M, [b1.m for b1 in b.oneps])
          push!(C, coeffs[ inner.b2iAA[b] ])
       end
    end
    ords = length.(M)
    perm = sortperm(ords)
    NLZZ = NLZZ[perm]
    M = M[perm]
    C = C[perm]
    @assert issorted(length.(M))
    bgrps = []
    alldone = fill(false, length(NLZZ))
    for i = 1:length(NLZZ)
       if alldone[i]; continue; end
       nlzz = NLZZ[i]
       Inl = findall(NLZZ .== Ref(nlzz))
       alldone[Inl] .= true
       Mnl = M[Inl]
       Cnl = C[Inl]
       pnl = sortperm(Mnl)
       Mnl = Mnl[pnl]
       Cnl = Cnl[pnl]
       order = length(nlzz[1])
       push!(bgrps, Dict("n" => nlzz[1], "l" => nlzz[2], "z0" => nlzz[4], "zs" => nlzz[3],
                         "M" => Mnl, "C" => Cnl, "ord" => order)) #correct?
    end
    return bgrps
end

end
