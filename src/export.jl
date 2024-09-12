"""
`export2lammps(pathtofile, model::ACE1Model)` : exports the potential to the
`.yace` format for use in LAMMPS.
"""
function export2lammps(pathtofile, model::ACE1Model)
   if pathtofile[end-4:end] != ".yace"
      @warn("the lammps potential filename should end in .yace")
   end
   export2lammps(pathtofile, model.potential)
end
