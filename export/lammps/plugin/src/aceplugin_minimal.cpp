/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/

   aceplugin_minimal.cpp - Plugin registration for ACE/Minimal pair style

   This file registers the ACE/Minimal pair style with LAMMPS.
------------------------------------------------------------------------- */

#include "lammpsplugin.h"
#include "version.h"

using namespace LAMMPS_NS;

static Pair *acepaircreator_minimal(LAMMPS *lmp)
{
  return new PairACEMinimal(lmp);
}

// Plugin registration
extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  // ACE/Minimal pair style
  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "ace/minimal";
  plugin.info = "ACE potential pair style (minimal export approach) v1.0";
  plugin.author = "ACEpotentials.jl developers";
  plugin.creator.v1 = (lammpsplugin_factory1 *) &acepaircreator_minimal;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
