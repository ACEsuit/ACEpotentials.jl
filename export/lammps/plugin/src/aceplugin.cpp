/* ----------------------------------------------------------------------
   LAMMPS ACE Plugin Registration

   This file registers the pair_style ace with LAMMPS as a loadable plugin.
   The plugin can be loaded at runtime using:
     plugin load /path/to/aceplugin.so

   Usage:
     pair_style ace
     pair_coeff * * /path/to/model.so Si O ...

   Where model.so is an ACE potential compiled from ACEpotentials.jl using
   Julia's juliac --trim feature.
------------------------------------------------------------------------- */

#include "lammpsplugin.h"
#include "version.h"

#include <cstring>

#include "pair_ace.h"

using namespace LAMMPS_NS;

static Pair *acecreator(LAMMPS *lmp)
{
  return new PairACE(lmp);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "ace";
  plugin.info = "ACE potential loader for ACEpotentials.jl compiled models";
  plugin.author = "ACEsuit developers";
  plugin.creator.v2 = (lammpsplugin_factory2 *) &acecreator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
