/*
 * OpenMP-accelerated wrapper for ACE potential library
 *
 * This provides parallelized system-level evaluation by:
 * 1. Computing neighbor list in C
 * 2. Using OpenMP to parallelize site evaluations
 * 3. Accumulating forces/virial with proper reductions
 *
 * Build:
 *   gcc -shared -fPIC -O3 -fopenmp -o libace_omp.so ace_omp_wrapper.c -ldl -lm
 *
 * Usage from Python:
 *   lib = ctypes.CDLL("libace_omp.so")
 *   lib.ace_omp_init("libace_test.so")
 *   lib.ace_omp_energy_forces_virial(...)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include <omp.h>

/* Function pointers to Julia library */
typedef double (*ace_site_efv_fn)(int, int, int*, double*, double*, double*);
typedef double (*ace_get_cutoff_fn)(void);
typedef int (*ace_get_n_species_fn)(void);
typedef int (*ace_get_species_fn)(int);

static void* julia_lib = NULL;
static ace_site_efv_fn ace_site_energy_forces_virial = NULL;
static ace_get_cutoff_fn ace_get_cutoff = NULL;
static ace_get_n_species_fn ace_get_n_species = NULL;
static ace_get_species_fn ace_get_species = NULL;
static double cutoff = 0.0;

/* Initialize by loading the Julia library */
int ace_omp_init(const char* lib_path) {
    if (julia_lib != NULL) {
        dlclose(julia_lib);
    }

    julia_lib = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!julia_lib) {
        fprintf(stderr, "Cannot load library: %s\n", dlerror());
        return -1;
    }

    ace_site_energy_forces_virial = (ace_site_efv_fn)dlsym(julia_lib, "ace_site_energy_forces_virial");
    ace_get_cutoff = (ace_get_cutoff_fn)dlsym(julia_lib, "ace_get_cutoff");
    ace_get_n_species = (ace_get_n_species_fn)dlsym(julia_lib, "ace_get_n_species");
    ace_get_species = (ace_get_species_fn)dlsym(julia_lib, "ace_get_species");

    if (!ace_site_energy_forces_virial || !ace_get_cutoff) {
        fprintf(stderr, "Cannot find required symbols in library\n");
        return -1;
    }

    /* Initialize cutoff (triggers Julia runtime init on main thread) */
    cutoff = ace_get_cutoff();

    return 0;
}

/* Get number of OpenMP threads */
int ace_omp_get_num_threads(void) {
    return omp_get_max_threads();
}

/* Set number of OpenMP threads */
void ace_omp_set_num_threads(int n) {
    omp_set_num_threads(n);
}

/* Get cutoff */
double ace_omp_get_cutoff(void) {
    return cutoff;
}

/*
 * Compute energy, forces, and virial with OpenMP parallelization.
 *
 * Arguments:
 *   natoms: Number of atoms
 *   species: [natoms] atomic numbers
 *   positions: [natoms*3] positions (x1,y1,z1,x2,...)
 *   cell: [9] cell vectors row-major, or NULL for non-periodic
 *   pbc: [3] periodic flags, or NULL
 *   forces: [natoms*3] output forces (output)
 *   virial: [9] virial tensor row-major (output)
 *
 * Returns: Total energy
 */
double ace_omp_energy_forces_virial(
    int natoms,
    int* species,
    double* positions,
    double* cell,
    int* pbc,
    double* forces,
    double* virial
) {
    if (!ace_site_energy_forces_virial) {
        fprintf(stderr, "Library not initialized. Call ace_omp_init first.\n");
        return 0.0;
    }

    double rcut = cutoff;
    double rcut_sq = rcut * rcut;

    /* Determine periodic images to search */
    int n_images[3] = {0, 0, 0};
    if (cell != NULL && pbc != NULL) {
        for (int d = 0; d < 3; d++) {
            if (pbc[d]) {
                /* Cell vector length (column d of cell matrix) */
                double len = sqrt(cell[d]*cell[d] + cell[3+d]*cell[3+d] + cell[6+d]*cell[6+d]);
                n_images[d] = (int)ceil(rcut / len);
            }
        }
    }

    /* Initialize outputs */
    memset(forces, 0, natoms * 3 * sizeof(double));
    memset(virial, 0, 9 * sizeof(double));

    double total_energy = 0.0;

    /* Parallel region with private neighbor arrays */
    #pragma omp parallel reduction(+:total_energy)
    {
        /* Thread-private work arrays - sized for worst case */
        int max_neigh = natoms * (2*n_images[0]+1) * (2*n_images[1]+1) * (2*n_images[2]+1);
        if (max_neigh > 10000) max_neigh = 10000;  /* Cap for memory */

        int* neigh_z = (int*)malloc(max_neigh * sizeof(int));
        double* neigh_R = (double*)malloc(max_neigh * 3 * sizeof(double));
        double* site_forces = (double*)malloc(max_neigh * 3 * sizeof(double));
        double site_virial[6];
        int* neigh_j = (int*)malloc(max_neigh * sizeof(int));

        /* Thread-private force accumulator */
        double* local_forces = (double*)calloc(natoms * 3, sizeof(double));
        double local_virial[9] = {0};

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < natoms; i++) {
            double xi = positions[i*3];
            double yi = positions[i*3 + 1];
            double zi = positions[i*3 + 2];
            int z0 = species[i];

            /* Build neighbor list for atom i */
            int nneigh = 0;

            for (int na = -n_images[0]; na <= n_images[0]; na++) {
                for (int nb = -n_images[1]; nb <= n_images[1]; nb++) {
                    for (int nc = -n_images[2]; nc <= n_images[2]; nc++) {
                        /* Compute shift vector */
                        double shift[3] = {0, 0, 0};
                        if (cell != NULL) {
                            for (int d = 0; d < 3; d++) {
                                shift[d] = na * cell[d] + nb * cell[3+d] + nc * cell[6+d];
                            }
                        }

                        for (int j = 0; j < natoms; j++) {
                            if (na == 0 && nb == 0 && nc == 0 && i == j) continue;

                            double dx = positions[j*3] + shift[0] - xi;
                            double dy = positions[j*3+1] + shift[1] - yi;
                            double dz = positions[j*3+2] + shift[2] - zi;
                            double rsq = dx*dx + dy*dy + dz*dz;

                            if (rsq < rcut_sq && rsq > 1e-10 && nneigh < max_neigh) {
                                neigh_j[nneigh] = j;
                                neigh_z[nneigh] = species[j];
                                neigh_R[nneigh*3] = dx;
                                neigh_R[nneigh*3+1] = dy;
                                neigh_R[nneigh*3+2] = dz;
                                nneigh++;
                            }
                        }
                    }
                }
            }

            if (nneigh == 0) continue;

            /* Initialize site outputs */
            memset(site_forces, 0, nneigh * 3 * sizeof(double));
            memset(site_virial, 0, 6 * sizeof(double));

            /* Call Julia library */
            double Ei = ace_site_energy_forces_virial(
                z0, nneigh, neigh_z, neigh_R, site_forces, site_virial);

            total_energy += Ei;

            /* Accumulate forces (Newton's 3rd law) */
            double fix = 0, fiy = 0, fiz = 0;
            for (int k = 0; k < nneigh; k++) {
                int j = neigh_j[k];
                double fx = site_forces[k*3];
                double fy = site_forces[k*3+1];
                double fz = site_forces[k*3+2];

                local_forces[j*3] += fx;
                local_forces[j*3+1] += fy;
                local_forces[j*3+2] += fz;

                fix -= fx;
                fiy -= fy;
                fiz -= fz;
            }
            local_forces[i*3] += fix;
            local_forces[i*3+1] += fiy;
            local_forces[i*3+2] += fiz;

            /* Accumulate virial (Voigt -> 3x3) */
            local_virial[0] += site_virial[0];  /* xx */
            local_virial[4] += site_virial[1];  /* yy */
            local_virial[8] += site_virial[2];  /* zz */
            local_virial[5] += site_virial[3];  /* yz */
            local_virial[7] += site_virial[3];  /* zy */
            local_virial[2] += site_virial[4];  /* xz */
            local_virial[6] += site_virial[4];  /* zx */
            local_virial[1] += site_virial[5];  /* xy */
            local_virial[3] += site_virial[5];  /* yx */
        }

        /* Reduce forces and virial to global arrays */
        #pragma omp critical
        {
            for (int i = 0; i < natoms * 3; i++) {
                forces[i] += local_forces[i];
            }
            for (int i = 0; i < 9; i++) {
                virial[i] += local_virial[i];
            }
        }

        free(neigh_z);
        free(neigh_R);
        free(site_forces);
        free(neigh_j);
        free(local_forces);
    }

    return total_energy;
}

/* Cleanup */
void ace_omp_finalize(void) {
    if (julia_lib) {
        dlclose(julia_lib);
        julia_lib = NULL;
    }
}
