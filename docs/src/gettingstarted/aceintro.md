
# Introduction to ACE Models

The purpose of this section is to give a brief summary of the mathematics behind linear ACE parameterisations of invariant atomic properties. It is not required to use the `ACEpotentials` and `ACE1` packages and can be skipped on a first reading.

### Invariant Properties

To explain the main ideas in the simplest non-trivial setting, we consider systems of indistinguishable particles. A configuration is an mset ``R := \{ \bm r_j \}_j \subset \mathbb{R}^3`` with arbitary numbers of particles and we wish to develop representation of properties 
```math 
   \varphi\big(R) \in \mathbb{R}
```
which are invariant under permutations (already implicit in the fact that ``R`` is an mset) and under isometries ``O(3)``. To make this explicit we can write this as
```math 
\varphi\big( \{ Q \bm r_{\sigma j} \}_j \big)
=
\varphi\big( \{ \bm r_{j} \}_j \big) \qquad \forall Q \in O(3), 
\quad \sigma \text{ a permutation}.
```
To that end we proceed in three steps: 

### Density Projection / Atomic Base 

We define the "atomic density"
```math 
\rho({\bm r}) := \sum_j \delta({\bm r} - {\bm r}_j)
```
Then we choose a one-particle basis 
```math 
\phi_v({\bm r}) = \phi_{nlm}({\bm r}) = R_n(r) Y_l^m(\hat{\bm r})
```
and project ``\rho``` onto that basis, 
```math 
A_{v} = A_{nlm} = \langle \phi_{nlm}, \rho \rangle = 
   \sum_j \phi_{nlm}({\bm r}_j).
```

### Density correlations 

Next, we form the $$N$$-correlations of the density, ``\rho^{\otimes N}`` and project them onto the tensor project basis, 
```math 
   {\bm A}_{{\bm nlm}} 
   = \Big\langle \otimes_{t = 1}^N \phi_{n_t l_t m_t}, \rho^{\otimes N} \Big\rangle 
   = \prod_{t = 1}^N A_{n_t l_t m_t}.
```
The reason to introduce these is that in the next step, the symmetrisation step the density project would loose all angular information while the ``N``-correlations retain most (though not all) of it. 

### Symmetrisation 

Finally, we symmetrize the ``N``-correlations, by integrating over the ``O(3)``-Haar measure, 
```math 
  B_{\bm nlm} \propto 
  \int_{O(3)} {\bm A}_{\bm nlm} \circ Q \, dQ 
```
Because of properties of the spherical harmonics one can write this as 
```math 
  {\bm B} = \mathcal{U} {\bm A},
```
where ``{\bm A}`` is the vector of 1, 2, ..., N correlations (the maximal ``N`` is an approximation parameter!) and ``\mathcal{U}`` is a sparse matrix (the coupling coefficients).

If one symmetrised all possible ``N``-correlations then this would create a spanning set, but one can easily reduce this to an actual basis. This construction then yields a basis of the space of symmetric polynomials. 

Notes: 
* Because of permutation symmetry only ordered ``{\bm v}`` tuples are retained

### Linear Dependence

The construction described above introduces a lot of linear dependence which is removed in the ACE basis construction in a mixed symbolic / numerical procedure. In the end we no longer index the symmetrized basis functions as ``B_{\bm nlm}`` but as ``B_{{\bm nl}i}`` with ``i`` indexing the linearly independent basis functions from the ``{\bm nl}`` block. 
