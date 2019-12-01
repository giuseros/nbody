#ifndef NBODY_CPU_HPP__
#define NBODY_CPU_HPP__

void compute_graviation_cpu(float *force, const float *pos_mass, float softening_squared, int N);

#endif
