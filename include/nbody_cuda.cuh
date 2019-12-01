#ifndef NBODY_CUDA_CUH__
#define NBODY_CUDA_CUH__
#include <vector>
std::vector<float> compute_graviation_gpu(const std::vector<float> &pos_mass, float softening_squared, int N);
#endif
