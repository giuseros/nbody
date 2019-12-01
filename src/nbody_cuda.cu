#include "nbody_cuda.cuh"
#include <cmath>
#include <iostream>

#define BLOCK_SIZE 128

namespace
{
constexpr float G = 6.67408e-11;

struct DVector
{
	int N;
	float *data;
};

__device__
void body_body_interaction(float &ax, float &ay, float &az,
		                   float x0, float y0, float z0,
						   float x1, float y1, float z1, float m1, float softening_squared )
{
	float dx = x0 - x1;
	float dy = y0 - y1;
	float dz = z0 - z1;

	float d_sqr = dx*dx + dy*dy + dz*dz + softening_squared;
	float inv = 1./sqrt(d_sqr);
	float inv3 = inv*inv*inv;

	float s = m1 * inv3;
	ax = dx * s;
	ay = dy * s; 
    az = dz * s;
}
}

__global__
void compute_graviation_cuda(float *force, const float *pos_mass, float softening_squared, int N)
{

	float4 *pos_mass4 = (float4 *)pos_mass;
	__shared__ float4 buffer[BLOCK_SIZE];

	for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i+=blockDim.x*gridDim.x)
	{
		float acc[3] = {0.f, 0.f, 0.f};
		float4 me = pos_mass4[i];
#pragma unroll 32
		for (int j = 0; j<N/4; j+=  blockDim.x)
		{
			buffer[threadIdx.x] = pos_mass4[0+threadIdx.x];
			__syncthreads();

			for (int k = 0; k < N; k++){

				float4 body = buffer[k];
				float fx=0, fy=0, fz=0;

				// force j on i (f_{ij})
				body_body_interaction(fx, fy, fz,
									  me.x, me.y, me.z,
									  body.x, body.y, body.z, body.w, softening_squared );

				acc[0]+=fx;
				acc[1]+=fy;
				acc[2]+=fz;
			}
            __syncthreads();
		}
        force[3*i + 0] = acc[0];
        force[3*i + 1] = acc[1];
        force[3*i + 2] = acc[2];
	}
}

DVector make_dvector(const std::vector<float>& V)
{
	DVector dV;
    dV.N = int(V.size());

	float *dV_ptr;
	cudaMalloc(&dV_ptr, dV.N*sizeof(float));
	cudaMemcpy(dV_ptr, V.data(), dV.N*sizeof(float), cudaMemcpyHostToDevice);
	dV.data = dV_ptr;
	return dV;
}

DVector make_dvector(int n)
{
	DVector dV;
	dV.N = n;

	float *dV_ptr;
	cudaMalloc(&dV_ptr, n*sizeof(float));
	dV.data = dV_ptr;
	return dV;
}

std::vector<float> extract_vector(DVector dV)
{
	std::vector<float>v(dV.N);
	cudaMemcpy(v.data(), dV.data, dV.N*sizeof(float), cudaMemcpyDeviceToHost);
	return v;
}

std::vector<float> compute_graviation_gpu(const std::vector<float> &pos_mass, float softening_squared, int N)
{
	auto d_force = make_dvector(3*N);
	auto d_posmas = make_dvector(pos_mass);

	dim3 block(BLOCK_SIZE);
	dim3 grid((N + BLOCK_SIZE - 1)/ BLOCK_SIZE);
    compute_graviation_cuda<<<grid, block>>>(d_force.data, d_posmas.data, 0.3, N);
    auto force = extract_vector(d_force);
    return force;
}


