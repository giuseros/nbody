#include "nbody_cpu.hpp"
#include <cmath>

namespace
{
constexpr float G = 6.67408e-11;

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

void compute_graviation_cpu(float *force, const float *pos_mass, float softening_squared, int N)
{
	for (int i = 0 ; i < N; i++)
	{
		float x = pos_mass[i*4 + 0];
		float y = pos_mass[i*4 + 1];
		float z = pos_mass[i*4 + 2];


		float tot_ax = 0;
		float tot_ay = 0;
		float tot_az = 0;

		for (int j = 0; j < N; j++)
		{
            float obj_x = pos_mass[j*4 + 0];
			float obj_y = pos_mass[j*4 + 1];
			float obj_z = pos_mass[j*4 + 2];
			float obj_m = pos_mass[j*4 + 3];

			float fx, fy, fz;
			body_body_interaction(fx, fy, fz,
					              x, y, z,
								  obj_x, obj_y, obj_z, obj_m, softening_squared );

			tot_ax += fx;
			tot_ay += fy;
			tot_az += fz;
		}

		force[3*i + 0] = tot_ax;
		force[3*i + 1] = tot_ay;
		force[3*i + 2] = tot_az;
	}
}

