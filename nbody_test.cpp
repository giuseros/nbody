
#include "nbody_cuda.cuh"
#include "nbody_cpu.hpp"

#include <iostream>
#include <vector>
#include <random>
#include <functional>
#include <algorithm>

using namespace std;

void print_vector(std::vector<float>& V)
{
    for (auto e : V){std::cout<<e<<std::endl;}

}

void fill(std::vector<float> &a)
{
    random_device rnd_device;

    // Specify the engine and distribution.
    mt19937 mersenne_engine {rnd_device()};  // Generates random integers
    uniform_int_distribution<int> dist {1, 10};

    auto gen = [&dist, &mersenne_engine](){
                   return float(dist(mersenne_engine));
               };

    generate(a.begin(), a.end(), gen);
}

int main()
{
	const int N_body = 8;
	std::vector<float> pos_mass(N_body*4);
	std::vector<float>test_F(N_body*3);
	fill(pos_mass);
    print_vector(pos_mass);

	compute_graviation_cpu(test_F.data(), pos_mass.data(), 0.3, N_body);
    print_vector(test_F);
    std::cout<<"*********"<<std::endl;

	auto F = compute_graviation_gpu(pos_mass, 0.3, N_body );
    print_vector(F);



}
