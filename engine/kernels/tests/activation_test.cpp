// CPU entry for the exact shared activation helper (also compiled into CUDA kernels).
#include "../activation.cuh"
#include <cstdio>
int main()
{
    const float values[] = {-100.f, -11.f, -10.f, -9.999f, -1.f, -0.f, 0.1f, 1.f, 9.999f, 10.f, 11.f, 100.f};
    for (float g : values) for (float u : values)
        printf("%.9g %.9g %.9g\n", g, u, swiglu_f(g, u));
}
