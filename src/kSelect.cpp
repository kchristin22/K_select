#include <omp.h>
#include <mpi.h>
#include "kSelect.hpp"

void localFn(std::vector<uint32_t> &arr, size_t k, size_t n)
{
    // find local min
    uint32_t min = arr[0];

#pragma omp parallel for reduction(min : min)
    for (size_t i = 1; i < n; i++)
    {
        if (arr[i] < min)
        {
            min = arr[i];
        }
    }

    // find local max
    uint32_t max = arr[0];

#pragma omp parallel for reduction(max : max)
    for (size_t i = 1; i < n; i++)
    {
        if (arr[i] > max)
        {
            max = arr[i];
        }
    }

    // broadcast local min and max
    MPI_Send(&min, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD);
}
