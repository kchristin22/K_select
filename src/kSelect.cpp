#include <omp.h>
#include <mpi.h>
#include "kSelect.hpp"

void localFn(std::vector<uint32_t> &arr, size_t k, size_t n)
{
    // find local min
    uint32_t min = arr[0];

#pragma omp parallel for reduction(min : min)
    for (size_t i = 1; i < arr.size(); i++)
    {
        if (arr[i] < min)
        {
            min = arr[i];
        }
    }

    printf("min: %d\n", min);

    // find local max
    uint32_t max = arr[0];

#pragma omp parallel for reduction(max : max)
    for (size_t i = 1; i < arr.size(); i++)
    {
        if (arr[i] > max)
        {
            max = arr[i];
        }
    }

    printf("max: %d\n", max);

    int SelfTID;
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    if (SelfTID != 0) // send local min and max
        MPI_Send(&max, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD);
    else
    { // gather all local min and max
        for (int i = 1; i < 4; i++)
        {
            uint32_t temp;
            MPI_Recv(&temp, 1, MPI_UINT32_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (temp > max)
                max = temp;
        }

        printf("whole max: %d\n", max);
    }
}
