#include <mpi.h>
#include <omp.h>
#include "quickSelect.hpp"

void localSorting(localDataQuick &local, std::vector<uint32_t> &arr, const uint32_t start, const uint32_t end, const uint32_t p)
{
    uint32_t i = start, j = end;
    while (true)
    {
        while ((arr[i] <= p) && i <= j)
            i++;
        while ((arr[j] > p) && i < j)
            j--;
        if (i < j)
        {
            std::swap(arr[i], arr[j]);
            i++;
            j--;
        }
        else if (i == (j + 1) || i == j)
            break;
    }
    local.count = j;
}

void quickSelect(uint32_t kth, std::vector<uint32_t> &arr, const size_t k, const size_t n, const size_t np)
{
    localDataQuick local;
    uint32_t start = 0, end = n - 1;
    uint32_t p = arr[rand() % n];
    uint32_t countSum = 0;
    int NumTasks, SelfTID;

    MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    while (true)
    {
        localSorting(local, arr, start, end, p);

        if (SelfTID != 0) // send local count
            MPI_Send(&local.count, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD);
        else
        { // gather all local counts
            countSum = local.count;
            for (size_t i = 1; i < np; i++)
            {
                uint32_t temp;
                MPI_Recv(&temp, 1, MPI_UINT32_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                countSum += temp;
            }
        }

        if (countSum == (k - 1))
            break;

        // broadcast new pivot, but check if my count is 0 or n depending on the value of the next pivot,
        // if I can't be a master, I must signal for a process that fullfills the condition to take over
        else if (countSum < (k - 1))
        {
            start = 0;
            end = local.count; // check if -1 is needed
        }
        else
        {
            start = local.count + 1;
            end = arr.size() - 1;
        }

        p = arr[rand() % (end - start + 1) + start];
        MPI_Bcast(&p, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    }

    kth = p;
}
