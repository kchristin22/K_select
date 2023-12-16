#include <mpi.h>
#include <omp.h>
#include "quickSelect.hpp"

void localSorting(localDataQuick &local, std::vector<uint32_t> &arr, const uint32_t start, const uint32_t end, const uint32_t p)
{
    uint32_t i = start, j = end;
    if (i > j)
    {
        local.count = i;
        return;
    }

    while (true)
    {
        while ((arr[i] <= p) && i < j)
            i++;
        while ((arr[j] > p) && i < j)
            j--;
        if (i < j)
            std::swap(arr[i], arr[j]);

        if (i == j || i == (j - 1))
            break;
    }

    local.count = (arr[i] <= p) ? i + 1 : i; // i and j equal
    return;
}

void quickSelect(uint32_t &kth, std::vector<uint32_t> &arr, const size_t k, const size_t n, const size_t np)
{
    localDataQuick local;
    uint32_t start = 0, end = arr.size() - 1;
    uint32_t p = arr[(rand() % (end - start + 1)) + start];
    uint32_t countSum = 0, prevCountSum = 0;
    int NumTasks, SelfTID;
    int master = 0, previous = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    MPI_Bcast(&p, 1, MPI_UINT32_T, master, MPI_COMM_WORLD);

    while (true)
    {
        localSorting(local, arr, start, end, p);

        if (master != SelfTID) // send local count
            MPI_Send(&local.count, 1, MPI_UINT32_T, master, 0, MPI_COMM_WORLD);
        else
        { // gather all local counts
            prevCountSum = countSum;
            countSum = local.count;
            for (int i = 0; i < NumTasks; i++)
            {
                if (i == master)
                    continue;
                uint32_t temp;
                MPI_Recv(&temp, 1, MPI_UINT32_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                countSum += temp;
            }
        }

        MPI_Bcast(&countSum, 1, MPI_UINT32_T, master, MPI_COMM_WORLD);
        // printf("countSum: %d\n", countSum);

        if (countSum == k || countSum == prevCountSum)
            break;
        else if (countSum > k)
        {
            start = 0;
            end = local.count; // check if -1 is needed, but then we have to check for the case where local.count == 0
        }
        else // we already know that countSum != k-1
        {
            end = arr.size() - 1;
            start = local.count;
        }

        for (int i = 0; i < NumTasks; i++) // round robin to find the next master, check at most all processes if necessary
        {
            if (master == SelfTID)
            {
                if (start > end)
                {                                      // check if I don't have elements in the range of the new pivot
                    master = (SelfTID + 1) % NumTasks; // assign the next process as a master
                    previous = SelfTID;
                }
                else
                {
                    previous = master;
                    p = arr[(rand() % (end - start + 1)) + start]; // new master has passed the test and can choose a new pivot
                    // printf("pivot chosen: %d\n", p);
                }
                for (int i = 0; i < NumTasks; i++)
                {
                    if (i == SelfTID)
                        continue;
                    MPI_Send(&master, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                    MPI_Send(&previous, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                }
            }
            else
            {
                previous = master;
                MPI_Recv(&master, 1, MPI_INT, master, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&previous, 1, MPI_INT, previous, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (master == previous)
            {
                MPI_Bcast(&p, 1, MPI_UINT32_T, master, MPI_COMM_WORLD);
                break;
            }
        }
    }

    kth = p;
    if (SelfTID == 0)
        printf("kth element quick 2: %d\n", kth);

    return;
}
