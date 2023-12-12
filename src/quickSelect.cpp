#include <mpi.h>
#include <omp.h>
#include "quickSelect.hpp"

void localSorting(localDataQuick &local, std::vector<uint32_t> &arr, const uint32_t start, const uint32_t end, const uint32_t p)
{
    uint32_t i = start, j = end;

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

    local.count = (arr[i] <= p) ? i + 1 - start : i - start; // i and j equal
    return;
}

void quickSelect(uint32_t kth, std::vector<uint32_t> &arr, const size_t k, const size_t n, const size_t np)
{
    localDataQuick local;
    uint32_t start = 0, end = arr.size() - 1;
    uint32_t p = arr[(rand() % (end - start + 1)) + start];
    uint32_t countSum = 0;
    int NumTasks, SelfTID;
    int master = 0, previous = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
    printf("NumTasks: %d\n", NumTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    MPI_Bcast(&p, 1, MPI_UINT32_T, master, MPI_COMM_WORLD);

    while (true)
    {
        // printf("pivot: %d\n", p);
        localSorting(local, arr, start, end, p);

        if (master != SelfTID) // send local count
            MPI_Send(&local.count, 1, MPI_UINT32_T, master, 0, MPI_COMM_WORLD);
        else
        { // gather all local counts
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

        if (countSum == k)
            break;
        else if (countSum > k)
        {
            start = 0;
            end = local.count; // check if -1 is needed, but then we have to check for the case where local.count == 0
        }
        else // we already know that countSum != k-1
        {
            end = arr.size() - 1;
            start = (local.count == arr.size()) ? end : local.count;
        }

        printf("p: %d, proc: %d, start: %d, end: %d\n", p, SelfTID, start, end);

        for (int i = 0; i < NumTasks; i++) // round robin to find the next master, check at most all processes if necessary
        {
            if (master == SelfTID)
            {
                // printf("master, start, end: %d %d %d\n", master, start, end);
                if (start == end)
                {                                      // check if I don't have elements in the range of the new pivot
                    master = (SelfTID + 1) % NumTasks; // assign the next process as a master
                    printf("new master: %d\n", master);
                    previous = SelfTID;
                }
                else
                {
                    // printf("choosing pivot, master: %d\n", master);
                    previous = SelfTID;
                    p = arr[(rand() % (end - start + 1)) + start]; // new master has passed the test and can choose a new pivot
                    printf("pivot chosen: %d\n", p);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            // printf("master: %d, previous: %d, proc: %d\n", master, previous, SelfTID);
            MPI_Bcast(&master, 1, MPI_INT, previous, MPI_COMM_WORLD); // other procs are waiting for the previous to the previous master
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Bcast(&previous, 1, MPI_INT, master, MPI_COMM_WORLD);
            printf("master: %d, previous: %d, proc: %d\n", master, previous, SelfTID);
            MPI_Barrier(MPI_COMM_WORLD);
            if (master == previous)
            { // new master has passed the test
                // printf("broadcasting pivot: %d\n", p);
                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Bcast(&p, 1, MPI_UINT32_T, master, MPI_COMM_WORLD);
                break;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    kth = p;
    if (SelfTID == 0)
        printf("kth element quick 2: %d\n", kth);

    return;
}
