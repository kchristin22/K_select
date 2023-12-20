#include <algorithm>
#include <mpi.h>
#include <omp.h>
#include "quickSelect.hpp"

void localSorting(localDataQuick &local, std::vector<uint32_t> &arr, const uint32_t start, const uint32_t end, const uint32_t p)
{
    uint32_t i = start, j = end;
    if (i > j) // previous pivot was even smaller than the current one
    {
        local.count = i;
        return;
    }
    else if (j == arr.size())
        j--;

    while (true)
    {
        while ((arr[i] <= p) && i <= j)
            i++; // the count
        while ((arr[j] > p) && i < j)
            j--;
        if (i < j)
        {
            std::swap(arr[i], arr[j]);
        }
        else
            break;
    }

    local.count = i; // save the count and not the index
    return;
}

void quickSelect(int &kth, std::vector<uint32_t> &arr, const size_t k, const size_t n, const size_t np)
{
    localDataQuick local;
    uint32_t start = 0, end = arr.size() - 1;
    local.rightMargin = end;
    uint32_t p = arr[(rand() % (end - start + 1)) + start], prevP = 0, prevPrevP = 0; // an alternation between pivots requires three pivot instances to be saved
    uint32_t countSum = 0;
    int NumTasks, SelfTID;
    int master = 0, previous = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    MPI_Bcast(&p, 1, MPI_UINT32_T, master, MPI_COMM_WORLD);

    while (true)
    {
        // printf("p: %d, prevP: %d, prevPrevP: %d, prevCount: %d, count: %d\n", p, prevP, prevPrevP, prevCountSum, countSum);
        localSorting(local, arr, start, end, p);

        MPI_Reduce(&local.count, &countSum, 1, MPI_UINT32_T, MPI_SUM, master, MPI_COMM_WORLD);

        MPI_Bcast(&countSum, 1, MPI_UINT32_T, master, MPI_COMM_WORLD);

        if (countSum == k)
            break;
        else if (countSum > k)
        {
            start = local.leftMargin;
            end = local.count;
            local.rightMargin = local.count;
        }
        else // we already know that countSum != k-1
        {
            start = local.count;
            end = local.rightMargin;
            local.leftMargin = local.count;
        }

        // printf("start: %d, end: %d, proc: %d, p: %d, prevP: %d, countSum: %d\n", start, end, SelfTID, p, prevP, countSum);
        // MPI_Barrier(MPI_COMM_WORLD);

        prevPrevP = prevP;
        prevP = p;
        // prevCountSum = countSum;

        for (int i = 0; i < 2 * NumTasks; i++) // round robin to find the next master, check at most all processes if necessary
        {
            if (master == SelfTID)
            {
                // printf("master: %d, start: %d, end: %d\n", master, start, end);

                if (end == 0 || start > end)           // next pivot is out of range of the master
                {                                      // check if I don't have elements in the range of the new pivot
                    master = (SelfTID + 1) % NumTasks; // assign the next process as a master
                    previous = SelfTID;
                }
                else
                { // new master has passed the test and can choose a new pivot
                    previous = master;
                    // maybe shuflle the array here, from start to end
                    size_t tempEnd = end == arr.size() ? end - 1 : end;
                    for (size_t i = start; i < tempEnd; i++) // end <= arr.size() - 1
                    {
                        // p = arr[(rand() % (end - start + 1)) + start];
                        p = arr[i];
                        if (p != prevP && p != prevPrevP)
                            break;

                        if (i == (tempEnd - 1)) // this master has only elements equal to the previous pivot
                            master = (SelfTID + 1) % NumTasks;
                        // printf("pivot chosen: %d\n", p);
                    }
                }
                // for (int i = 0; i < NumTasks; i++)
                // {
                //     if (i == SelfTID)
                //         continue;
                //     MPI_Send(&master, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                //     MPI_Send(&previous, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                // }
            }
            MPI_Bcast(&master, 1, MPI_INT, previous, MPI_COMM_WORLD);
            if (master == previous)
            {
                MPI_Bcast(&p, 1, MPI_UINT32_T, master, MPI_COMM_WORLD);
                break;
            }
            else
                previous = master;
        }
        if (p == prevP) // only elements equal to the kth's value are left
            return;
        else if (p == prevPrevP) // only elements equal to the kth and the kth +1/-1 elements' values are left
        {
            kth = std::max(prevP, prevPrevP);
            return;
        }
    }

    kth = p;

    return;
}
