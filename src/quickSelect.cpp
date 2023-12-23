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

void parSorting(localDataQuick &local, std::vector<uint32_t> &arr, const uint32_t start, const uint32_t end, const uint32_t p)
{
    uint32_t i = start, j = end;
    if (i > j)
    {
        local.count = i;
        return;
    }
    else if (j == arr.size())
        j--;

    while (i < j)
    {
#pragma omp parallel sections shared(i, j)
        {
#pragma omp section
            {
                while ((arr[i] <= p) && i <= j)
                    i++; // the count
            }

#pragma omp section
            {
                while ((arr[j] > p) && i < j)
                    j--;
            }
        }

        if (i < j)
            std::swap(arr[i], arr[j]);
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

    MPI_Comm proc = MPI_COMM_WORLD;

    std::vector<uint32_t> array;
    bool gathered = false;

    if (array.size() * np < CACHE_SIZE / 2)
    {
        array.resize(arr.size() * np);
        MPI_Gather(arr.data(), arr.size(), MPI_UINT32_T, array.data(), arr.size(), MPI_UINT32_T, 0, proc);

        if (SelfTID != 0)
            return;

        proc = MPI_COMM_SELF;
        gathered = true;
    }
    else
        array = std::move(arr);

    MPI_Comm_size(proc, &NumTasks);
    MPI_Comm_rank(proc, &SelfTID);

    MPI_Bcast(&p, 1, MPI_UINT32_T, master, proc);

    while (true)
    {
        parSorting(local, array, start, end, p);

        MPI_Reduce(&local.count, &countSum, 1, MPI_UINT32_T, MPI_SUM, master, proc);

        MPI_Bcast(&countSum, 1, MPI_UINT32_T, master, proc);

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

        if (gathered == false && countSum > k && countSum < CACHE_SIZE / 2) // work on cache condition
        {
            std::vector<uint32_t> tempArr(countSum); // store local array
            std::vector<int> recvCount(np);
            std::vector<int> disp(np);

            MPI_Gather(&local.count, 1, MPI_INT, recvCount.data(), 1, MPI_UINT32_T, 0, proc);

            for (size_t i = 1; i < np; i++)
                disp[i] = disp[i - 1] + recvCount[i - 1];

            MPI_Gatherv(array.data(), local.count, MPI_UINT32_T, tempArr.data(), recvCount.data(), disp.data(), MPI_UINT32_T, 0, proc);

            if (SelfTID != 0)
                return;

            array.resize(countSum);
            array = std::move(tempArr); // we lose the local array here, we can copy it if we want to keep it but maybe both won't fit in memory

            proc = MPI_COMM_SELF;
            gathered = true;
            NumTasks = 1;
            master = 0;

            // reset start and end
            start = 0;
            end = array.size() - 1;

            printf("arr left: ");
            for (size_t i = 0; i < array.size(); i++)
            {
                printf("%d, ", array[i]);
            }
            printf("\n");
        }

        prevPrevP = prevP;
        prevP = p;

        for (int i = 0; i < 2 * NumTasks; i++) // round robin to find the next master, check at most all processes if necessary
        {
            if (master == SelfTID)
            {
                if (end == 0 || start > end)           // next pivot is out of range of the master
                {                                      // check if I don't have elements in the range of the new pivot
                    master = (SelfTID + 1) % NumTasks; // assign the next process as a master
                    previous = SelfTID;
                }
                else
                { // new master has passed the test and can choose a new pivot
                    previous = master;
                    // we could shuflle the array, from start to end, before choosing the first fit value of the array
                    size_t tempEnd = end == array.size() ? end - 1 : end;
                    for (size_t i = start; i < tempEnd; i++) // end <= arr.size() - 1
                    {
                        // p = array[(rand() % (end - start + 1)) + start];
                        p = array[i];
                        if (p != prevP && p != prevPrevP)
                            break;

                        if (i == (tempEnd - 1)) // this master has only elements equal to the previous pivot
                            master = (SelfTID + 1) % NumTasks;
                    }
                }
            }
            MPI_Bcast(&master, 1, MPI_INT, previous, proc);
            if (master == previous)
            {
                MPI_Bcast(&p, 1, MPI_UINT32_T, master, proc);
                break;
            }
            else
                previous = master;
        }
        if (p == prevP) // only elements equal to the kth's value are left
            break;
        else if (p == prevPrevP) // only elements equal to the kth and the kth +1/-1 elements' values are left
        {
            kth = std::max(prevP, prevPrevP);
            return;
        }
    }

    kth = p;

    return;
}
