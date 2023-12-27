#include <algorithm>
#include <mpi.h>
#include <omp.h>
#include "quickSelect.hpp"

void localSorting(localDataQuick &local, std::vector<int> &arr, const size_t start, const size_t end, const int p)
{
    size_t i = start, j = end;
    if (i > j) // in this implementation this signifies that the previous pivot was even smaller than the current one
    {
        local.count = i;
        return;
    }
    else if (j == arr.size())
        j--;

    while (true)
    {
        while ((arr[i] <= p) && i <= j)
            i++; // the count (not the index, as it can reach the size of the array)
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

void parSorting(localDataQuick &local, std::vector<int> &arr, const size_t start, const size_t end, const int p)
{
    size_t i = start, j = end;
    if (i > j)
    {
        local.count = i;
        return;
    }
    else if (j == arr.size())
        j--;

    while (i < j)
    {
#pragma omp parallel sections shared(i, j) // the i and j variables are altered in parallel, the array is scanned in both directions simultaneously
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

void quickSelect(int &kth, std::vector<int> &arr, const size_t k, const size_t n, const size_t np)
{
    localDataQuick local;
    size_t start = 0, end = arr.size() - 1;
    local.rightMargin = end;
    int p = arr[(rand() % (end - start + 1)) + start], prevP = 0, prevPrevP = 0; // an alternation between pivots requires three pivot instances to be saved
    size_t countSum = 0, prevCountSum = 0;
    int NumTasks, SelfTID;
    int master = 0, previous = 0;

    MPI_Comm proc = np > 1 ? MPI_COMM_WORLD : MPI_COMM_SELF;

    bool gathered = np == 1 ? true : false; // already gathered prior to the call

    MPI_Comm_size(proc, &NumTasks); // number of processes in the communicator
    MPI_Comm_rank(proc, &SelfTID);

    MPI_Bcast(&p, 1, MPI_INT, master, proc); // broadcast the pivot

    while (true)
    {
        parSorting(local, arr, start, end, p); // partition the array based on the pivot

        prevPrevP = prevP;
        prevP = p;
        prevCountSum = countSum;

        MPI_Allreduce(&local.count, &countSum, 1, MPI_UNSIGNED_LONG, MPI_SUM, proc); // also broadcasts the number of elements <= p so all processes can do calculations with it

        if (countSum == k) // if countSum is equal to k, then the current pivot is the kth element
            break;
        else if (countSum > k)
        {
            start = local.leftMargin;
            end = local.count;               // search in the left part of the array
            local.rightMargin = local.count; // limit the search space from the right, as we know that the kth element is in the left part
        }
        else
        {
            start = local.count; // search in the right part of the array
            end = local.rightMargin;
            local.leftMargin = local.count; // limit the search space from the left, as we know that the kth element is in the right part
        }

        // gather the array if i) it is not gathered already, ii) the pivot is larger than the kth and iii) the size is small enough
        if (gathered == false && countSum > k && countSum < CACHE_SIZE / 2) // /2 to ensure that there's enough space to have two copies of the gathered array
        {
            arr.erase(arr.begin() + local.count + 1, arr.end()); // remove the elements that are larger than the pivot, so there's enough space to gather the elements
            std::vector<int> tempArr(countSum);                  // store local array
            std::vector<int> recvCount(np);
            std::vector<int> disp(np, 0);

            // store the amount of data each process will send
            MPI_Gather(&local.count, 1, MPI_UNSIGNED_LONG, recvCount.data(), 1, MPI_UNSIGNED_LONG, 0, proc);

            // calculate the displacement of each process' data
            for (size_t i = 1; i < np; i++)
                disp[i] = disp[i - 1] + recvCount[i - 1];

            // use Gatherv to gather different amounts of data from each process
            MPI_Gatherv(arr.data(), local.count, MPI_INT, tempArr.data(), recvCount.data(), disp.data(), MPI_INT, 0, proc);

            if (SelfTID != 0)
                return;

            arr.resize(countSum);
            arr = std::move(tempArr); // we lose the local array here, we can copy it if we want to keep it but maybe both won't fit in memory

            proc = MPI_COMM_SELF;
            gathered = true;
            NumTasks = 1;
            master = 0;

            // reset start and end
            start = 0;
            end = arr.size() - 1;
        }

        for (int i = 0; i < 2 * NumTasks; i++) // round robin to find the next master, check at most all processes if necessary
                                               // each potential master requires two iterations to check if it is the next master and choose a pivot or not
        {
            if (master == SelfTID)
            {
                if (end == 0 || start > end) // next pivot is out of range of the master
                {
                    master = (SelfTID + 1) % NumTasks; // assign the next process as a master
                    previous = SelfTID;
                }
                else
                { // new master has passed the test and can choose a new pivot
                    previous = master;
                    // we could shuflle the array, from start to end, before choosing the first fit value of this range
                    size_t tempEnd = end == arr.size() ? end - 1 : end;
                    for (size_t i = start; i < tempEnd; i++) // end <= arr.size() - 1
                    {
                        p = arr[i];
                        if (p != prevP && p != prevPrevP) // we want to choose a different pivot than the previous two to avoid infinite loops
                            break;

                        if (i == (tempEnd - 1)) // this master has only elements equal to the previous pivots
                            master = (SelfTID + 1) % NumTasks;
                    }
                }
            }
            MPI_Bcast(&master, 1, MPI_INT, previous, proc); // broadcast new master
            if (master == previous)                         // the master has passed the tests
            {
                MPI_Bcast(&p, 1, MPI_INT, master, proc); // broadcast new pivot
                break;
            }
            else
                previous = master; // the master has changed
        }
        if (p == prevPrevP || p == prevP) // only elements equal to the kth or the kth +1/-1 elements' values are left
        {
            if (std::clamp(k, prevCountSum, countSum) == k || std::clamp(k, countSum, prevCountSum) == k) // k is between the two countSums
                kth = prevCountSum < countSum ? prevP : prevPrevP;                                        // prevCountSum corresponds to the prevPrevP
            else if (countSum > k)                                                                        // both are larger than k
                kth = std::min(countSum, prevCountSum) == countSum ? prevP : prevPrevP;
            else // both are smaller than k
                kth = std::max(countSum, prevCountSum) == countSum ? prevP : prevPrevP;

            return;
        }
    }

    kth = p; // countSum == k

    return;
}