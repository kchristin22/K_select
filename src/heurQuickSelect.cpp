#include <limits.h>
#include <mpi.h>
#include <omp.h>
#include "heurQuickSelect.hpp"

inline bool lessEqualThan(const uint32_t &a, const uint32_t &b)
{
    return a <= b;
}

inline bool greaterEqualThan(const uint32_t &a, const uint32_t &b)
{
    return a >= b;
}

inline bool lessThan(const uint32_t &a, const uint32_t &b)
{
    return a < b;
}

inline bool greaterThan(const uint32_t &a, const uint32_t &b)
{
    return a > b;
}

void findLocalMinMax(localDataHeurQuick &local, const std::vector<uint32_t> &arr)
{
    // find local min
    uint32_t localmin = arr[0];

#pragma omp parallel for reduction(min : localmin)
    for (size_t i = 1; i < arr.size(); i++)
    {
        if (arr[i] < localmin)
        {
            localmin = arr[i];
        }
    }

    local.localMin = localmin;

    // find local max
    uint32_t localmax = arr[0];

#pragma omp parallel for reduction(max : localmax)
    for (size_t i = 1; i < arr.size(); i++)
    {
        if (arr[i] > localmax)
        {
            localmax = arr[i];
        }
    }

    local.localMax = localmax;

    return;
}

void heurlocalSorting(localDataHeurQuick &local, std::vector<uint32_t> &arr, const size_t start, const size_t end, const uint32_t p)
{
    uint32_t i = start, j = end;
    if (i > j) // in this implementation this signifies that the previous pivot was even smaller than the current one
    {
        local.count = i;
        return;
    }
    else if (j == arr.size())
        j--;

    while (i < j)
    {
        while ((arr[i] <= p) && i <= j)
            i++; // the count (not the index, as it can reach the size of the array)
        while ((arr[j] > p) && i < j)
            j--;
        if (i < j)
            std::swap(arr[i], arr[j]);
    }

    local.count = i; // save the count and not the index
    return;
}

void heurParSorting(localDataHeurQuick &local, std::vector<uint32_t> &arr, const size_t start, const size_t end, const uint32_t p)
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

inline void setComp(bool (*&comp)(const uint32_t &, const uint32_t &), const size_t k, const uint32_t countSum)
{
    if (countSum >= k)
        comp = lessEqualThan; // the next element less than or equal to the pivot is the kth element
    else if (countSum == (k - 1))
        comp = greaterThan; // the next element bigger than the pivot is the kth element

    return;
}

void findClosest(uint32_t &distance, const std::vector<uint32_t> &arr, const size_t start, const size_t end, const uint32_t &p, bool (*comp)(const uint32_t &, const uint32_t &))
{
    distance = INT_MAX; // if there is no element fulfilling the condition, return INT_MAX to increase its distance from the pivot
#pragma omp parallel for reduction(min : distance)
    for (size_t i = start; i <= end; i++) // end refers to an index inside the array (see localSorting)
    {
        if (comp(arr[i], p))
        {
            distance = abs(arr[i] - p) < distance ? abs(arr[i] - p) : distance; // smallest distance from the pivot that fullfills the condition
        }
    }

    return;
}

void heurQuickSelect(uint32_t &kth, std::vector<uint32_t> &arr, const size_t k, const size_t n, const size_t np)
{
    int SelfTID;
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    MPI_Comm proc = MPI_COMM_WORLD;

    std::vector<uint32_t> array;
    bool gathered = false;

    if (array.size() * np < CACHE_SIZE / 2) // check if the array fits in a single machine
    {
        array.resize(arr.size() * np);
        MPI_Gather(arr.data(), arr.size(), MPI_UINT32_T, array.data(), arr.size(), MPI_UINT32_T, 0, MPI_COMM_WORLD); // gather all the arrays in the master process

        if (SelfTID != 0)
            return;

        proc = MPI_COMM_SELF; // the MPI Communicator contains only the master now
        gathered = true;      // set flag to true to avoid re-gathering the array
    }
    else
        array = std::move(arr);

    localDataHeurQuick local;
    findLocalMinMax(local, array);

    uint32_t min = local.localMin, max = local.localMax;

    // reduce the local mins to find the overall min
    MPI_Allreduce(&local.localMin, &min, 1, MPI_UINT32_T, MPI_MIN, proc);

    // reduce the local maxes to find the overall max
    MPI_Allreduce(&local.localMax, &max, 1, MPI_UINT32_T, MPI_MAX, proc);

    if (min == max) // the array contains only one value
    {
        kth = min;
        return;
    }
    else if (k == 1) // the min value is the first element of the sorted array
    {
        kth = min;
        return;
    }
    else if (k == n) // the max value is the last element of the sorted array
    {
        kth = max;
        return;
    }

    uint32_t p = max, prevP = min; // use the max value as the first pivot as we know the number of elements <= p for it,
                                   // and the min value as the previous pivot to calculate the median of the range
    uint32_t newP;
    uint32_t countSum = n, prevCountSum = 0;
    uint32_t start = 0, end = array.size() - 1;
    local.rightMargin = array.size() - 1;

    while (true)
    {
        if (countSum == prevCountSum)
            prevCountSum = (k > countSum && prevP > p) ? prevCountSum + 1 : prevCountSum - 1;

        // find new pivot through linear interpolation, and make sure it's not out of bounds
        newP = (int)(p + ((k - countSum) * (prevP - p)) / (prevCountSum - countSum)) > 0 ? p + ((k - countSum) * (prevP - p)) / (prevCountSum - countSum) : min;
        if (newP > max)
            newP = max;
        else if (newP == p)
            newP = (k > countSum) ? p + 1 : p - 1;

        prevP = p;
        p = newP;
        prevCountSum = countSum;

        heurParSorting(local, array, start, end, p); // find local count

        MPI_Allreduce(&local.count, &countSum, 1, MPI_UINT32_T, MPI_SUM, proc); // also broadcasts the number of elements <= p so all processes can do calculations with it

        if (countSum >= k)
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

        if (k == countSum || countSum == (k - 1)) // if countSum is equal to k or k-1, then we can now find the kth element
            break;

        if (abs(prevP - p) == 1) // check for the case where there are multiple instances of some values and the pivot alternates between them
        {
            // find the closest element to the pivot with the largest count of the two
            if (prevCountSum < countSum && k > prevCountSum && k < countSum)
                break;

            else if (prevCountSum > countSum && k < prevCountSum && k > countSum)
            {
                p = prevP;
                countSum = prevCountSum;
                break;
            }
        }

        // gather the array if i) it is not gathered already, ii) the pivot is larger than the kth and iii) the size is small enough
        if (gathered == false && countSum > k && countSum < CACHE_SIZE / 2)
        {
            std::vector<uint32_t> tempArr(countSum); // store local array
            std::vector<int> recvCount(np);
            std::vector<int> disp(np, 0);

            // store the amount of data each process will send
            MPI_Gather(&local.count, 1, MPI_INT, recvCount.data(), 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

            // calculate the displacement of each process' data
            for (size_t i = 1; i < np; i++)
                disp[i] = disp[i - 1] + recvCount[i - 1];

            // use Gatherv to gather different amounts of data from each process
            MPI_Gatherv(array.data(), local.count, MPI_UINT32_T, tempArr.data(), recvCount.data(), disp.data(), MPI_UINT32_T, 0, MPI_COMM_WORLD);

            if (SelfTID != 0)
                return;

            array.resize(countSum);
            array = std::move(tempArr); // we lose the local array here, we can copy it if we want to keep it but maybe both won't fit in memory

            proc = MPI_COMM_SELF;
            gathered = true;

            // reset start and end
            start = 0;
            end = array.size() - 1;
        }
    }

    bool (*comp)(const uint32_t &, const uint32_t &);
    setComp(comp, k, countSum); // set comp based on countSum and k's relation

    uint32_t localDistance, distance;

    // make sure that start and end are inside the array
    start %= array.size();
    end %= array.size(); // end refers to an index inside the array (see localSorting)

    // find local potential kth element
    findClosest(localDistance, array, start, end, p, comp);

    MPI_Allreduce(&localDistance, &distance, 1, MPI_UINT32_T, MPI_MIN, proc); // find the overall closest element to the pivot
                                                                              // that fullfills the condition imposed by the countSum-k relation

    kth = (countSum >= k) ? p - distance : p + distance; // calculate the kth element from its distance from the pivot

    return;
}
