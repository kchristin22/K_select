#include <mpi.h>
#include <omp.h>
#include "heurQuickSelect.hpp"

inline bool lessEqualThan(const uint32_t &a, const uint32_t &b)
{
    return a <= b; // count the elements that are equal to the pivot as well, to avoid infinite loop
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

inline void setComp(bool (*&comp)(const uint32_t &, const uint32_t &), const size_t k, const size_t n, const size_t countSum)
{
    if (countSum == (k + 1))
        comp = lessThan; // the next element less than the pivot is the kth element
    else if (countSum == k)
        comp = lessEqualThan; // the next element less than or equal to the pivot is the kth element
    else                      // ((countSum == (k - 1))
        comp = greaterThan;   // the next element bigger than the pivot is the kth element

    return;
}

void findLocalMinMax(localDataHeurQuick &local, const std::vector<uint32_t> &arr)
{
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

    // printf("local min: %d\n", local.localMin);

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

    // printf("localMax: %d\n", local.localMax);

    return;
}

void heurlocalSorting(localDataHeurQuick &local, std::vector<uint32_t> &arr, const uint32_t start, const uint32_t end, const uint32_t p)
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
    local.count = i;

    return;
}

void heurfindClosest(uint32_t &closest, const std::vector<uint32_t> &arr, const uint32_t &p, bool (*comp)(const uint32_t &, const uint32_t &))
{ // check for custom reduction with template functions
    closest = arr[0];
#pragma omp parallel for
    for (size_t i = 1; i < arr.size(); i++)
    {
        if (comp(arr[i], p))
        {
#pragma omp atomic write
            closest = comp(arr[i], closest) ? closest : arr[i];
        }
    }

    return;
}

void heurQuickSelect(uint32_t kth, std::vector<uint32_t> &arr, const size_t k, const size_t n, const size_t np)
{
    // find local min
    localDataHeurQuick local;
    findLocalMinMax(local, arr);

    uint32_t min = local.localMin, max = local.localMax;

    int SelfTID;
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    if (SelfTID != 0) // send local min and max
        MPI_Send(&local.localMin, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD);
    else
    { // gather all local min and max
        for (size_t i = 1; i < np; i++)
        {
            uint32_t temp;
            MPI_Recv(&temp, 1, MPI_UINT32_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (temp < min)
                min = temp;
        }

        // printf("whole min: %d\n", min);
    }

    MPI_Bcast(&min, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    // printf("whole min %d, process %d\n", min, SelfTID);

    if (SelfTID != 0) // send local min and max
        MPI_Send(&max, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD);
    else
    { // gather all local min and max
        for (size_t i = 1; i < np; i++)
        {
            uint32_t temp;
            MPI_Recv(&temp, 1, MPI_UINT32_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (temp > max)
                max = temp;
        }
        // printf("whole max: %d\n", max);
    }

    MPI_Bcast(&max, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);
    // printf("whole max %d, process %d\n", max, SelfTID);

    if (min == max)
    {
        kth = min;
        printf("kth element: %d\n", kth);
        return;
    }

    uint32_t p = min - 1;
    uint32_t countSum = 0;
    uint32_t start = 0, end = arr.size() - 1;

    while (true)
    {
        p = (p + ((max - min + 1) * (k - countSum)) / n) > 0 ? p + ((max - min + 1) * (k - countSum)) / n : min; // find pivot

        heurlocalSorting(local, arr, start, end, p); // find local count

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

        MPI_Bcast(&countSum, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

        if (k == countSum || countSum == (k - 1) || countSum == (k + 1)) // if countSum is equal to k, then we have found the kth element
            break;
        else if (countSum < k)
        {
            start = 0;
            end = local.count;
        }
        else
        {
            start = local.count + 1;
            end = arr.size() - 1;
        }
    }

    bool (*comp)(const uint32_t &, const uint32_t &); // set comp based on countSum
    setComp(comp, k, n, countSum);

    heurfindClosest(kth, arr, p, comp);

    if (SelfTID != 0)
        MPI_Send(&kth, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD);
    else
    {
        for (size_t i = 1; i < np; i++)
        {
            uint32_t temp;
            MPI_Recv(&temp, 1, MPI_UINT32_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (comp(temp, p))
                kth = comp(kth, temp) ? temp : kth;
        }
        printf("kth element quick: %d\n", kth);
    }

    return;
}
