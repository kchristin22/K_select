#include <omp.h>
#include <mpi.h>
#include "kSelect.hpp"

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

inline bool checkCond(const size_t k, const size_t n, const uint32_t countSum)
{
    if (k < n / 2)
        return ((countSum == k) || (countSum == (k - 1)) || (countSum == (k + 1))) ? true : false;
    else
        return ((countSum == (n - k)) || (countSum == (n - k - 1)) || (countSum == (n - k + 1))) ? true : false;
}

inline void setComp(bool (*&comp)(const uint32_t &, const uint32_t &), const size_t k, const size_t n, const size_t countSum)
{
    if ((countSum == (k + 1)) || (countSum == (n - k - 1)))
        comp = lessThan; // the next element less than the pivot is the kth element
    else if ((countSum == k) || (countSum == (n - k)))
        comp = lessEqualThan; // the next element less than or equal to the pivot is the kth element
    else                      // ((countSum == (k - 1)) || (countSum == (n - k + 1)))
        comp = greaterThan;   // the next element bigger than the pivot is the kth element

    return;
}

void findLocalMinMax(localData &local, const std::vector<uint32_t> &arr)
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

inline void findLocalCount(localData &local, const std::vector<uint32_t> &arr, const uint32_t &p, bool (*comp)(const uint32_t &, const uint32_t &), const size_t k, const size_t n)
{
    uint32_t count = 0;

    // check conditions of pivot being out of local range, where we can instantly know the value of count
    if (((k < n / 2) && (p < local.localMin)) || ((k >= n / 2) && (p > local.localMax))) // if pivot is less than local min, when we count the less than / greater than local max, when we count the greater than
        count = 0;                                                                       // no elements are less than the pivot / greater than the pivot
    else if (((k < n / 2) && (p >= local.localMax)) || ((k >= n / 2) && (p < local.localMin)))
        count = arr.size();
    else
    {
#pragma omp parallel for reduction(+ : count)
        for (size_t i = 0; i < arr.size(); i++) // count elements less than or greater than pivot, depending on the percentile of k
        {
            if (comp(arr[i], p))
                count++;
        }
    }

    local.count = count;

    return;
}

void findClosest(uint32_t &closest, const std::vector<uint32_t> &arr, const uint32_t &p, bool (*comp)(const uint32_t &, const uint32_t &))
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

void localFn(uint32_t kth, std::vector<uint32_t> &arr, const size_t k, const size_t n, const size_t np)
{
    // find local min
    localData local;
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

    bool (*comp)(const uint32_t &, const uint32_t &) = (k < n / 2) ? lessEqualThan : greaterThan; // optimize counting of elements based on the percentile of k

    uint32_t p = min - 1;
    uint32_t countSum = 0;

    while (true)
    {
        p = p + (max - min + 1) * (k - countSum) / n; // find pivot
        findLocalCount(local, arr, p, comp, k, n); // find local count

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

        if (checkCond(k, n, countSum)) // if countSum is equal to k, then we have found the kth element
            break;
        else
            std::erase_if(arr, [p, k, countSum, comp](uint32_t x)
                          { return (k - countSum) ? x > p : x < p; });
    }

    setComp(comp, k, n, countSum);

    // find local potential kth element
    findClosest(kth, arr, p, comp);

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
        printf("kth element: %d\n", kth);
    }

    return;
}
