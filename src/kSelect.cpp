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

void findLocalMinMax(localData &local, const std::vector<uint32_t> &arr)
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

inline void findLocalCount(localData &local, const std::vector<uint32_t> &arr, const uint32_t &p, bool (*comp)(const uint32_t &, const uint32_t &), const size_t k, const size_t n)
{
    uint32_t count = 0;

    if (p < local.localMin)
        local.count = 0; // no elements are less than or equal to the pivot
    else if (p > local.localMax)
        local.count = arr.size(); // all elements are less than or equal to the pivot
    else
    {
#pragma omp parallel for reduction(+ : count)
        for (size_t i = 0; i < arr.size(); i++) // count elements less than or eqaul to the pivot
        {
            if (comp(arr[i], p))
                count++;
        }

        if (k >= n / 2)
            count = arr.size() - count; // save the less than or equal to the pivot

        local.count = count;
    }

    return;
}

void findClosest(int &closest, const std::vector<uint32_t> &arr, const int &p, bool (*comp)(const uint32_t &, const uint32_t &))
{ // check for custom reduction with template functions
    closest = arr[0];
#pragma omp parallel for
    for (size_t i = 1; i < arr.size(); i++)
    {
        if (comp(arr[i], p))
        {
#pragma omp atomic write
            closest = comp(closest, p) && comp(arr[i], closest) ? closest : arr[i];
        }
    }

    return;
}

void kSearch(int &kth, std::vector<uint32_t> &arr, const size_t k, const size_t n, const size_t np)
{
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
    }

    MPI_Bcast(&min, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

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
    }

    MPI_Bcast(&max, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    if (min == max)
    {
        kth = min;
        return;
    }
    else if (k == 1)
    {
        kth = min;
        return;
    }
    else if (k == n)
    {
        kth = max;
        return;
    }

    int p = min, prevP = max;
    int newP;
    uint32_t countSumLess = 0, prevCountSumLess = n;
    bool (*comp)(const uint32_t &, const uint32_t &) = (k < n / 2) ? lessEqualThan : greaterThan;

    while (true)
    {

        findLocalCount(local, arr, p, comp, k, n); // find local count

        if (SelfTID != 0) // send local count
            MPI_Send(&local.count, 1, MPI_UINT32_T, 0, 0, MPI_COMM_WORLD);
        else
        { // gather all local counts
            countSumLess = local.count;
            for (size_t i = 1; i < np; i++)
            {
                uint32_t temp;
                MPI_Recv(&temp, 1, MPI_UINT32_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                countSumLess += temp;
            }
        }

        MPI_Bcast(&countSumLess, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

        if (abs(prevP - p) == 1)
        {
            // if (!(prevP == (int)min && prevCountSumLess == 0))
            // { // initialization of countSumLess is not correct

            if (lessThan(prevCountSumLess, countSumLess) && k > prevCountSumLess && k < countSumLess)
            {
                kth = p; // pivot with the greatest count of the two (and thus the largest pivot value) is in the array and is the kth element
                return;
            }
            // else if (greaterThan(prevCountSumLess, countSumLess) && k < prevCountSumLess && k > countSumLess)
            // {
            //     kth = prevP; // pivot with the greatest count of the two (and thus the largest pivot value) is in the array and is the kth element
            //     return;
            // }
            // }
        }

        if ((countSumLess == k) || (countSumLess == (k - 1))) // if countSum is equal to k, then we have found the kth element
            break;
        // else
        //     std::erase_if(arr, [p, k, countSumLess, comp](uint32_t x)
        //                   { return (k - countSumLess) > 0 ? x < p : x > p; }); // check if arr size changes

        if (countSumLess == prevCountSumLess)
            countSumLess = (k > countSumLess && prevP > p) ? countSumLess - 1 : countSumLess + 1; // change countSum instead of prevCountSum to balance
                                                                                                  // the numerator and denominator of the pivot formula
                                                                                                  // cannot harm the algorithm, since the relationship of countSum and k will not change
                                                                                                  // (whichever is bigger will remain bigger)

        newP = p + ((k - countSumLess) * (prevP - p)) / (prevCountSumLess - countSumLess); // find pivot
        if (newP < (int)min)
            newP = min;
        else if (newP > (int)max)
            newP = max;
        else if (newP == p)
            newP = (k > countSumLess) ? p + 1 : p - 1;

        prevP = p;
        p = newP;
        prevCountSumLess = countSumLess;
    }

    if ((countSumLess == k))
        comp = lessEqualThan; // the next element less than or equal to the pivot is the kth element
    else if (countSumLess == (k - 1))
        comp = greaterThan; // the next element bigger than the pivot is the kth element

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
            {
                if (!comp(kth, p))
                    kth = temp;
                else
                    kth = abs(p - temp) < abs(p - kth) ? temp : kth;
            }
        }
    }

    return;
}
