#include <limits.h>
#include <omp.h>
#include <mpi.h>
#include "kSearch.hpp"

inline bool lessEqualThan(const uint32_t &a, const uint32_t &b)
{
    return a <= b; // count the elements that are equal to the pivot as well, to avoid infinite loop
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

void findClosest(uint32_t &distance, const std::vector<uint32_t> &arr, const int &p, bool (*comp)(const uint32_t &, const uint32_t &))
{
    distance = INT_MAX; // if there is no element fulfilling the condition, return INT_MAX to increase its distance from the pivot
#pragma omp parallel for reduction(min : distance)
    for (size_t i = 0; i < arr.size(); i++)
    {
        if (comp(arr[i], p))
        {
            distance = abs(arr[i] - p) < distance ? abs(arr[i] - p) : distance; // smallest distance from the pivot that fullfills the condition
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

    // reduce only the local mins of the processes and not with all their elements
    MPI_Allreduce(&local.localMin, &min, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD);

    MPI_Allreduce(&local.localMax, &max, 1, MPI_UINT32_T, MPI_MAX, MPI_COMM_WORLD);

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

        MPI_Allreduce(&local.count, &countSumLess, 1, MPI_UINT32_T, MPI_SUM, MPI_COMM_WORLD);

        if (abs(prevP - p) == 1)
        {
            // if (!(prevP == (int)min && prevCountSumLess == 0)) // this is to be used if we assign the mean value to the pivot first, instead of the min
            // { // initialization of countSumLess is not correct

            if (prevCountSumLess < countSumLess && k > prevCountSumLess && k < countSumLess)
            {
                kth = p; // pivot with the greatest count of the two (and thus the largest pivot value) is in the array and is the kth element
                return;
            }
            // }

            else if (prevCountSumLess > countSumLess && k < prevCountSumLess && k > countSumLess)
            {
                kth = prevP;
                return;
            }
        }

        if ((countSumLess == k) || (countSumLess == (k - 1))) // if countSum is equal to k, then we have found the kth element
            break;
        // else
        //     std::erase_if(arr, [p, k, countSumLess, comp](uint32_t x)
        //                   { return (k - countSumLess) > 0 ? x < p : x > p; }); // check if arr size changes

        if (countSumLess == prevCountSumLess)
            prevCountSumLess = (k > countSumLess && prevP > p) ? prevCountSumLess + 1 : prevCountSumLess - 1; // change countSum instead of prevCountSum to balance
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

    uint32_t localDistance, distance;

    // find local potential kth element
    findClosest(localDistance, arr, p, comp);

    MPI_Allreduce(&localDistance, &distance, 1, MPI_UINT32_T, MPI_MIN, MPI_COMM_WORLD); // find the overall closest element to the pivot
                                                                                        // that fullfills the condition imposed by the countSum-k relation

    kth = (countSumLess == k) ? p - distance : p + distance; // calculate the kth element

    return;
}
