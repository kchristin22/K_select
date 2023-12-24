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
    else if (p >= local.localMax)
        local.count = arr.size(); // all elements are less than or equal to the pivot
    else
    {
#pragma omp parallel for reduction(+ : count)
        for (size_t i = 0; i < arr.size(); i++)
        {
            if (comp(arr[i], p)) // count the elements that are less than or equal to the pivot in case of k < n/2, or greater than the pivot in case of k >= n/2
                count++;
        }

        if (k >= n / 2)
            count = arr.size() - count; // store the number of elements that are less than or equal to the pivot

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
            distance = abs(arr[i] - p) < distance ? abs(arr[i] - p) : distance; // store the smallest distance from the pivot that fullfills the condition
        }
    }

    return;
}

void kSearch(int &kth, std::vector<uint32_t> &arr, const size_t k, const size_t n, const size_t np)
{
    int SelfTID;
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    MPI_Comm proc = MPI_COMM_WORLD;

    std::vector<uint32_t> array;

    if (array.size() * np < CACHE_SIZE / 2) // check if the array fits in a single machine
    {
        array.resize(arr.size() * np);
        MPI_Gather(arr.data(), arr.size(), MPI_UINT32_T, array.data(), arr.size(), MPI_UINT32_T, 0, MPI_COMM_WORLD);

        if (SelfTID != 0)
            return;

        proc = MPI_COMM_SELF; // the MPI Communicator contains only the master now
    }
    else
        array = std::move(arr);

    localData local;
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

    int p = max, prevP = min; // use the max value as the first pivot as we know the number of elements <= p for it,
                              // and the min value as the previous pivot to calculate the median of the range
    int newP;
    uint32_t countSumLess = n, prevCountSumLess = 0;
    bool (*comp)(const uint32_t &, const uint32_t &) = (k < n / 2) ? lessEqualThan : greaterThan; // optimize the number of elements to count based on k's position

    while (true)
    {

        if (countSumLess == prevCountSumLess)
            prevCountSumLess = (k > countSumLess && prevP > p) ? prevCountSumLess + 1 : prevCountSumLess - 1; // change prevCountSum instead of countSum to not affect the next iteration of the algorithm
                                                                                                              // this change drives the sign of the difference between the two pivots:
                                                                                                              // the pivot decreases if the countSum is bigger than k, and increases if the countSum is smaller than k

        newP = p + ((k - countSumLess) * (prevP - p)) / (prevCountSumLess - countSumLess); // find new pivot through linear interpolation
        if (newP < (int)min)
            newP = min;
        else if (newP > (int)max)
            newP = max;
        else if (newP == p)
            newP = (k > countSumLess) ? p + 1 : p - 1;

        prevP = p;
        p = newP;
        prevCountSumLess = countSumLess;

        findLocalCount(local, array, p, comp, k, n); // find the local number of elements that are less than or equal to the pivot

        MPI_Allreduce(&local.count, &countSumLess, 1, MPI_UINT32_T, MPI_SUM, proc); // find the overall number and broadcast it so all processes can do calculations with it

        if ((countSumLess == k) || (countSumLess == (k - 1))) // if countSum is equal to k or k-1, then we can now find the kth element
            break;

        // else
        //     std::erase_if(arr, [p, k, countSumLess, comp](uint32_t x)
        //                   { return (k - countSumLess) > 0 ? x < p : x > p; }); // check if arr size changes

        if (abs(prevP - p) == 1) // check for the case where there are multiple instances of some values and the pivot alternates between them
        {
            // find the closest element to the pivot with the smallest count of the two
            if (prevCountSumLess < countSumLess && k > prevCountSumLess && k < countSumLess)
            {
                p = prevP;
                countSumLess = prevCountSumLess;
                break;
            }
            else if (prevCountSumLess > countSumLess && k < prevCountSumLess && k > countSumLess)
                break;
        }
    }

    if ((countSumLess >= k))
        comp = lessEqualThan; // the next element less than or equal to the pivot is the kth element
    else if (countSumLess == (k - 1))
        comp = greaterThan; // the next element bigger than the pivot is the kth element

    uint32_t localDistance, distance;

    // find local potential kth element
    findClosest(localDistance, array, p, comp);

    MPI_Allreduce(&localDistance, &distance, 1, MPI_UINT32_T, MPI_MIN, proc); // find the overall closest element to the pivot
                                                                              // that fullfills the condition imposed by the countSum-k relation

    kth = (countSumLess == k) ? p - distance : p + distance; // calculate the kth element from its distance from the pivot

    return;
}
