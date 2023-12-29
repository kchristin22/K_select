#include <limits.h>
#include <omp.h>
#include <mpi.h>
#include "kSearch.hpp"

inline bool lessEqualThan(const int &a, const int &b)
{
    return a <= b;
}

inline bool lessThan(const int &a, const int &b)
{
    return a < b;
}

inline bool greaterThan(const int &a, const int &b)
{
    return a > b;
}

void findLocalMinMax(localData &local, const std::vector<int> &arr)
{
    // find local min
    int localmin = arr[0];

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
    int localmax = arr[0];

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

inline void findLocalCount(localData &local, const std::vector<int> &arr, const int &p, bool (*comp)(const int &, const int &), const size_t k, const size_t n)
{
    size_t count = 0;

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

void findClosest(uint32_t &distance, const std::vector<int> &arr, const int &p, bool (*comp)(const int &, const int &))
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

void kSearch(int &kth, std::vector<int> &arr, const size_t k, const size_t n, const size_t np)
{

    MPI_Comm proc = np > 1 ? MPI_COMM_WORLD : MPI_COMM_SELF;

    int SelfTID;
    MPI_Comm_rank(proc, &SelfTID);

    localData local;
    findLocalMinMax(local, arr);

    int min = local.localMin, max = local.localMax;

    // reduce the local mins to find the overall min
    MPI_Allreduce(&local.localMin, &min, 1, MPI_INT, MPI_MIN, proc);

    // reduce the local maxes to find the overall max
    MPI_Allreduce(&local.localMax, &max, 1, MPI_INT, MPI_MAX, proc);

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
    size_t countSumLess = n, prevCountSumLess = 0;
    bool (*comp)(const int &, const int &) = (k < n / 2) ? lessEqualThan : greaterThan; // optimize the number of elements to count based on k's position

    while (true)
    {
        if (countSumLess == prevCountSumLess)
            prevCountSumLess = (k > countSumLess && prevP > p) ? prevCountSumLess + 1 : prevCountSumLess - 1; // change prevCountSum instead of countSum to not affect the next iteration of the algorithm
                                                                                                              // this change drives the sign of the difference between the two pivots:
                                                                                                              // the pivot decreases if the countSum is bigger than k, and increases if the countSum is smaller than k

        // find new pivot through linear interpolation, and make sure it's not out of bounds
        int fraction = (((int)k - countSumLess) * (prevP - p)) / ((int)prevCountSumLess - countSumLess);
        newP = fraction < 0 && abs(fraction) > p ? min : p + fraction;
        if (newP > max)
            newP = max;
        if (newP == p)
            newP = (k > countSumLess) ? p + 1 : p - 1;
        else if (newP == prevP)
            newP = (k > prevCountSumLess) ? prevP + 1 : prevP - 1; // avoid looping between two pivots

        prevP = p;
        p = newP;
        prevCountSumLess = countSumLess;

        findLocalCount(local, arr, p, comp, k, n); // find the local number of elements that are less than or equal to the pivot

        MPI_Allreduce(&local.count, &countSumLess, 1, MPI_UNSIGNED_LONG, MPI_SUM, proc); // find the overall number and broadcast it so all processes can do calculations with it

        // printf("p: %d, prevP: %d, prevCountSum: %ld, countSum: %ld\n", p, prevP, prevCountSumLess, countSumLess);

        if (abs(prevP - p) == 1) // check for the case where there are multiple instances of some values and the pivot alternates between them
        {
            // if two consequential pivots showcase great difference in countSum, then the largest is surely in the array and is the kth element
            if (prevCountSumLess < countSumLess && k > prevCountSumLess && k < countSumLess)
            {
                kth = p; 
                return;
            }
            else if (prevCountSumLess > countSumLess && k < prevCountSumLess && k > countSumLess)
            {
                kth = prevP;
                return;
            }
        }

        if ((countSumLess == k) || (countSumLess == (k - 1))) // if countSum is equal to k or k-1, then we can now find the kth element
            break;
        else if (countSumLess > k)
            std::erase_if(arr, [p](int x)
                          { return x > p; }); // if countSum is bigger than k, then the kth element is smaller than the pivot
    }

    if ((countSumLess >= k))
        comp = lessEqualThan; // the next element less than or equal to the pivot is the kth element
    else if (countSumLess == (k - 1))
        comp = greaterThan; // the next element bigger than the pivot is the kth element

    uint32_t localDistance, distance;

    // find local potential kth element
    findClosest(localDistance, arr, p, comp);

    MPI_Allreduce(&localDistance, &distance, 1, MPI_UINT32_T, MPI_MIN, proc); // find the overall closest element to the pivot
                                                                              // that fullfills the condition imposed by the countSum-k relation

    kth = (countSumLess >= k) ? p - distance : p + distance; // calculate the kth element from its distance from the pivot

    return;
}