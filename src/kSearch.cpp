#include <limits.h>
#include <omp.h>
#include <mpi.h>
#include "kSearch.hpp"

inline bool lessEqualThan(const uint32_t &a, const uint32_t &b)
{
    return a <= b;
}

inline bool greaterThan(const uint32_t &a, const uint32_t &b)
{
    return a > b;
}

void findLocalMinMax(localData &local, const std::vector<uint32_t> &arr)
{
    // find local min
    uint32_t localmin = arr[0];

#pragma omp parallel
#pragma omp for nowait reduction(min : localmin)
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

#pragma omp parallel
#pragma omp for nowait reduction(max : localmax)
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
    size_t count = 0;

    if (p < local.localMin)
        local.count = 0; // no elements are less than or equal to the pivot
    else if (p >= local.localMax)
        local.count = arr.size(); // all elements are less than or equal to the pivot
    else
    {
#pragma omp parallel
#pragma omp for nowait reduction(+ : count)
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

void findClosest(uint32_t &distance, const std::vector<uint32_t> &arr, const uint32_t &p, bool (*comp)(const uint32_t &, const uint32_t &))
{
    distance = UINT_MAX; // if there is no element fulfilling the condition, return INT_MAX to increase its distance from the pivot
#pragma omp parallel
#pragma omp for nowait reduction(min : distance)
    for (size_t i = 0; i < arr.size(); i++)
    {
        if (comp(arr[i], p))
        {
            distance = abs(arr[i] - p) < distance ? abs(arr[i] - p) : distance; // store the smallest distance from the pivot that fullfills the condition
        }
    }

    return;
}

void kSearch(uint32_t &kth, std::vector<uint32_t> &arr, const size_t k, const size_t n, const size_t np)
{

    MPI_Comm proc = np > 1 ? MPI_COMM_WORLD : MPI_COMM_SELF;

    int SelfTID;
    MPI_Comm_rank(proc, &SelfTID);

    localData local;
    findLocalMinMax(local, arr);

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
    size_t countSumLess = n, prevCountSumLess = 0;
    bool (*comp)(const uint32_t &, const uint32_t &) = (k < n / 2) ? lessEqualThan : greaterThan; // optimize the number of elements to count based on k's position
    bool gathered = np == 1 ? true : false;                                                       // already gathered prior to the call

    while (true)
    {
        if (countSumLess == prevCountSumLess)
            prevCountSumLess = prevP > p ? prevCountSumLess + 1 : prevCountSumLess - 1; // change prevCountSum instead of countSum to not affect the next iteration of the algorithm
                                                                                        // this change drives the sign of the difference between the two pivots, by letting only the countSum-k relationship determine it:
                                                                                        // the pivot decreases if the countSum is bigger than k, and increases if the countSum is smaller than k

        // find new pivot through linear interpolation, and make sure it's not out of bounds
        int64_t fraction = ((static_cast<int64_t>(k) - static_cast<int64_t>(countSumLess)) * (static_cast<int64_t>(prevP) - static_cast<int64_t>(p))) / (static_cast<int64_t>(prevCountSumLess) - static_cast<int64_t>(countSumLess));
        // printf("fraction: %ld\n", fraction);
        newP = static_cast<int64_t>(p) + fraction < 0 ? min : static_cast<int64_t>(p) + fraction;
        if (newP > max)
            newP = max;
        if (newP == p)
            newP = (k > countSumLess || p == 0) ? p + 1 : p - 1; // ensure that the pivot doesnot underflow with p == 0 check
        else if (newP == prevP)
            newP = (k > prevCountSumLess || prevP == 0) ? prevP + 1 : prevP - 1; // avoid looping between two pivots

        prevP = p;
        p = newP;
        prevCountSumLess = countSumLess;

        findLocalCount(local, arr, p, comp, k, n); // find the local number of elements that are less than or equal to the pivot

        MPI_Allreduce(&local.count, &countSumLess, 1, MPI_UNSIGNED_LONG, MPI_SUM, proc); // find the overall number and broadcast it so all processes can do calculations with it

        // printf("p: %u, prevP: %u, countSum: %ld, prevCountSum: %ld\n", p, prevP, countSumLess, prevCountSumLess);

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
            else if (p == 0 && countSumLess > k) // min element is 0 and is included in the array multiple times, cannot go lower than zero to fullfill the above conditions
            {
                kth = p;
                return;
            }
        }

        if ((countSumLess == k) || (countSumLess == (k - 1))) // if countSum is equal to k or k-1, then we can now find the kth element
            break;
        else if (countSumLess > k)
        {
            std::erase_if(arr, [p](uint32_t x)
                          { return x > p; }); // if countSum is bigger than k, then the kth element is smaller than the pivot
            max = p;                          // mimic quickSelect's value range reduction
        }

        if (gathered == false && countSumLess > k && countSumLess < CACHE_SIZE / 2) // /2 to ensure that there's enough space to have two copies of the gathered array
        {
            if (local.count < arr.size())
                arr.erase(arr.begin() + local.count, arr.end()); // remove the elements that are larger than the pivot, so there's enough space to gather the elements
            std::vector<uint32_t> tempArr(countSumLess);         // store local array
            std::vector<int> recvCount(np);
            std::vector<int> disp(np, 0);

            // store the amount of data each process will send
            MPI_Gather(&local.count, 1, MPI_INT, recvCount.data(), 1, MPI_INT, 0, proc);

            // calculate the displacement of each process' data
            for (size_t i = 1; i < np; i++)
                disp[i] = disp[i - 1] + recvCount[i - 1];

            // use Gatherv to gather different amounts of data from each process
            MPI_Gatherv(arr.data(), local.count, MPI_UINT32_T, tempArr.data(), recvCount.data(), disp.data(), MPI_UINT32_T, 0, proc);

            if (SelfTID != 0)
                return;

            arr.resize(countSumLess);
            arr = std::move(tempArr); // we lose the local array here, we can copy it if we want to keep it but maybe both won't fit in memory

            proc = MPI_COMM_SELF;
            gathered = true;
        }
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