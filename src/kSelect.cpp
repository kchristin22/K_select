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

// inline bool checkCond(const size_t k, const size_t n, const uint32_t countSum)
// {
//     if (k < n / 2)
//         return ((countSum == k) || (countSum == (k - 1))) ? true : false;
//     else
//         return ((countSum == (n - k)) || (countSum == (n - k + 1))) ? true : false;
// }

// inline void setComp(bool (*&comp)(const uint32_t &, const uint32_t &), const size_t k, const size_t n, const size_t countSum)
// {
//     if (k < n / 2)
//     {
//         if ((countSum >= k))
//             comp = lessEqualThan; // the next element less than or equal to the pivot is the kth element
//         else if (countSum <= (k - 1))
//             comp = greaterThan; // the next element bigger than the pivot is the kth element
//     }
//     else
//     {
//         if (countSum == (n - k))
//             comp = lessEqualThan; // the next element less than or equal to the pivot is the kth element
//         else if (countSum == (n - k + 1))
//             comp = greaterThan; // the next element bigger than the pivot is the kth element
//     }

//     return;
// }

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

void localFnLessEqual(int &kth, int &p, uint32_t &countSum, int prevP, uint32_t prevCountSum, localData local, std::vector<uint32_t> &arr, const uint32_t min, const uint32_t max, const size_t k, const size_t n, const size_t np)
{
    int SelfTID;
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    kth = 0;
    int newP;
    while (true) // break this in two while loops, based on k position
    {

        if (countSum == prevCountSum)
            countSum = k > countSum ? countSum + 1 : countSum - 1;

        newP = p + ((k - countSum) * (prevP - p)) / (prevCountSum - countSum); // find pivot
        if (newP < (int)min)
            newP = min;
        else if (newP > (int)max)
            newP = max;
        else if (newP == p)
            newP = (k > countSum) ? p++ : p--;

        prevP = p;
        p = newP;
        prevCountSum = countSum;

        findLocalCount(local, arr, p, lessEqualThan, k, n); // find local count

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

        if (abs(prevP - p) == 1)
        {
            if (lessEqualThan(prevCountSum, countSum) && k > prevCountSum && k < countSum) // pivots with distance =1 and k belongs in the range of their count sums
            {
                kth = p; // pivot with largest count of the two (and thus largest pivot value) is in the array and is the kth element
                return;
            }
        }

        if ((countSum == k) || (countSum == (k - 1))) // if countSum is equal to k, then we have found the kth element
            return;
        // else
        //     std::erase_if(arr, [p, k, countSum, comp](uint32_t x)
        //                   { return (k - countSum) > 0 ? x < p : x > p; });
    }
}

void localFnGreater(int &kth, int &p, uint32_t &countSum, int prevP, uint32_t prevCountSum, localData local, std::vector<uint32_t> &arr, const uint32_t min, const uint32_t max, const size_t k, const size_t n, const size_t np)
{
    int SelfTID;
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    kth = 0;
    int newP;
    while (true) // break this in two while loops, based on k position
    {
        if (countSum == prevCountSum)
            countSum = k > (n - countSum) ? countSum + 1 : countSum - 1;

        newP = p + ((k - (n - countSum)) * (prevP - p)) / ((n - prevCountSum) - (n - countSum)); // find pivot
        if (newP < (int)min)
            newP = min;
        else if (newP > (int)max)
            newP = max;
        else if (newP == p)
            newP = (k > (n - countSum)) ? p + 1 : p - 1;

        prevP = p;
        p = newP;
        prevCountSum = countSum;

        findLocalCount(local, arr, p, greaterThan, k, n); // find local count

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

        if (abs(prevP - p) == 1)
        {
            if (greaterThan(prevCountSum, countSum) && k > (n - prevCountSum) && k < (n - countSum))
            {
                kth = p; // pivot with the smallest count of the two (and thus the largest pivot value) is in the array and is the kth element
                return;
            }
        }

        if ((countSum == (n - k)) || (countSum == (n - k + 1))) // if countSum is equal to k, then we have found the kth element
            return;
        // else
        //     std::erase_if(arr, [p, k, countSum, comp, n](uint32_t x)
        //                   { return (k - (n - countSum)) > 0 ? x < p : x > p; });
    }
}

void kSearch(int &kth, std::vector<uint32_t> &arr, const size_t k, const size_t n, const size_t np)
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

    if (min == max)
    {
        kth = min;
        return;
    }

    int p = min, prevP = max;
    uint32_t countSum, prevCountSum;
    bool (*comp)(const uint32_t &, const uint32_t &);

    if (k < n / 2)
    {
        countSum = 0;
        prevCountSum = n;
        comp = lessEqualThan;                                                                    // optimize counting of elements based on the percentile of k
        localFnLessEqual(kth, p, countSum, prevP, prevCountSum, local, arr, min, max, k, n, np); // consider making p and countsum a struct

        if ((countSum >= k))
            comp = lessEqualThan; // the next element less than or equal to the pivot is the kth element
        else if (countSum <= (k - 1))
            comp = greaterThan; // the next element bigger than the pivot is the kth element
    }
    else
    {
        countSum = n;
        prevCountSum = 0;
        comp = greaterEqualThan;
        localFnGreater(kth, p, countSum, prevP, prevCountSum, local, arr, min, max, k, n, np);

        if (countSum == (n - k))
            comp = lessEqualThan; // the next element less than or equal to the pivot is the kth element
        else if (countSum == (n - k + 1))
            comp = greaterThan; // the next element bigger than the pivot is the kth element
    }

    if (kth == p)
        return;

    // setComp(comp, k, n, countSum);

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
