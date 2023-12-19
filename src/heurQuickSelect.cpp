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
    // if (countSum == (k + 1))
    //     comp = lessThan; // the next element less than the pivot is the kth element
    if (countSum == k)
        comp = lessEqualThan;     // the next element less than or equal to the pivot is the kth element
    else if (countSum == (k - 1)) // ((countSum == (k - 1))
        comp = greaterThan;       // the next element bigger than the pivot is the kth element

    return;
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

void heurlocalSorting(localDataHeurQuick &local, std::vector<uint32_t> &arr, const uint32_t start, const uint32_t end, const uint32_t p)
{
    uint32_t i = start, j = end;
    if (i > j)
    {
        local.count = i;
        return;
    }
    else if (j == arr.size())
        j--;

    while (true)
    {
        while ((arr[i] <= p) && i <= j)
            i++; // the count
        while ((arr[j] > p) && i < j)
            j--;
        if (i < j)
            std::swap(arr[i], arr[j]);
        else
            break;
    }

    local.count = i; // save the count and not the index
    return;
}

void heurfindClosest(int &closest, const std::vector<uint32_t> &arr, const uint32_t &p, bool (*comp)(const uint32_t &, const uint32_t &))
{ // check for custom reduction with template functions
    closest = arr[0];
#pragma omp parallel for
    for (size_t i = 1; i < arr.size(); i++) // change limits
    {
        if (comp(arr[i], p))
        {
#pragma omp atomic write
            closest = comp(closest, p) && comp(arr[i], closest) ? closest : arr[i];
        }
    }

    return;
}

void heurQuickSelect(int &kth, std::vector<uint32_t> &arr, const size_t k, const size_t n, const size_t np)
{
    localDataHeurQuick local;
    findLocalMinMax(local, arr);

    uint32_t min = local.localMin, max = local.localMax;

    int SelfTID;
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    MPI_Reduce(&local.localMin, &min, 1, MPI_UINT32_T, MPI_MIN, 0, MPI_COMM_WORLD);

    MPI_Bcast(&min, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    MPI_Reduce(&local.localMax, &max, 1, MPI_UINT32_T, MPI_MAX, 0, MPI_COMM_WORLD);

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
    uint32_t countSum = 0, prevCountSum = n;
    uint32_t start = 0, end = arr.size() - 1;
    local.rightMargin = arr.size() - 1;

    while (true)
    {
        heurlocalSorting(local, arr, start, end, p); // find local count

        MPI_Reduce(&local.count, &countSum, 1, MPI_UINT32_T, MPI_SUM, 0, MPI_COMM_WORLD);

        MPI_Bcast(&countSum, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

        if (abs(prevP - p) == 1)
        {
            if (prevCountSum < countSum && k > prevCountSum && k < countSum)
            {
                kth = p; // pivot with the greatest count of the two (and thus the largest pivot value) is in the array and is the kth element
                return;
            }
            else if (prevCountSum > countSum && k < prevCountSum && k > countSum)
            {
                kth = prevP;
                return;
            }
        }

        if (k == countSum || countSum == (k - 1)) // if countSum is equal to k, then we have found the kth element
            break;
        else if (countSum > k)
        {
            start = local.leftMargin;
            end = local.count;
            local.rightMargin = local.count;
        }
        else
        {
            start = local.count;
            end = local.rightMargin;
            local.leftMargin = local.count;
        }

        if (countSum == prevCountSum)
            prevCountSum = (k > countSum && prevP > p) ? prevCountSum + 1 : prevCountSum - 1;

        newP = p + ((k - countSum) * (prevP - p)) / (prevCountSum - countSum); // find pivot
        if (newP < (int)min)
            newP = min;
        else if (newP > (int)max)
            newP = max;
        else if (newP == p)
            newP = (k > countSum) ? p + 1 : p - 1;

        prevP = p;
        p = newP;
        prevCountSum = countSum;
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
