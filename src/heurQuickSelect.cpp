#include <limits.h>
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

void heurParSorting(localDataHeurQuick &local, std::vector<uint32_t> &arr, const uint32_t start, const uint32_t end, const uint32_t p)
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
#pragma omp parallel sections shared(i, j)
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

    while (i < j)
    {
        while ((arr[i] <= p) && i <= j)
            i++; // the count
        while ((arr[j] > p) && i < j)
            j--;
        if (i < j)
            std::swap(arr[i], arr[j]);
    }

    local.count = i; // save the count and not the index
    return;
}

void findClosest(uint32_t &distance, const std::vector<uint32_t> &arr, const uint32_t start, const uint32_t end, const uint32_t &p, bool (*comp)(const uint32_t &, const uint32_t &))
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

void heurQuickSelect(int &kth, std::vector<uint32_t> &arr, const size_t k, const size_t n, const size_t np)
{
    int SelfTID;
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    MPI_Comm proc = MPI_COMM_WORLD;

    std::vector<uint32_t> array;
    bool gathered = false;

    if (array.size() * np < CACHE_SIZE / 2)
    {
        array.resize(arr.size() * np);
        MPI_Gather(arr.data(), arr.size(), MPI_UINT32_T, array.data(), arr.size(), MPI_UINT32_T, 0, MPI_COMM_WORLD);

        if (SelfTID != 0)
            return;

        proc = MPI_COMM_SELF;
        gathered = true;
    }
    else
        array = std::move(arr);

    // printf("arr: ");
    // for (size_t i = 0; i < array.size(); i++)
    // {
    //     printf("%d, ", array[i]);
    // }
    // printf("\n");

    localDataHeurQuick local;
    findLocalMinMax(local, array);

    uint32_t min = local.localMin, max = local.localMax;

    MPI_Allreduce(&local.localMin, &min, 1, MPI_UINT32_T, MPI_MIN, proc);

    MPI_Allreduce(&local.localMax, &max, 1, MPI_UINT32_T, MPI_MAX, proc);

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
    uint32_t start = 0, end = array.size() - 1;
    local.rightMargin = array.size() - 1;

    while (true)
    {
        heurParSorting(local, array, start, end, p); // find local count

        MPI_Allreduce(&local.count, &countSum, 1, MPI_UINT32_T, MPI_SUM, proc);

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

        if (countSum >= k)
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

        if (k == countSum || countSum == (k - 1)) // if countSum is equal to k, then we have found the kth element
            break;

        // gather the array if i) it is not gathered already, ii) the pivot is larger than the kth and iii) the size is small enough
        if (gathered == false && countSum > k && countSum < CACHE_SIZE / 2) // work on cache condition
        {
            std::vector<uint32_t> tempArr(countSum); // store local array
            std::vector<int> recvCount(np);
            std::vector<int> disp(np);

            MPI_Gather(&local.count, 1, MPI_INT, recvCount.data(), 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

            for (size_t i = 1; i < np; i++)
                disp[i] = disp[i - 1] + recvCount[i - 1];

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

            printf("arr left: ");
            for (size_t i = 0; i < array.size(); i++)
            {
                printf("%d, ", array[i]);
            }
            printf("\n");
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

    uint32_t localDistance, distance;

    start %= array.size();
    end %= array.size(); // end refers to an index inside the array (see localSorting)

    findClosest(localDistance, array, start, end, p, comp);

    MPI_Allreduce(&localDistance, &distance, 1, MPI_UINT32_T, MPI_MIN, proc); // find the overall closest element to the pivot
                                                                              // that fullfills the condition imposed by the countSum-k relation

    kth = (countSum == k) ? p - distance : p + distance; // calculate the kth element

    return;
}
