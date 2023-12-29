#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdint.h>

#define CACHE_SIZE 800

/* Struct to store local process' data */
struct localDataHeurQuick
{
    uint32_t localMin;
    uint32_t localMax;
    size_t count = 0;
    size_t leftMargin = 0;
    size_t rightMargin = 0;
};

/* Find the local process' min and max
 * @param:
 *      local (output): local process' data whose min and max have been filled
 *      arr (input): the process' array to be searched
 */
void findLocalMinMax(localDataHeurQuick &local, const std::vector<uint32_t> &arr);

/* Partition the local process' array based on the pivot and find the count
 * @param:
 *      local (output): local process' data whose count variable has been filled
 *      arr (input): the process' array to be searched
 *      start (input): the first index of the array to start searching from
 *      end (input): the last index of the array to search to
 *      p (input): the pivot
 */
void heurlocalSorting(localDataHeurQuick &local, std::vector<uint32_t> &arr, const size_t start, const size_t end, const uint32_t p);

/* The parallel version of heurlocalSorting */
void heurParSorting(localDataHeurQuick &local, std::vector<uint32_t> &arr, const size_t start, const size_t end, const uint32_t p);

/* Set the compare function to use on the elements against the pivot, based on the count of the elements in comparison to k
 * @param:
 *      comp (output): the compare function to use
 *      k (input): the index of the sorted array whose value I'm looking for
 *      countSum (input): the number of elements less than or equal to the current pivot
 */
inline void setComp(bool (*&comp)(const uint32_t &, const uint32_t &), const size_t k, const size_t countSum);

/* Find the closest element to the pivot
 * @param:
 *      distance (output): the distance of the closest element to the pivot
 *      arr (input): the process' array to be searched
 *      start (input): the first index of the array to start searching from
 *      end (input): the last index of the array to search to
 *      p (input): the pivot
 *      comp (input): the comparison function to compare the elements to the pivot
 *      k (input): the index of the sorted array whose value I'm looking for
 */
void findClosest(uint32_t &distance, const std::vector<uint32_t> &arr, const size_t start, const size_t end, const uint32_t &p, bool (*comp)(const uint32_t &, const uint32_t &));

/* Find the kth element of the array
 * @param:
 *      kth (output): the kth element of the array
 *      arr (input): the process' array to be searched
 *      k (input): the index of the sorted array whose value I'm looking for
 *      n (input): the size of the whole array
 *      np (input): the number of processes in the MPI communicator
 */
void heurQuickSelect(uint32_t &kth, std::vector<uint32_t> &arr, const size_t k, const size_t n, const size_t np);
