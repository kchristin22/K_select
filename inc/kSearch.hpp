#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdint.h>

#define CACHE_SIZE 800

/* Struct to store local process' data */
struct localData
{
    uint32_t localMin;
    uint32_t localMax;
    uint32_t count = 0;
};

/* Find the local process' min and max
 * @param:
 *      local (output): local process' data to fill the min and max
 *      arr (input): the process' array to be searched
 */
void findLocalMinMax(localData &local, const std::vector<uint32_t> &arr);

/* Find the local process' count
 * @param:
 *      local (output): local process' data to fill the count
 *      arr (input): the process' array to be searched
 *      p (input): the pivot
 *      comp (input): the comparison function to compare the elements with the pivot
 *      k (input): the index of the sorted array whose element I'm looking for
 *      n (input): the size of the whole array
 */
inline void findLocalCount(localData &local, const std::vector<uint32_t> &arr, const uint32_t &p, const bool (*comp)(const uint32_t &, const uint32_t &), const size_t k, const size_t n);

/* Find the closest element to the pivot
 * @param:
 *      distance (output): the distance of the closest element to the pivot
 *      arr (input): the process' array to be searched
 *      p (input): the pivot
 *      comp (input): the comparison function to compare the elements with the pivot
 *      k (input): the index of the sorted array whose element I'm looking for
 */
void findClosest(uint32_t &distance, const std::vector<uint32_t> &arr, const int &p, const bool (*comp)(const uint32_t &, const uint32_t &), const size_t k);

/* Find the closest element to the pivot
 * @param:
 *      kth (output): the kth element of the array
 *      arr (input): the process' array to be searched
 *      k (input): the index of the sorted array whose element I'm looking for
 *      n (input): the size of the whole array
 *      np (input): the number of processes in the MPI communicator
 */
void kSearch(int &kth, std::vector<uint32_t> &arr, const size_t k, const size_t n, const size_t np);