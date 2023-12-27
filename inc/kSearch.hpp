#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdint.h>

#define CACHE_SIZE 800

/* Struct to store local process' data */
struct localData
{
    int localMin;
    int localMax;
    size_t count = 0;
};

/* Find the local process' min and max
 * @param:
 *      local (output): local process' data whose min and max have been filled
 *      arr (input): the process' array to be searched
 */
void findLocalMinMax(localData &local, const std::vector<int> &arr);

/* Find the local process' count
 * @param:
 *      local (output): local process' data whose count variable has been filled
 *      arr (input): the process' array to be searched
 *      p (input): the pivot
 *      comp (input): the comparison function to compare the elements to the pivot
 *      k (input): the index of the sorted array whose value I'm looking for
 *      n (input): the size of the whole array
 */
inline void findLocalCount(localData &local, const std::vector<int> &arr, const int &p, const bool (*comp)(const int &, const int &), const size_t k, const size_t n);

/* Find the closest element to the pivot
 * @param:
 *      distance (output): the distance of the closest element to the pivot
 *      arr (input): the process' array to be searched
 *      p (input): the pivot
 *      comp (input): the comparison function to compare the elements to the pivot
 */
void findClosest(uint32_t &distance, const std::vector<int> &arr, const int &p, const bool (*comp)(const int &, const int &));

/* Find the kth element of the array
 * @param:
 *      kth (output): the kth element of the array
 *      arr (input): the process' array to be searched
 *      k (input): the index of the sorted array whose value I'm looking for
 *      n (input): the size of the whole array
 *      np (input): the number of processes in the MPI communicator
 */
void kSearch(int &kth, std::vector<int> &arr, const size_t k, const size_t n, const size_t np);