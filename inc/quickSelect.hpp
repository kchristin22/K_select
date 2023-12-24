#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdint.h>

#define CACHE_SIZE 800

/* Struct to store local process' data */
struct localDataQuick
{
    uint32_t localMin;
    uint32_t localMax;
    uint32_t count = 0;
    uint32_t leftMargin = 0;
    uint32_t rightMargin = 0;
};

/* Partition the local process' array based on the pivot
 * @param:
 *      local (output): local process' data whose count variable has been filled
 *      arr (input): the process' array to be searched
 *      start (input): the first index of the array to start searching from
 *      end (input): the last index of the array to search to
 *      p (input): the pivot
 */
void localSorting(localDataQuick &local, std::vector<uint32_t> &arr, const size_t start, const size_t end, const uint32_t p);

/* The parallel version of localSorting */
void parSorting(localDataQuick &local, std::vector<uint32_t> &arr, const size_t start, const size_t end, const uint32_t p);

/* Find the kth element of the array
 * @param:
 *      kth (output): the kth element of the array
 *      arr (input): the process' array to be searched
 *      k (input): the index of the sorted array whose value I'm looking for
 *      n (input): the size of the whole array
 *      np (input): the number of processes in the MPI communicator
 */
void quickSelect(uint32_t &kth, std::vector<uint32_t> &arr, const size_t k, const size_t n, const size_t np);