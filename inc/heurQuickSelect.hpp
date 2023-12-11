#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdint.h>

struct localDataQuick
{
    uint32_t localMin;
    uint32_t localMax;
    uint32_t count = 0;
};

inline void setComp(bool (*&comp)(const uint32_t &, const uint32_t &), const size_t k, const size_t n, const size_t countSum);

void findLocalMinMax(localDataQuick &local, const std::vector<uint32_t> &arr);

void heurlocalSorting(localDataQuick &local, std::vector<uint32_t> &arr, const uint32_t start, const uint32_t end, const uint32_t p);

void findClosest(uint32_t &closest, const std::vector<uint32_t> &arr, const uint32_t &p, bool (*comp)(const uint32_t &, const uint32_t &));

void heurQuickSelect(uint32_t kth, std::vector<uint32_t> &arr, const size_t k, const size_t n, const size_t np);
