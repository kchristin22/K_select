#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdint.h>

#define CACHE_SIZE 800

struct localDataHeurQuick
{
    uint32_t localMin;
    uint32_t localMax;
    uint32_t count = 0;
    uint32_t leftMargin = 0;
    uint32_t rightMargin = 0;
};

inline void setComp(bool (*&comp)(const uint32_t &, const uint32_t &), const size_t k, const size_t n, const size_t countSum);

void findLocalMinMax(localDataHeurQuick &local, const std::vector<uint32_t> &arr);

void heurlocalSorting(localDataHeurQuick &local, std::vector<uint32_t> &arr, const uint32_t start, const uint32_t end, const uint32_t p);

void findClosest(uint32_t &distance, const std::vector<uint32_t> &arr, const uint32_t start, const uint32_t end, const uint32_t &p, bool (*comp)(const uint32_t &, const uint32_t &));

void heurQuickSelect(int &kth, std::vector<uint32_t> &arr, const size_t k, const size_t n, const size_t np);
