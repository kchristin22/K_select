#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdint.h>

struct localData
{
    uint32_t localMin;
    uint32_t localMax;
    uint32_t count = 0;
};

void localFn(uint32_t kth, std::vector<uint32_t> &arr, const size_t k, const size_t n, const size_t np);
