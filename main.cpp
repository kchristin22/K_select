#include <iostream>
#include <algorithm>
#include <mpi.h>
#include "kSelect.hpp"

int main()
{
    std::vector<uint32_t> arr = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<std::vector<uint32_t>> arrs(4, std::vector<uint32_t>(2));
    for (int i = 0; i < 4; i++)
    {
        std::copy(arr.begin(), arr.begin() + 2, arrs[i].begin());
    }
    MPI_Init(NULL, NULL);

    localFn(arrs[0], 5, 8);

    MPI_Finalize();

    printf("Hello World!\n");
    return 0;
}