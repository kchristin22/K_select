#include <iostream>
#include <algorithm>
#include <mpi.h>
#include "kSelect.hpp"

int main()
{
    std::vector<uint32_t> arr(8);
    arr = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<std::vector<uint32_t>> arrs(4, std::vector<uint32_t>(2));
    for (uint32_t i = 0; i < 8; i += 2)
    {
        std::copy(arr.begin() + i, arr.begin() + i + 2, arrs[i / 2].begin());
    }
    int NumTasks, SelfTID;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);
    printf("Hello World from % i of % i\n", SelfTID, NumTasks);

    switch (SelfTID)
    {
    case 0:
        localFn(arrs[0], 5, 8);
        break;
    case 1:
        localFn(arrs[1], 5, 8);
        break;
    case 2:
        localFn(arrs[2], 5, 8);
        break;
    case 3:
        localFn(arrs[3], 5, 8);
        break;
    }

    MPI_Finalize();

    printf("Hello World from % i\n", SelfTID);
    return 0;
}