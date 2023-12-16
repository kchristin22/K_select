#include <iostream>
#include <algorithm>
#include <mpi.h>
#include "kSelect.hpp"
#include "heurQuickSelect.hpp"
#include "quickSelect.hpp"

#define k 8

int main()
{
    std::vector<uint32_t> arr(8);
    srand(time(NULL));
    printf("arr: ");
    // arr = {8, 8, 8, 9, 9, 8, 2, 4};
    // arr = {4, 4, 4, 2, 1, 2, 3, 2};
    arr = {1, 2, 3, 4, 5, 6, 7, 8};

    for (uint32_t i = 0; i < 8; i++)
    {
        // arr[i] = rand() % 10 + 1;
        printf("%d, ", arr[i]);
    }
    printf("\n");

    std::vector<std::vector<uint32_t>> arrs(4, std::vector<uint32_t>(2));
    for (uint32_t i = 0; i < 8; i += 2) // change this
    {
        std::copy(arr.begin() + i, arr.begin() + i + 2, arrs[i / 2].begin()); // change this
    }
    int NumTasks, SelfTID;
    int kth = 0;

    MPI_Init(NULL, NULL);
    // MPI_Barrier_init(MPI_COMM_WORLD, NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    switch (SelfTID)
    {
    case 0:
        kSearch(kth, arrs[0], k, 8, NumTasks);
        printf("kth element: %d\n", kth);
        break;
    case 1:
        kSearch(kth, arrs[1], k, 8, NumTasks);
        break;
    case 2:
        kSearch(kth, arrs[2], k, 8, NumTasks);
        break;
    case 3:
        kSearch(kth, arrs[3], k, 8, NumTasks);
        break;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // switch (SelfTID)
    // {
    // case 0:
    //     heurQuickSelect(kth, arrs[0], k, 8, NumTasks);
    //     printf("kth element heur: %d\n", kth);
    //     break;
    // case 1:
    //     heurQuickSelect(kth, arrs[1], k, 8, NumTasks);
    //     break;
    // case 2:
    //     heurQuickSelect(kth, arrs[2], k, 8, NumTasks);
    //     break;
    // case 3:
    //     heurQuickSelect(kth, arrs[3], k, 8, NumTasks);
    //     break;
    // }

    // MPI_Barrier(MPI_COMM_WORLD);

    // switch (SelfTID)
    // {
    // case 0:
    //     quickSelect(kth, arrs[0], k, 8, NumTasks);
    //     printf("kth element quick: %d\n", kth);
    //     break;
    // case 1:
    //     quickSelect(kth, arrs[1], k, 8, NumTasks);
    //     break;
    // case 2:
    //     quickSelect(kth, arrs[2], k, 8, NumTasks);
    //     break;
    // case 3:
    //     quickSelect(kth, arrs[3], k, 8, NumTasks);
    //     break;
    // }

    // MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    // printf("Hello World from % i\n", SelfTID);
    return 0;
}