#include <iostream>
#include <algorithm>
#include <mpi.h>
#include "kSelect.hpp"
#include "heurQuickSelect.hpp"
#include "quickSelect.hpp"

#define k_default 4

int main(int argc, char **argv)
{
    int k;
    if (argc == 2)
        k = atoi(argv[1]);
    else
        k = k_default;

    std::vector<uint32_t> arr(8);
    if (k > (int)arr.size())
    {
        printf("k is out pt array range\n");
        return 0;
    }
    srand(time(NULL));
    printf("arr: ");
    // arr = {8, 8, 8, 9, 9, 8, 2, 4};
    // arr = {4, 4, 4, 2, 1, 2, 3, 2};
    // arr = {1, 2, 3, 4, 5, 6, 7, 8};
    // arr = {3, 1, 4, 2, 6, 10, 10, 9};
    // arr = {10, 10, 3, 10, 10, 7, 5, 6};
    // arr = {4, 1, 1, 4, 2, 5, 8, 4};
    // arr = {4, 3, 5, 5, 5, 4, 1, 1};
    // arr = {3, 10, 1, 10, 3, 7, 3, 9};
    // arr = {9, 1, 2, 10, 7, 10, 7, 10};
    // arr = {8, 8, 8, 8, 1, 1, 1, 1};

    for (uint32_t i = 0; i < 8; i++)
    {
        arr[i] = rand() % 10 + 1;
        printf("%d, ", arr[i]);
    }
    printf("\n");

    std::vector<std::vector<uint32_t>> arrs(4, std::vector<uint32_t>(2));

    int NumTasks, SelfTID;
    int kth = 0;

    MPI_Init(NULL, NULL);
    // MPI_Barrier_init(MPI_COMM_WORLD, NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    MPI_Scatter(arr.data(), 2, MPI_UINT32_T, arrs[SelfTID].data(), 2, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    kSearch(kth, arrs[SelfTID], k, arr.size(), NumTasks);

    if (SelfTID == 0)
    {
        switch (k)
        {
        case 1:
            printf("1st element: %d\n", kth);
            break;
        case 2:
            printf("2nd element: %d\n", kth);
            break;
        case 3:
            printf("3rd element: %d\n", kth);
            break;
        default:
            printf("%dth element: %d\n", k, kth);
            break;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Scatter(arr.data(), 2, MPI_UINT32_T, arrs[SelfTID].data(), 2, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    heurQuickSelect(kth, arrs[SelfTID], k, arr.size(), NumTasks);

    if (SelfTID == 0)
        printf("kth element heur: %d\n", kth);

    MPI_Barrier(MPI_COMM_WORLD);

    quickSelect(kth, arrs[SelfTID], k, arr.size(), NumTasks);

    if (SelfTID == 0)
        printf("kth element quick: %d\n", kth);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}