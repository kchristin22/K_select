#include <iostream>
#include <algorithm>
#include <mpi.h>
#include "kSearch.hpp"
#include "heurQuickSelect.hpp"
#include "quickSelect.hpp"

#define k_default 4
#define ARRAY_SIZE 10

int main(int argc, char **argv)
{
    size_t k;
    if (argc == 2)
        k = atoi(argv[1]);
    else
        k = k_default;

    std::vector<uint32_t> arr(ARRAY_SIZE);
    if (k > arr.size())
    {
        printf("k is out pt array range\n");
        return 0;
    }
    srand(time(NULL));
    printf("arr: ");
    arr = {8, 8, 8, 9, 9, 8, 2, 4};
    // arr = {4, 4, 4, 2, 1, 2, 3, 2};
    // arr = {1, 2, 3, 4, 5, 6, 7, 8};
    // arr = {3, 1, 4, 2, 6, 10, 10, 9};
    // arr = {10, 10, 3, 10, 10, 7, 5, 6};
    // arr = {4, 1, 1, 4, 2, 5, 8, 4};
    // arr = {4, 3, 5, 5, 5, 4, 1, 1};
    // arr = {3, 10, 1, 10, 3, 7, 3, 9};
    // arr = {9, 1, 2, 10, 7, 10, 7, 10};
    // arr = {8, 8, 8, 8, 1, 1, 1, 1};

    for (uint32_t i = 0; i < arr.size(); i++)
    {
        // arr[i] = rand() % 10 + 1;
        printf("%d, ", arr[i]);
    }
    printf("\n");

    int NumTasks, SelfTID;
    uint32_t kth = 0;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    if (arr.size() < CACHE_SIZE) // check if the array fits in a single machine
    {

        if (SelfTID != 0)
            return 0;

        kSearch(kth, arr, k, arr.size(), NumTasks);
        printf("kth element kSearch: %d\n", kth);

        heurQuickSelect(kth, arr, k, arr.size(), NumTasks);
        printf("kth element heur: %d\n", kth);

        quickSelect(kth, arr, k, arr.size(), NumTasks);
        printf("kth element quick: %d\n", kth);

        MPI_Finalize();
        return 0;
    }

    int sendCount = arr.size() / NumTasks;
    int lastSendCount = sendCount + arr.size() % (sendCount * NumTasks);

    std::vector<int> sendCounts(NumTasks, sendCount);
    sendCounts[NumTasks - 1] = lastSendCount;
    std::vector<int> disp(NumTasks);

    for (int i = 1; i < NumTasks; i++)
        disp[i] = disp[i - 1] + sendCounts[i - 1];

    std::vector<std::vector<uint32_t>> arrs(NumTasks);
    for (int i = 0; i < NumTasks; i++)
    {
        arrs[i].resize(sendCounts[i]);
    }

    MPI_Scatterv(arr.data(), sendCounts.data(), disp.data(), MPI_UINT32_T, arrs[SelfTID].data(), sendCounts[SelfTID], MPI_UINT32_T, 0, MPI_COMM_WORLD);

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
            printf("%ldth element: %d\n", k, kth);
            break;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); // no need for a single barrier request, using Barrier_init, here, the use of the barrier is out of scope of the program (not included in the timings)

    MPI_Scatterv(arr.data(), sendCounts.data(), disp.data(), MPI_UINT32_T, arrs[SelfTID].data(), lastSendCount, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    heurQuickSelect(kth, arrs[SelfTID], k, arr.size(), NumTasks);

    if (SelfTID == 0)
        printf("kth element heur: %d\n", kth);

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<std::vector<uint32_t>> arrs2(NumTasks);

    for (int i = 0; i < NumTasks; i++)
    {
        arrs2[i].resize(sendCounts[i]);
    }

    MPI_Scatterv(arr.data(), sendCounts.data(), disp.data(), MPI_UINT32_T, arrs2[SelfTID].data(), lastSendCount, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    quickSelect(kth, arrs2[SelfTID], k, arr.size(), NumTasks);

    if (SelfTID == 0)
        printf("kth element quick: %d\n", kth);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}