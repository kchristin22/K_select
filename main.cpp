#include <iostream>
#include <algorithm>
#include <execution>
#include <mpi.h>
#include "kSearch.hpp"
#include "heurQuickSelect.hpp"
#include "quickSelect.hpp"

#define k_default 4
#define ARRAY_SIZE 1600

int main(int argc, char **argv)
{
    size_t k;
    if (argc == 2)
        k = atoi(argv[1]);
    else
        k = k_default;

    std::vector<int> arr(ARRAY_SIZE);
    if (k > arr.size())
    {
        printf("k is out pt array range\n");
        return 0;
    }
    srand(time(NULL));

    for (size_t i = 0; i < arr.size(); i++)
    {
        arr[i] = rand() % 100 + 1;
    }

    int NumTasks, SelfTID;
    int kth = 0;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    if(SelfTID == 0){
        std::vector<int> arrSort(ARRAY_SIZE);
        arrSort = arr;
        std::sort(std::execution::par_unseq, arrSort.begin(), arrSort.end());
        printf("k correct: %d\n", arrSort[k - 1]);
    }

    if (arr.size() < CACHE_SIZE) // check if the array fits in a single machine
    {
        NumTasks = 1;

        if (SelfTID == 0)
        {
            std::vector<int> arr2(ARRAY_SIZE);
            arr2 = arr;
            std::vector<int> arr3(ARRAY_SIZE);
            arr3 = arr;

            kSearch(kth, arr, k, arr.size(), NumTasks);
            printf("kth element kSearch: %d\n", kth);

            heurQuickSelect(kth, arr2, k, arr2.size(), NumTasks);
            printf("kth element heur: %d\n", kth);

            quickSelect(kth, arr3, k, arr3.size(), NumTasks);
            printf("kth element quick: %d\n", kth);
        }

        MPI_Barrier(MPI_COMM_WORLD);

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

    std::vector<std::vector<int>> arrs(NumTasks);
    for (int i = 0; i < NumTasks; i++)
    {
        arrs[i].resize(sendCounts[i]);
    }

    MPI_Scatterv(arr.data(), sendCounts.data(), disp.data(), MPI_INT, arrs[SelfTID].data(), sendCounts[SelfTID], MPI_INT, 0, MPI_COMM_WORLD);

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

    std::vector<std::vector<int>> arrs2(NumTasks);

    for (int i = 0; i < NumTasks; i++)
    {
        arrs2[i].resize(sendCounts[i]);
    }

    MPI_Scatterv(arr.data(), sendCounts.data(), disp.data(), MPI_INT, arrs2[SelfTID].data(), lastSendCount, MPI_INT, 0, MPI_COMM_WORLD);

    heurQuickSelect(kth, arrs2[SelfTID], k, arr.size(), NumTasks);

    if (SelfTID == 0)
        printf("kth element heur: %d\n", kth);

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<std::vector<int>> arrs3(NumTasks);

    for (int i = 0; i < NumTasks; i++)
    {
        arrs3[i].resize(sendCounts[i]);
    }

    MPI_Scatterv(arr.data(), sendCounts.data(), disp.data(), MPI_INT, arrs3[SelfTID].data(), lastSendCount, MPI_INT, 0, MPI_COMM_WORLD);

    quickSelect(kth, arrs3[SelfTID], k, arr.size(), NumTasks);

    if (SelfTID == 0)
        printf("kth element quick: %d\n", kth);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}