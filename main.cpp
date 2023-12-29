#include <iostream>
#include <fstream>
#include <algorithm>
#include <execution>
#include <mpi.h>
#include "kSearch.hpp"
#include "heurQuickSelect.hpp"
#include "quickSelect.hpp"

int main(int argc, char **argv)
{
    size_t k = 1;
    std::ifstream file;

    switch (argc)
    {
    case 1:
        std::cout << "No arguments provided\n The usage is mpirun -np x ./output input_vector.txt k" << std::endl;
        return 0;
    case 3:
        k = atoll(argv[2]);
    case 2:
        file.open(argv[1]);
        if (!file.is_open())
        {
            std::cout << "File not found" << std::endl;
            return 0;
        }
        if (argc == 3)
            break;
        std::cout << "K is not specified provided, using default k = ARRAY_SIZE / 2" << std::endl;
        break;
    default:
        std::cout << "Too many arguments provided" << std::endl;
        return 0;
    }

    std::vector<uint32_t> arr;
    uint32_t value;
    while (file >> value)
    {
        arr.push_back(value);
    }

    file.close();
    size_t n = arr.size();

    if (argc == 2)
        k = n / 2;
    else if (k > n)
    {
        printf("k is out of array range\n");
        return 0;
    }

    int NumTasks, SelfTID;
    uint32_t kth = 0;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    if (SelfTID == 0)
    {
        printf("k = %ld\n", k);
        std::vector<uint32_t> arrSort(n);
        arrSort = arr;
        std::sort(std::execution::par_unseq, arrSort.begin(), arrSort.end());
        printf("kth correct: %u\n", arrSort[k - 1]);
    }

    if (arr.size() < CACHE_SIZE) // check if the array fits in a single machine
    {
        NumTasks = 1;

        if (SelfTID == 0)
        {
            std::vector<uint32_t> arr2(n);
            arr2 = arr;
            std::vector<uint32_t> arr3(n);
            arr3 = arr;

            kSearch(kth, arr, k, n, NumTasks);
            printf("kth element kSearch: %u\n", kth);

            heurQuickSelect(kth, arr2, k, n, NumTasks);
            printf("kth element heur: %u\n", kth);

            quickSelect(kth, arr3, k, n, NumTasks);
            printf("kth element quick: %u\n", kth);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Finalize();

        return 0;
    }

    int sendCount = n / NumTasks;
    int lastSendCount = sendCount + n % (sendCount * NumTasks);

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

    kSearch(kth, arrs[SelfTID], k, n, NumTasks);

    if (SelfTID == 0)
        printf("kth element kSearch: %u\n", kth);

    MPI_Barrier(MPI_COMM_WORLD); // no need for a single barrier request, using Barrier_init, here, the use of the barrier is out of scope of the program (not included in the timings)

    std::vector<std::vector<uint32_t>> arrs2(NumTasks);

    for (int i = 0; i < NumTasks; i++)
    {
        arrs2[i].resize(sendCounts[i]);
    }

    MPI_Scatterv(arr.data(), sendCounts.data(), disp.data(), MPI_UINT32_T, arrs2[SelfTID].data(), lastSendCount, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    heurQuickSelect(kth, arrs2[SelfTID], k, n, NumTasks);

    if (SelfTID == 0)
        printf("kth element heur quick: %u\n", kth);

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<std::vector<uint32_t>> arrs3(NumTasks);

    for (int i = 0; i < NumTasks; i++)
    {
        arrs3[i].resize(sendCounts[i]);
    }

    MPI_Scatterv(arr.data(), sendCounts.data(), disp.data(), MPI_UINT32_T, arrs3[SelfTID].data(), lastSendCount, MPI_UINT32_T, 0, MPI_COMM_WORLD);

    quickSelect(kth, arrs3[SelfTID], k, n, NumTasks);

    if (SelfTID == 0)
        printf("kth element quick: %u\n", kth);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}