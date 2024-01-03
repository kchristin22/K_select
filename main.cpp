#include <iostream>
#include <fstream>
#include <algorithm>
#include <execution>
#include <mpi.h>
#include "kSearch.hpp"
#include "heurQuickSelect.hpp"
#include "quickSelect.hpp"
#include "getWiki.hpp"

#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

#define URL_DEFAULT (char *)"https://dumps.wikimedia.org/other/static_html_dumps/current/el/wikipedia-el-html.tar.7z"

int main(int argc, char **argv)
{
    size_t k = 1;
    char *url = NULL;

    switch (argc)
    {
    case 1:
        std::cout << "No arguments provided\n The usage is mpirun -np x ./output ${k} ${url}" << std::endl;
        std::cout << "Using default url: " << URL_DEFAULT << std::endl;
        url = URL_DEFAULT;
        std::cout << "Using default k = ARRAY_SIZE / 2" << std::endl;
        break;
    case 2:
        k = atoll(argv[1]);
        std::cout << "Using default url: " << URL_DEFAULT << std::endl;
        url = URL_DEFAULT;
        break;
    case 3:
        url = argv[2];
        k = atoll(argv[1]);
        break;
    default:
        std::cout << "Too many arguments provided" << std::endl;
        return 0;
    }

    int NumTasks, SelfTID;
    uint32_t kth = 0;
    size_t n = 0;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &NumTasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &SelfTID);

    CURL *curl_handle = NULL;
    n = getWikiInfo(url, curl_handle) / sizeof(uint32_t);

    if (argc == 1)
        k = n / 2;
    else if (k > n)
    {
        printf("k is out of array range\n");
        return 0;
    }

    // if (SelfTID == 0)
    // {
    //     printf("k = %ld\n", k);
    //     std::vector<uint32_t> arrSort(n);
    //     arrSort = arr;
    //     std::sort(std::execution::par_unseq, arrSort.begin(), arrSort.end());
    //     printf("kth correct: %u\n", arrSort[k - 1]);
    // }

    if (n < CACHE_SIZE) // check if the array fits in a single machine
    {
        NumTasks = 1;

        if (SelfTID == 0)
        {
            ARRAY result = getWikiPartition(url, SelfTID, NumTasks);
            std::vector<uint32_t> arr(result.size);
            memmove(arr.data(), result.data, result.size * sizeof(uint32_t));
            free(result.data);

            kSearch(kth, arr, k, n, NumTasks);
            printf("kth element kSearch: %u\n", kth);
            free(arr.data());

            arr.resize(n);
            result = getWikiPartition(url, SelfTID, NumTasks);
            memmove(arr.data(), result.data, result.size * sizeof(uint32_t));
            free(result.data);

            heurQuickSelect(kth, arr, k, n, NumTasks);
            printf("kth element heur: %u\n", kth);
            free(arr.data());

            arr.resize(n);
            result = getWikiPartition(url, SelfTID, NumTasks);
            memmove(arr.data(), result.data, result.size * sizeof(uint32_t));
            free(result.data);

            quickSelect(kth, arr, k, n, NumTasks);
            printf("kth element quick: %u\n", kth);
            free(arr.data());
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Finalize();

        return 0;
    }

    ARRAY result = getWikiPartition(url, SelfTID, NumTasks);
    printf("Result size: %zu\n", result.size);

    std::vector<uint32_t> arr(result.size);
    MPI_Barrier(MPI_COMM_WORLD);

    arr.resize(result.size);

    bool exit_flag = false;
    for (size_t knew = 1; knew <= n; knew++)
    {
        if (SelfTID == 0)
            printf("k = %ld of %ld\n", knew, n);
        arr.resize(result.size);
        std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin());
        MPI_Barrier(MPI_COMM_WORLD);
        kSearch(kth, arr, knew, n, NumTasks);
        // printf("kSearch done\n");
        uint32_t kth1 = kth;
        arr.resize(result.size);
        std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin());
        MPI_Barrier(MPI_COMM_WORLD);
        heurQuickSelect(kth, arr, knew, n, NumTasks);
        // printf("heur done\n");
        uint32_t kth2 = kth;
        arr.resize(result.size);
        std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin());
        MPI_Barrier(MPI_COMM_WORLD);
        quickSelect(kth, arr, knew, n, NumTasks);
        // printf("quick done\n");
        uint32_t kth3 = kth;
        MPI_Barrier(MPI_COMM_WORLD);
        if (SelfTID == 0)
            if (kth1 != kth2 || kth1 != kth3 || kth2 != kth3)
            {
                printf("kth1: %u, kth2: %u, kth3: %u\n", kth1, kth2, kth3);
                exit_flag = true;
            }
        MPI_Bcast(&exit_flag, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        if (exit_flag)
            break;
    }
    return 0;

    arr.resize(result.size);
    std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin());
    for (size_t i = 0; i < arr.size(); i++)
    {
        if (arr[i] == 60346)
            printf("i: %ld\n", i);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // return 0;

    if (SelfTID == 0)
    {
        std::fstream file("kSearch.json", std::ios::out);
        arr.resize(result.size);
        std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin()); // copy to save time from re-reading the file
        kSearch(kth, arr, k, n, NumTasks);

        ankerl::nanobench::Bench()
            .minEpochIterations(1)
            .epochs(1)
            .run("kSearch", [&]
                 {  arr.resize(result.size);
                    std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin());
                    kSearch(kth, arr, k, n, NumTasks); })
            .render(ankerl::nanobench::templates::pyperf(), file);
    }
    else
    {
        arr.resize(result.size);
        std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin());
        kSearch(kth, arr, k, n, NumTasks);
        ankerl::nanobench::Bench()
            .minEpochIterations(1)
            .epochs(1)
            .output(nullptr)
            .run("kSearch", [&]
                 {  arr.resize(result.size);
                    std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin());
                    kSearch(kth, arr, k, n, NumTasks); });
    }

    if (SelfTID == 0)
        printf("kth element kSearch: %u\n", kth);

    MPI_Barrier(MPI_COMM_WORLD); // no need for a single barrier request, using Barrier_init, here, the use of the barrier is out of scope of the program (not included in the timings)

    if (SelfTID == 0)
    {
        std::fstream file("heurQuick.json", std::ios::out);

        ankerl::nanobench::Bench()
            .minEpochIterations(1)
            .epochs(1)
            .run("heurQuickSelect", [&]
                 {  arr.resize(result.size);
                    std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin());
                    heurQuickSelect(kth, arr, k, n, NumTasks); })
            .render(ankerl::nanobench::templates::pyperf(), file);
    }
    else
    {
        ankerl::nanobench::Bench()
            .minEpochIterations(1)
            .epochs(1)
            .output(nullptr)
            .run("heurQuickSelect", [&]
                 {  arr.resize(result.size);
                    std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin());
                    heurQuickSelect(kth, arr, k, n, NumTasks); });
    }

    if (SelfTID == 0)
        printf("kth element heur quick: %u\n", kth);

    MPI_Barrier(MPI_COMM_WORLD);

    if (SelfTID == 0)
    {
        std::fstream file("quick.json", std::ios::out);

        ankerl::nanobench::Bench()
            .minEpochIterations(1)
            .epochs(1)
            .run("quickSelect", [&]
                 {  arr.resize(result.size);
                    std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin());
                    quickSelect(kth, arr, k, n, NumTasks); })
            .render(ankerl::nanobench::templates::pyperf(), file);
    }
    else
    {
        ankerl::nanobench::Bench()
            .minEpochIterations(1)
            .epochs(1)
            .output(nullptr)
            .run("quickSelect", [&]
                 {  arr.resize(result.size);
                    std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin());
                    quickSelect(kth, arr, k, n, NumTasks); });
    }

    if (SelfTID == 0)
        printf("kth element quick: %u\n", kth);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}