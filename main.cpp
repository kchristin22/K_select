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

void kCorrect(std::vector<uint32_t> &arr, const std::vector<uint32_t> &sorted_arr, const ARRAY &result, const size_t start, const size_t end, const int SelfTID, const size_t NumTasks)
{
    size_t n = sorted_arr.size();
    uint32_t kth1 = 0, kth2 = 0, kth3 = 0;
    bool exit_flag = false;
    for (size_t k = start; k <= end; k++)
    {
        if (SelfTID == 0)
            printf("k = %ld of %ld\n", k, n);
        arr.resize(result.size);
        std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin());
        MPI_Barrier(MPI_COMM_WORLD);
        kSearch(kth1, arr, k, n, NumTasks);
        // printf("kSearch done\n");
        arr.resize(result.size);
        std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin());
        MPI_Barrier(MPI_COMM_WORLD);
        heurQuickSelect(kth2, arr, k, n, NumTasks);
        // printf("heur done\n");
        arr.resize(result.size);
        std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin());
        MPI_Barrier(MPI_COMM_WORLD);
        quickSelect(kth3, arr, k, n, NumTasks);
        // printf("quick done\n");
        MPI_Barrier(MPI_COMM_WORLD);
        if (SelfTID == 0)
        {
            uint32_t kthCorrect = sorted_arr[k - 1];
            if (kth1 != kth2 || kth1 != kth3 || kth2 != kth3 || kth1 != kthCorrect)
            {
                printf("kthCorrect: %u, kth1: %u, kth2: %u, kth3: %u\n", kthCorrect, kth1, kth2, kth3);
                exit_flag = true;
            }
        }
        MPI_Bcast(&exit_flag, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        if (exit_flag)
            break;
    }
    return;
}

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

    std::string fileName = "sorted_data.txt";
    std::vector<uint32_t> sorted_arr(n);
    std::ifstream file(fileName);

    size_t index = 0;

    while (file >> sorted_arr[index])
        index++;

    printf("Sorted array size: %ld\n", index);

    if (SelfTID == 0)
        printf("kth = %u\n", sorted_arr[k - 1]);

    std::vector<uint32_t> arr(result.size);

    // kCorrect(arr, sorted_arr, result, 1, n, SelfTID, NumTasks); // call this if you want to validate a range of k's

    if (SelfTID == 0)
    {
        std::fstream file("std_copy.json", std::ios::out);

        ankerl::nanobench::Bench()
            .minEpochIterations(10)
            .epochs(5)
            .run("std_copy", [&]
                 {  arr.resize(result.size);
                    std::copy(result.data, result.data + result.size, arr.begin()); })
            .render(ankerl::nanobench::templates::pyperf(), file);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (SelfTID == 0)
    {
        std::fstream file("kSearch.json", std::ios::out);

        ankerl::nanobench::Bench()
            .minEpochIterations(10)
            .epochs(5)
            .run("kSearch", [&]
                 {  arr.resize(result.size);
                    std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin());
                    kSearch(kth, arr, k, n, NumTasks); })
            .render(ankerl::nanobench::templates::pyperf(), file);
    }
    else
    {
        ankerl::nanobench::Bench()
            .minEpochIterations(10)
            .epochs(5)
            .output(nullptr)
            .run("kSearch", [&]
                 {  arr.resize(result.size);
                    std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin());
                    kSearch(kth, arr, k, n, NumTasks); });
    }

    if (SelfTID == 0)
    {
        if (kth != sorted_arr[k - 1])
            printf("Error! kSearch returned %u, while correct kth is %u\n", kth, sorted_arr[k - 1]);
        else
            printf("kth element kSearch: %u\n", kth);
    }

    MPI_Barrier(MPI_COMM_WORLD); // no need for a single barrier request, using Barrier_init, here, the use of the barrier is out of scope of the program (not included in the timings)

    if (SelfTID == 0)
    {
        std::fstream file("heurQuick.json", std::ios::out);

        ankerl::nanobench::Bench()
            .minEpochIterations(10)
            .epochs(5)
            .run("heurQuickSelect", [&]
                 {  arr.resize(result.size);
                    std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin());
                    heurQuickSelect(kth, arr, k, n, NumTasks); })
            .render(ankerl::nanobench::templates::pyperf(), file);
    }
    else
    {
        ankerl::nanobench::Bench()
            .minEpochIterations(10)
            .epochs(5)
            .output(nullptr)
            .run("heurQuickSelect", [&]
                 {  arr.resize(result.size);
                    std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin());
                    heurQuickSelect(kth, arr, k, n, NumTasks); });
    }

    if (SelfTID == 0)
    {
        if (kth != sorted_arr[k - 1])
            printf("Error! Heur quickselect returned %u, while correct kth is %u\n", kth, sorted_arr[k - 1]);
        else
            printf("kth element heur quick: %u\n", kth);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (SelfTID == 0)
    {
        std::fstream file("quick.json", std::ios::out);

        ankerl::nanobench::Bench()
            .minEpochIterations(10)
            .epochs(5)
            .run("quickSelect", [&]
                 {  arr.resize(result.size);
                    std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin());
                    quickSelect(kth, arr, k, n, NumTasks); })
            .render(ankerl::nanobench::templates::pyperf(), file);
    }
    else
    {
        ankerl::nanobench::Bench()
            .minEpochIterations(10)
            .epochs(5)
            .output(nullptr)
            .run("quickSelect", [&]
                 {  arr.resize(result.size);
                    std::copy(std::execution::par_unseq, result.data, result.data + result.size, arr.begin());
                    quickSelect(kth, arr, k, n, NumTasks); });
    }

    if (SelfTID == 0)
    {
        if (kth != sorted_arr[k - 1])
            printf("Error! Quickselect returned %u, while correct kth is %u\n", kth, sorted_arr[k - 1]);
        else
            printf("kth element quick: %u\n", kth);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}