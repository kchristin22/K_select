# K_select

The purpose of this project is to find the kth smallest element of a given vector that is too large
to fit in one machine, thus making the use of MPI necessary. Three different versions have been implemented for this purpose: k-Search, Heuristic Quickselect and Quickselect.

### Branches
* `main`: reads an array from a file and scatters it to MPI processes
* `nanobench`: performs benchmarks using the [nanobench](https://github.com/andreas-abel/nanoBench) tool. Includes the project's report.
* `hpc_bench`: used to perform the benchmarks on [Aristotle HPC](https://hpc.auth.gr/pun/sys/dashboard/). The results mentioned in the report are also contained in corresponding folders. Due to the use of an older version of the gcc compiler, the following changes had to be made in comparison to the `nanobench` branch:
  * CMake version required reduced
  * `-std=c++20` -> `-std=c++2a`
  * removal of the `execution` header and its functions from `main.cpp`  

### How to build and run the repo

To successfully compile and run the program you need to execute the follow commands:

1. `rm -f -r build/`
2. `mkdir build`
3. `cd build/`
4. `cmake ..`
5. `cmake --build .`
6. `cd bin/`
7. `cp ../../sorted_data.txt .` # the file where the sorted array is placed, in order to check the correctness of the results. In case of the [default url](https://dumps.wikimedia.org/other/static_html_dumps/current/el/wikipedia-el-html.tar.7z), the sorted array can be found [here](https://drive.google.com/file/d/14oI-r5W7kl2FcGCbQ1Udg1GPARdabDDE/view?usp=sharing). The file name "sorted_data.txt" is hardcoded in the program. If the file is larger than the one denoted by the url, the program will fail. If you don't want to evaluate the correctness of the program, you may not include the "sorted_data.txt" file in your bin folder or comment out lines 154-155 in `main.cpp`
8. `mpirun ./output ${k} ${url}` # the arguments are optional

### Notes
* MPICH has a bug regarding performing reduction with elements of type uint32_t. Make sure to use OpenMPI instead.
* You may need to alter the minimum version of CMake needed in CMakeLists.txt
* If CMake doesn't find the curl package even though curl is installed, check out this [forum](https://stackoverflow.com/questions/34914944/could-not-find-curl-missing-curl-library-curl-include-dir-on-cmake)
* The correct answer is displayed, along with the result of each implementation. If the results don't match, the user gets notified accordingly.
* There is a function in main that can be used for contiguous values of k
* For further statistics regarding each HPC Job result you may use: `python3 -m pyperf stats file1.json`
* To compare two results you may use: `python3 -m pyperf compare_to --table file1.json file2.json`



