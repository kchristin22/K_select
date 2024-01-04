#include "getWiki.hpp"
#include <mpi.h>

// Write callback function for curl.
size_t write_callback(void *data, size_t size, size_t nmemb, void *destination)
{
  STREAM *stream = (STREAM *)destination;
  // printf("size: %zu, nmemb: %zu\n", size, nmemb);
  size_t realsize = size * nmemb;
  // If first time calling, allocate the needed memory.
  if (stream->data == NULL)
  {
    stream->data = (char *)malloc(stream->total_byte_size * sizeof(char));
  }
  if (stream->current_byte_size + realsize > stream->total_byte_size)
  {
    printf("ERROR: OVERFLOW\n");
    printf("Current size: %zu, Total size: %zu, realsize: %zu\n", stream->current_byte_size, stream->total_byte_size, realsize);
    exit(1);
  }
  memcpy(&(stream->data[stream->current_byte_size]), data, realsize);
  stream->current_byte_size += realsize;

  return realsize;
}

size_t getWikiInfo(const char *url, CURL *curl_handle)
{
  curl_global_init(CURL_GLOBAL_ALL);

  curl_handle = curl_easy_init();

  CURLcode res; // Error checking
  double content_length;

  if (curl_handle)
  {
    // Set up url
    curl_easy_setopt(curl_handle, CURLOPT_URL, url);
    // Get file total size. (No body yet)
    curl_easy_setopt(curl_handle, CURLOPT_NOBODY, 1L);
    // printf("Request1\n");
    res = curl_easy_perform(curl_handle);
    // printf("Request1 done\n");
    if (res != CURLE_OK)
    {
      printf("Error in curl_easy_perform()\n");
      exit(1);
    }
    else
    {
      res = curl_easy_getinfo(curl_handle, CURLINFO_CONTENT_LENGTH_DOWNLOAD, &content_length);
      if (res != CURLE_OK)
      {
        printf("Error in curl_easy_getinfo().\n");
        exit(1);
      }
    }
  }

  curl_easy_cleanup(curl_handle);
  curl_global_cleanup();

  // Got size, convert to size_t.
  return (size_t)content_length;
}

ARRAY getWikiPartition(const char *url, int world_rank, int world_size)
{
  // Input checking
  if (world_rank < 0 || world_rank >= world_size)
  {
    printf("World rank out of bounds.\n");
    exit(1);
  }
  CURL **curl_handle = (CURL **)malloc(world_size * sizeof(CURL *));
  CURL *local_curl_handle = curl_handle[world_rank];
  CURLcode res; // Error checking
  ARRAY result;

  curl_global_init(CURL_GLOBAL_ALL);
  MPI_Barrier(MPI_COMM_WORLD);
  local_curl_handle = curl_easy_init();
  if (local_curl_handle)
  {
    size_t file_byte_size = getWikiInfo(url, local_curl_handle);
    curl_easy_reset(local_curl_handle);
    // Got size, convert to size_t.
    printf("Got size: %zu\n", file_byte_size);
    // Round out to multiples of 4 since element size is 32 bits (ignore remainder)
    file_byte_size = (file_byte_size / 4) * 4;
    // Partition is done using 32 bit ints as elements
    size_t file_int_size = file_byte_size / 4;
    size_t start_byte = (file_int_size / world_size) * world_rank * 4;
    size_t end_byte = (file_int_size / world_size) * (world_rank + 1) * 4 - 1;
    // Last guy takes the remaining elements
    if (world_rank == (world_size - 1))
    {
      end_byte = file_byte_size - 1;
    }
    // Init downstream
    STREAM stream;
    stream.data = NULL;
    stream.current_byte_size = 0;
    stream.total_byte_size = end_byte - start_byte + 1; // total for this process
    char *range = (char *)malloc(100 * sizeof(char));
    sprintf(range, "%zu-%zu", start_byte, end_byte);
    printf("Range: %s\n", range);
    // Request body and set rest of options
    curl_easy_setopt(local_curl_handle, CURLOPT_URL, url);
    curl_easy_setopt(local_curl_handle, CURLOPT_RANGE, range);
    curl_easy_setopt(local_curl_handle, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(local_curl_handle, CURLOPT_WRITEDATA, (void *)&stream);
    // Get data..
    // printf("Request2\n");
    for (int i = 0; i < world_size; i++)
    {
      if (world_rank == i)
      {
        res = curl_easy_perform(local_curl_handle);
        // printf("Request2 done\n");
        if (res != CURLE_OK)
        {
          printf("Problem in data reception\n");
          exit(1);
        }
        // Check if you got everything you asked for.
        printf("Sizes: %zu %zu of process: %d\n", stream.current_byte_size, stream.total_byte_size, world_rank);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    result.size = stream.current_byte_size / 4;
    result.data = (uint32_t *)malloc(result.size * sizeof(uint32_t));
    result.data = (uint32_t *)stream.data;

    // Clean up and pass everything to the array struct.
    curl_easy_cleanup(local_curl_handle);
    MPI_Barrier(MPI_COMM_WORLD);
    curl_global_cleanup();
    free(range);
  }
  return result;
}
