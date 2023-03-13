
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <unistd.h>
#include <cuda.h>
#include "sha256.cuh"
// #include <dirent.h>
#include <ctype.h>
#include <BaseTsd.h>

__global__ void sha256_cuda(JOB ** jobs, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// perform sha256 calculation here
	if (i < n){
		SHA256_CTX ctx;
		sha256_init(&ctx);
		sha256_update(&ctx, jobs[i]->data, jobs[i]->size);
		sha256_final(&ctx, jobs[i]->digest);
	}
}

void pre_sha256() {
	// compy symbols
	checkCudaErrors(cudaMemcpyToSymbol(dev_k, host_k, sizeof(host_k), 0, cudaMemcpyHostToDevice));
}


void runJobs(JOB ** jobs, int n){
	int blockSize = 4;
	int numBlocks = (n + blockSize - 1) / blockSize;
	sha256_cuda <<< numBlocks, blockSize >>> (jobs, n);
}


JOB * JOB_init(BYTE * data, long size, char * fname) {
	JOB * j;
	checkCudaErrors(cudaMallocManaged(&j, sizeof(JOB)));	//j = (JOB *)malloc(sizeof(JOB));
	checkCudaErrors(cudaMallocManaged(&(j->data), size));

	j->data = data;
	j->size = size;
	for (int i = 0; i < 64; i++)
	{
		j->digest[i] = 0xff;
	}
	strcpy(j->fname, fname);
	return j;
}

void print_usage(){
	printf("Usage: CudaSHA256...\n");
	printf("Calculate sha256 hash of given input\n\n");
}





// char * hash_to_string(BYTE * buff) {
// 	char * string = (char *)malloc(70);
// 	int k, i;
// 	for (i = 0, k = 0; i < 32; i++, k+= 2)
// 	{
// 		sprintf(string + k, "%.2x", buff[i]);
// 		//printf("%02x", buff[i]);
// 	}
// 	string[64] = 0;
// 	return string;
// }

// void print_job(JOB * j){
// 	// printf("--test\n");
// 	printf("%s  %s\n", hash_to_string(j->digest), j->fname);
// }




void print_jobs_with_zero(JOB ** jobs, int n, int n_zero) {
	for (int i = 0; i < n; i++) {
		bool show = true;

		for (int j = 0; j < n_zero; j++) {
			if (jobs[i]->digest[j] != 0) {
				show = false;
			}
		}

		if (show) {
	        print_job(jobs[i]);
		}
	}
}

void prog(int start, int n) {
	JOB ** jobs;
	char size[20];
	char buff2[20];


    checkCudaErrors(cudaMallocManaged(&jobs, n * sizeof(JOB *)));

    for (int x = 0; x < n; x++) {
        sprintf(buff2, "45801600%X", start + x);
		BYTE *buff;

		checkCudaErrors(cudaMallocManaged(&buff, strlen(buff2)*sizeof(char)));
		memcpy(buff, buff2, strlen(buff2));
        sprintf((char *)size, "%X", start + x);

        jobs[x] = JOB_init(buff, strlen((char *)buff), size);
    }

	printf("Created jobs\n");

    pre_sha256();
    runJobs(jobs, n);

	cudaDeviceSynchronize();

	// print_jobs(jobs, n);
	print_jobs_with_zero(jobs, n,2);

	cudaDeviceReset();
}

int main(int argc, char **argv) {
	print_usage();


	int n = 1000; // Number of batches
	int inc = 5000; // Number to do at a time

	int x = 0; // Start value

    for (int i = 0; i < n; i++) {
		prog(x, inc);
		printf("Max value: %d\n", x);

		x += inc;
	}


	return 0;
}