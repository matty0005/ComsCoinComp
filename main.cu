
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

int main(int argc, char **argv) {
	print_usage();
	int n = 10;
	JOB ** jobs;
	BYTE buff[20];
	char size[20];
	char buff2[20];



    checkCudaErrors(cudaMallocManaged(&jobs, n * sizeof(JOB *)));
	printf("Checked for errors\n");



    for (int x = 0; x < 10; x++) {

        sprintf(buff2, "4501600%d", x);
		printf("Input: %s\n", buff2);

		memcpy(buff, buff2, strlen(buff2));

        sprintf((char *)size, "%d", x);

        jobs[x] = JOB_init(buff, strlen((char *)buff), size);
    }

	printf("Created jobs\n");

    // line = trim(line);
    // buff = get_file_data(line, &temp);
    
	printf("Start Jobs\n");
    pre_sha256();
    runJobs(jobs, n);

	cudaDeviceSynchronize();
	printf("Print Jobs\n");

	// printf("\t @ 0x%x \n", jobs[0]->data);
	// print_jobs(jobs, n);
	print_job(jobs[0]);
	printf("End Jobs\n");

	cudaDeviceReset();
	return 0;
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
// 	printf("--test\n");
// 	printf("%s  %s\n", hash_to_string(j->digest), j->fname);
// }

// void print_jobs(JOB ** jobs, int n) {
// 	printf("N: %d\n", n);
// 	for (int i = 0; i < n; i++)
// 	{
//         print_job(jobs[i]);
// 		// printf("@ %p JOB[%i] \n", jobs[i], i);
// 		// printf("\t @ 0x%p data = %x \n", jobs[i]->data, (jobs[i]->data == 0)? 0 : jobs[i]->data[0]);
// 		// printf("\t @ 0x%p size = %llu \n", &(jobs[i]->size), jobs[i]->size);
// 		// printf("\t @ 0x%p fname = %s \n", &(jobs[i]->fname), jobs[i]->fname);
// 		// printf("\t @ 0x%p digest = %s \n------\n", jobs[i]->digest, hash_to_string(jobs[i]->digest));
// 	}
// }

