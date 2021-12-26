// %%writefile cuda_filtro.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "sod/sod.h"

#define MAX_H 4320
#define MAX_W 8192
#define MAX_INTERVAL 2048
#define INIT_KERNEL float kernel[3][3] = {{-2,-1,0}, {-1, 1,1}, {0,1,2}};
//#define INIT_KERNEL float kernel[3][3] = {{0,1,0}, {1, 4,1}, {0,1,0}};
//#define INIT_KERNEL float kernel[3][3] = {{0,1,0}, {1, 4,1}, {0,1,0}};


char* INIMAGE;
char* OUTIMAGE;
int ARG;
int THREADSNUM;
int BLOCKS;
int THREADSGPU;
int INTERVAL[MAX_INTERVAL][2];


__global__ 
void filter(int d_interval[MAX_INTERVAL][2], float (*d_board)[MAX_W], float (*d_output)[MAX_W], int *d_W, int *d_blocks) {
    int ID = blockIdx.x * blockDim.x + threadIdx.x;
    if (ID < (*d_blocks)) {        
        INIT_KERNEL;        
        int from = d_interval[ID][0];//Interval
        int to = d_interval[ID][1];
        for(int y = from; y <= to; ++y) {
            for(int x = 1; x < (*d_W)-1; ++x) {
                float sum = 0.0;
                for(int ky = -1; ky <= 1; ++ky) {
                    for(int kx = -1; kx <= 1; ++kx) {
                        float val = d_board[x+kx][y+ky];
                        sum += kernel[ky+1][kx+1] * val; //Filter
                    }
                }
                d_output[x][y] = abs(sum);
            }
        }
    }
}

double Time() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main(int argc, char *argv[]) {

    //Paths
    INIMAGE = argv[1];
    OUTIMAGE = argv[2];
    ARG = atof(argv[3]);
    //Size
    BLOCKS = atoi(argv[4]);
    THREADSGPU = atoi(argv[5]);
    THREADSNUM = BLOCKS * THREADSGPU;  
    sod_img imgIn;
    sod_img imgOut; 

    //Memory
    imgIn = sod_img_load_from_file(INIMAGE, SOD_IMG_COLOR);
    imgOut = sod_img_load_from_file(INIMAGE, SOD_IMG_COLOR);
    
    if (imgIn.data == 0) { // Image validation       
        printf("Image not found %s\n", INIMAGE);
        return 0;
    }

    //Intervals
    int factor = imgIn.h / THREADSNUM;
    int last = 1;
    for(int i = 0; i < THREADSNUM; ++i) {
        INTERVAL[i][0] = last;
        if(i != (THREADSNUM-1)) {
            INTERVAL[i][1] = last + factor-1;
        } else {
            INTERVAL[i][1] = imgIn.h - 1;
        }
        last = INTERVAL[i][1] + 1;
    }

    //printf("width=%d height=%d\n", imgIn.w, imgIn.h);

    // memory
    // host board
    float (*board)[MAX_W] = (float (*)[MAX_W]) malloc(MAX_H*MAX_W*sizeof(float));

    for(int y = 0; y <= imgIn.h; ++y) {
        for(int x = 0; x < imgIn.w; ++x) {
            float val = sod_img_get_pixel(imgIn, x, y, 0); // RED
            board[x][y] = val;
        }
    }

    // device board
    float (*d_board)[MAX_W];
    cudaMalloc(&d_board, MAX_H*MAX_W*sizeof(float));
    cudaMemcpy(d_board, board, MAX_H*MAX_W*sizeof(float), cudaMemcpyHostToDevice);

    // Device output
    float (*d_output)[MAX_W];
    cudaMalloc(&d_output, MAX_H*MAX_W*sizeof(float));

    // Device Interval
    int (*d_intervalo)[2];
    cudaMalloc(&d_intervalo, MAX_INTERVAL*2*sizeof(int));
    cudaMemcpy(d_intervalo, INTERVAL, MAX_INTERVAL*2*sizeof(int), cudaMemcpyHostToDevice);

    // Device W
    int *d_W;
    int *tmp_W;
    int tmp = imgIn.w;
    tmp_W = &tmp;
    cudaMalloc(&d_W, sizeof(int));
    cudaMemcpy(d_W, tmp_W, sizeof(int), cudaMemcpyHostToDevice);

    int *d_blocks;
    int *threads;
    threads = &THREADSNUM;
    cudaMalloc(&d_blocks, sizeof(int));
    cudaMemcpy(d_blocks, threads, sizeof(int), cudaMemcpyHostToDevice);

    double start = Time(); // Get start time
    filter<<<BLOCKS, THREADSGPU>>>(d_intervalo, d_board, d_output, d_W, d_blocks); // Run filter
    cudaDeviceSynchronize();
    double stop = Time();

    cudaMemcpy(board, d_output, MAX_H*MAX_W*sizeof(float), cudaMemcpyDeviceToHost);

    // To black and white
    for(int y = 0; y <= imgIn.h; ++y) {
        for(int x = 0; x < imgIn.w; ++x) {
            float val = board[x][y];
            sod_img_set_pixel(imgOut, x, y, 0, abs(val));
            sod_img_set_pixel(imgOut, x, y, 1, abs(val));
            sod_img_set_pixel(imgOut, x, y, 2, abs(val));
        }
    }        
    sod_img_save_as_png(imgOut, OUTIMAGE);// Save image

    // Free memory
    sod_free_image(imgIn);
    sod_free_image(imgOut);

    double time_elapsed = stop - start;

    // Time log
    printf("\nTime: %.8f blocks:%d, threads:%d\n", time_elapsed, BLOCKS, THREADSGPU);
    fflush(stdout);

    return 0;
}