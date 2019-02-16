/*
 * Université Pierre et Marie Curie
 * Calcul de transport de neutrons
 * Version séquentielle
 */
//nvcc -o exec neutron-par.cu -O3 --generate-code arch =compute_35, code=sm_35 && ./exec 
//nvcc -o exec neutron-par.cu -O3 --generate-code arch=compute_35,code=sm_35 && ./exec

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include<curand_kernel.h>
#include <thrust/remove.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <omp.h>

#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define THREADS_PER_BLOCK 1024
#define OUTPUT_FILE "/tmp/hybrid-absorbed.dat"

char info[] = "\
Usage:\n\
    neutron-seq H Nb C_c C_s\n\
\n\
    H  : épaisseur de la plaque\n\
    Nb : nombre d'échantillons\n\
    C_c: composante absorbante\n\
    C_s: componente diffusante\n\
\n\
Exemple d'execution : \n\
    neutron-seq 1.0 500000000 0.5 0.5\n\
";

/*
 * générateur uniforme de nombres aléatoires dans l'intervalle [0,1)
 */
struct drand48_data alea_buffer;


struct is_not_zero
{
  __host__ __device__
  bool operator()(float x)
  {
    return  x == 0;
  }
};


struct is_even
{
  __host__ __device__
  bool operator()(const int x)
  {
    return (x % 2) == 0;
  }
};


void init_uniform_random_number() {
  srand48_r(0, &alea_buffer);
}

float uniform_random_number() {
  double res = 0.0; 
  drand48_r(&alea_buffer,&res);
  return res;
}


/*
 * notre gettimeofday()
 */
double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}

/*
 * kernel
 */ 
__global__ void neutron_gpu(int n,int* r,int* t,int* b, float* absorbed,float c, float c_c, float c_s, float h, int num,int nb_block){
  // distance parcourue par le neutron avant la collision
  float L;
  // direction du neutron (0 <= d <= PI)
  float d;
  // variable aléatoire uniforme
  float u;
  // position de la particule (0 <= x <= h)
  float x;
  //(n,r,t,b,absorbed,c,c_c,c_s,L,h,d,x,u)
  int j, old, next_ind;
  unsigned int seed;
  curandState state;
  
  next_ind = (num + 1) * blockDim.x * nb_block + threadIdx.x + blockIdx.x * blockDim.x;
  j = num * blockDim.x * nb_block + threadIdx.x + blockIdx.x * blockDim.x;
  seed = j;
  curand_init(seed, 0, 0, &state); 
  if((threadIdx.x+1)%1024==0)
	printf("threadcpu=%d && threadgpu=%d\n",num,j); 
  if(j < next_ind && j < n){ 
    d = 0.0;
    x = 0.0;

    while (1) {
      u = curand_uniform(&state);	
      L = -(1 / c) * log(u);
      x = x + L * cos(d);
      if (x < 0) {
	atomicAdd(r, 1);
	break;
      } else if (x >= h) {
	atomicAdd(t, 1);
	break;
      } else if ((u = curand_uniform(&state)) < c_c / c) {
	old = atomicAdd(b, 1);
	absorbed[old] = x;					
	break;
      } else {
	u = curand_uniform(&state);
	d = u * M_PI;
      }
    }
  }
}

/*
 * main()
 */
int main(int argc, char *argv[]) {
  // La distance moyenne entre les interactions neutron/atome est 1/c. 
  // c_c et c_s sont les composantes absorbantes et diffusantes de c. 
  float c, c_c, c_s;
  // épaisseur de la plaque
  float h;
  // nombre d'échantillons
  int n;
  // nombre de neutrons refléchis, absorbés et transmis
  int r, b, t;
  // chronometrage
  double start, finish;
  //int i, j = 0; // compteurs 
  int j,i;
  float* absorbed;
  float* g_absorbed;
  int *gpu_r, *gpu_t, *gpu_b;
  int nb_thread_cpu, NB_BLOCK;

  if( argc == 1)
    fprintf( stderr, "%s\n", info);

  // valeurs par defaut
  h = 1.0;
  n = 500000000;
  c_c = 0.5;
  c_s = 0.5;

  // recuperation des parametres
  if (argc > 1)
    h = atof(argv[1]);
  if (argc > 2)
    n = atoi(argv[2]);
  if (argc > 3)
    c_c = atof(argv[3]);
  if (argc > 4)
    c_s = atof(argv[4]);
  r = b = t = 0;
  c = c_c + c_s;
	
  // affichage des parametres pour verificatrion
  printf("Épaisseur de la plaque : %4.g\n", h);
  printf("Nombre d'échantillons  : %d\n", n);
  printf("C_c : %g\n", c_c);
  printf("C_s : %g\n", c_s);

  
  cudaMalloc((void**)&g_absorbed, n*sizeof(float));
  cudaMalloc((void**)&gpu_b, sizeof(int));
  cudaMalloc((void**)&gpu_r, sizeof(int));
  cudaMalloc((void**)&gpu_t, sizeof(int));
  
  cudaMemcpy(gpu_r, &r, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_t, &t, sizeof(int), cudaMemcpyHostToDevice);  
  cudaMemcpy(gpu_b, &b, sizeof(int), cudaMemcpyHostToDevice);
  
  absorbed = (float *) calloc(n, sizeof(float));
  cudaMemcpy(g_absorbed, absorbed,n*sizeof(float), cudaMemcpyHostToDevice);
  
  // debut du chronometrage
  start = my_gettimeofday();

  //Partie à parraléliser
  //echantillon par thread
  nb_thread_cpu = omp_get_max_threads();
  NB_BLOCK =ceil(n/(float)(THREADS_PER_BLOCK * nb_thread_cpu));
  printf("NB_BLOCK=%d && max_thread=%d\n",NB_BLOCK,nb_thread_cpu);
  #pragma omp parallel for
  for (i = 0; i < nb_thread_cpu; i++){
    neutron_gpu<<<NB_BLOCK,THREADS_PER_BLOCK>>>(n, gpu_r, gpu_t, gpu_b, g_absorbed, c, c_c, c_s, h,omp_get_thread_num(),NB_BLOCK);
  }
  
  cudaDeviceSynchronize();
  // fin du chronometrage  
  finish = my_gettimeofday();
  printf("num thread = %d \n",omp_get_thread_num());
  cudaMemcpy(absorbed, g_absorbed,n*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&r, gpu_r, sizeof(int), cudaMemcpyDeviceToHost);  
  cudaMemcpy(&t, gpu_t, sizeof(int), cudaMemcpyDeviceToHost);  
  cudaMemcpy(&b, gpu_b, sizeof(int), cudaMemcpyDeviceToHost);
  
  printf("\nPourcentage des neutrons refléchis : %4.2g\n", (float) r / (float) n);
  printf("Pourcentage des neutrons absorbés : %4.2g\n", (float) b / (float) n);
  printf("Pourcentage des neutrons transmis : %4.2g\n", (float) t / (float) n);
  printf("num thread = %d r=%d b=%d t=%d\n",omp_get_thread_num(),r,b,t);
  printf("\nTemps total de calcul: %.8g sec\n", finish - start);
  printf("Millions de neutrons /s: %.2g\n", (double) n / ((finish - start)*1e6));

  // ouverture du fichier pour ecrire les positions des neutrons absorbés
  FILE *f_handle = fopen(OUTPUT_FILE, "w");
  if (!f_handle) {
    fprintf(stderr, "Cannot open " OUTPUT_FILE "\n");
    exit(EXIT_FAILURE);
  }
  //float *new_end = thrust::remove_if(absorbed, absorbed+n, is_not_zero());
  for (j = 0; j < b; j++){
    fprintf(f_handle, "%f\n", absorbed[j]);
  }

  fclose(f_handle);
  printf("Result written in " OUTPUT_FILE "\n"); 
  cudaFree(g_absorbed);
  cudaFree(gpu_r);
  cudaFree(gpu_t);
  cudaFree(gpu_b);
	
  free(absorbed);

  return EXIT_SUCCESS;
}
