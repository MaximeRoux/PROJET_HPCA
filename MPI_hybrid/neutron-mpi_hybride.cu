/*
 * Université Pierre et Marie Curie
 * Calcul de transport de neutrons
 * Version séquentielle
 */
//nvcc -o exec neutron-par.cu -O3 --generate-code arch =compute_35, code=sm_35 && ./exec
//nvcc -o exec neutron-par.cu -O3 --generate-code arch=compute_35,code=sm_35 && ./exec
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include<curand_kernel.h>
#include <thrust/remove.h>

#include<mpi.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <omp.h>

#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define THREADS_PER_BLOCK 1024
#define OUTPUT_FILE "/tmp/absorbed.dat"

char info[] = "\
Usage:\n\
    neutron-mpi_hybride H Nb C_c C_s\n\
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

float uniform_random_number(struct drand48_data* r) {
  double res = 0.0;
  drand48_r(r,&res);
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
__global__ void neutron_gpu(int n,int* r,int* t,int* b, float* absorbed,float c, float c_c, float c_s, float h, int nb_proc){
    // distance parcourue par le neutron avant la collision
    float L;
    // direction du neutron (0 <= d <= PI)
    float d;
    // variable aléatoire uniforme
    float u;
    // position de la particule (0 <= x <= h)
    float x;
    int j, old;
    unsigned int seed;
    curandState state;
  
    j = threadIdx.x+blockIdx.x*blockDim.x;
    seed = j;
  
  
    curand_init(seed, 0, 0, &state); 

    if(j<(n/nb_proc)){ 
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
					} 
	
					else if ((u = curand_uniform(&state)) < c_c / c) {
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
  int i;
  int j=0;
  float* absorbed;
  float* c_absorbed;
  float* g_absorbed;
  int *gpu_r, *gpu_t, *gpu_b;
  // int epth;
  int NB_BLOCK;
  int n_cpu, n_gpu;
  int r_aux, b_aux, t_aux;
 
 	int global_b,global_r,global_t;
 	int global_b_aux,global_r_aux,global_t_aux;
  unsigned int seed;
  float L;
  float d;
  float u;
  float x;
  float pn_cpu=0.1;
  struct drand48_data test;
  double finish_cpu,finish_gpu,start_cpu,start_gpu;
  int tmp=0;
  int my_rank, nb_proc;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nb_proc);
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
 	
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
  if (argc > 5)
    pn_cpu = atof(argv[5]);
  r = b = t = 0;
  c = c_c + c_s;

  n_cpu = (int)(pn_cpu * n);
  n_gpu = n - n_cpu;
 	printf("n_cpu=%d n_gpu=%d n=%d\n",n_cpu,n_gpu,n_cpu+n_gpu);
  // affichage des parametres pour verificatrion
  printf("Épaisseur de la plaque : %4.g\n", h);
  printf("Nombre d'échantillons  : %d\n", n);
  printf("C_c : %g\n", c_c);
  printf("C_s : %g\n", c_s);

 
  
  absorbed = (float *) calloc(n_gpu/nb_proc, sizeof(float));
 	c_absorbed = (float *) calloc(n_cpu/nb_proc, sizeof(float));
  // debut du chronometrage
 
  //Partie à parraléliser
  //echantillon par thread
  //epth = omp_get_max_threads();
  NB_BLOCK = (n_gpu+THREADS_PER_BLOCK*nb_proc-1)/(THREADS_PER_BLOCK*nb_proc);

  MPI_Barrier(MPI_COMM_WORLD);
  start = my_gettimeofday();
	
  #pragma omp parallel private (seed, x, L, u, d, tmp, test) shared(r,t,b)
  {
  	seed=omp_get_thread_num();
  	srand48_r(seed,&test);
  	
    #pragma omp master
    {    
      start_gpu = my_gettimeofday();
      cudaMalloc((void**)&g_absorbed, (n_gpu/nb_proc)*sizeof(float));
			cudaMalloc((void**)&gpu_b, sizeof(int));
			cudaMalloc((void**)&gpu_r, sizeof(int));
			cudaMalloc((void**)&gpu_t, sizeof(int));
      cudaMemcpy(gpu_r, &r, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(gpu_t, &t, sizeof(int), cudaMemcpyHostToDevice); 
			cudaMemcpy(gpu_b, &b, sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(g_absorbed, absorbed,(n_gpu/nb_proc)*sizeof(float), cudaMemcpyHostToDevice);
			
      neutron_gpu<<<NB_BLOCK,THREADS_PER_BLOCK>>>(n_gpu, gpu_r, gpu_t, gpu_b, g_absorbed, c,
                          c_c, c_s, h,nb_proc); 
      cudaDeviceSynchronize();

      cudaMemcpy(absorbed, g_absorbed,(n_gpu/nb_proc)*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(&r_aux, gpu_r, sizeof(int), cudaMemcpyDeviceToHost); 
			cudaMemcpy(&t_aux, gpu_t, sizeof(int), cudaMemcpyDeviceToHost); 
			cudaMemcpy(&b_aux, gpu_b, sizeof(int), cudaMemcpyDeviceToHost);
      finish_gpu = my_gettimeofday();
      printf("\nTemps total de calcul GPU: %.8g sec\n", finish_gpu - start_gpu);
    }
		

    
   	start_cpu = omp_get_wtime();
    #pragma omp for reduction(+:r) reduction(+:b) reduction(+:t)
    for (i = 0; i < (n_cpu)/nb_proc; i++) {
      d = 0.0;
      x = 0.0;
      while (1) {
   
				u = uniform_random_number(&test);
				L = -(1 / c) * log(u);
				x = x + L * cos(d);
				if (x < 0) {
				  r++;
				  break;
				}else if (x >= h) {
				  t++;
				  break;
				} else if ((u = uniform_random_number(&test)) < c_c / c) {
				  b++;
				  // fonctionne si on enleve tmp de private
				  // on laisse les deux ~9sec et ~53n/s
				  // sans le atomic ~8sec et ~61n/s
				  // sans le private tmp ~9sec et ~53n/s
				  // 8 threads  
				  #pragma omp atomic capture
				  tmp=j++;
				  /*if(tmp>=n_cpu)
				  	printf("tmp=%d\n",tmp);*/
				  c_absorbed[tmp] = x;
				  break;
				} else {
				  u = uniform_random_number(&test);
				  d = u * M_PI;
				}
			}
    }
    finish_cpu = omp_get_wtime();
  }
  
 	MPI_Reduce(&b, &global_b, 1, MPI_INT, MPI_SUM, 0,MPI_COMM_WORLD);
  MPI_Reduce(&t, &global_t, 1, MPI_INT, MPI_SUM, 0,MPI_COMM_WORLD);
  MPI_Reduce(&r, &global_r, 1, MPI_INT, MPI_SUM, 0,MPI_COMM_WORLD);
  
  MPI_Reduce(&b_aux, &global_b_aux, 1, MPI_INT, MPI_SUM, 0,MPI_COMM_WORLD);
  MPI_Reduce(&t_aux, &global_t_aux, 1, MPI_INT, MPI_SUM, 0,MPI_COMM_WORLD);
  MPI_Reduce(&r_aux, &global_r_aux, 1, MPI_INT, MPI_SUM, 0,MPI_COMM_WORLD);
  global_r += global_r_aux;
  global_b += global_b_aux;
  global_t += global_t_aux;
  MPI_Barrier(MPI_COMM_WORLD);
  finish = my_gettimeofday();
  if(my_rank==0){
  	printf("Nombre neutrons refléchis : %d\n",global_r);
  	printf("Nombre neutrons absorbés : %d\n",global_b);
  	printf("Nombre neutrons transmis : %d\n",global_t);
		printf("\nPourcentage des neutrons refléchis : %4.2g\n", (float) global_r / (float) n);
		printf("Pourcentage des neutrons absorbés : %4.2g\n", (float) global_b / (float) n);
		printf("Pourcentage des neutrons transmis : %4.2g\n", (float) global_t / (float) n);
		
		printf("\nTemps total de calcul C: %.8g sec\n", finish_cpu - start_cpu);
		printf("\nTemps total de calcul: %.8g sec\n", finish - start);
		printf("Millions de neutrons /s: %.2g\n", (double) n / ((finish - start)*1e6));

	}
  // ouverture du fichier pour ecrire les positions des neutrons absorbés
  /*
  FILE *f_handle = fopen(OUTPUT_FILE, "w");
  if (!f_handle) {
    fprintf(stderr, "Cannot open " OUTPUT_FILE "\n");
    exit(EXIT_FAILURE);
  }
  
  //float *new_end = thrust::remove_if(absorbed, absorbed+n, is_not_zero());
  for (j = 0; j < b_aux; j++){
    fprintf(f_handle, "%f\n", absorbed[j]);
  }
	for (j = b_aux; j < b; j++){
    fprintf(f_handle, "%f\n", c_absorbed[j-b_aux]);
  }
  fclose(f_handle);
  printf("Result written in " OUTPUT_FILE "\n");
  */
  cudaFree(g_absorbed);
  cudaFree(gpu_r);
  cudaFree(gpu_t);
  cudaFree(gpu_b);
   
  free(absorbed);

  return EXIT_SUCCESS;
}
