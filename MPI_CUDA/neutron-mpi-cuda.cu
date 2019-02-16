/*
 * Université Pierre et Marie Curie
 * Calcul de transport de neutrons
 * Version séquentielle
 */

// nvcc -I/usr/mpi/gcc/openmpi-1.4.6/include -L/usr/mpi/gcc/openmpi-1.4.6/lib64 -lmpi neutron-mpi.cu -o neutron-mpi
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include<curand_kernel.h>
#include <thrust/remove.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define THREADS_PER_BLOCK 1024
#define OUTPUT_FILE "/tmp/mpi-absorbed.dat"

char info[] = "\
Usage:\n\
    neutron-mpi-cuda H Nb C_c C_s\n\
\n\
    H  : épaisseur de la plaque\n\
    Nb : nombre d'échantillons\n\
    C_c: composante absorbante\n\
    C_s: componente diffusante\n\
\n\
Exemple d'execution : \n\
    neutron-mpi 1.0 500000000 0.5 0.5\n\
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
 * main()
 */
 
__global__ void neutron_gpu(int n,int* r,int* t,int* b, float* absorbed,float c, float c_c, float c_s, float h, int my_rank, int nb_proc){
  // distance parcourue par le neutron avant la collision
  float L;
  // direction du neutron (0 <= d <= PI)
  float d;
  // variable aléatoire uniforme
  float u;
  // position de la particule (0 <= x <= h)
  float x;
  //(n,r,t,b,absorbed,c,c_c,c_s,L,h,d,x,u)
  int j, old;
  unsigned int seed;
  curandState state;
  
  j = threadIdx.x+blockIdx.x*blockDim.x;
  seed = j;
  
  
  curand_init(seed, 0, 0, &state); 
  /*if (j == 0)
			printf(" j=%d r=%d t=%d b=%d\n",j,*r, *t, *b);*/
  if(j<n/nb_proc){ 
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
					
					/* if(absorbed[*b]==0){
						printf("x=%f et *b=%d\n",x,*b);
					} */
					
					break;
			} else {
					u = curand_uniform(&state);
					d = u * M_PI;
			}
		 }

  }
} 
int main(int argc, char *argv[]) {
  // La distance moyenne entre les interactions neutron/atome est 1/c. 
  // c_c et c_s sont les composantes absorbantes et diffusantes de c. 
  float c, c_c, c_s;
  // épaisseur de la plaque
  float h;
  
  // nombre d'échantillons
  int n;
  // nombre de neutrons refléchis, absorbés et transmis
  int r, b, t,global_b,global_r,global_t;
  // chronometrage
  double start, finish;
  //int i, j = 0; // compteurs 
	
	float* g_absorbed;
  int *gpu_r, *gpu_t, *gpu_b;

	int my_rank, nb_proc;

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nb_proc);
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);

  
  if( argc == 1)
    fprintf( stderr, "%s\n", info);

  // valeurs par defaut
  h = 1.0;
  n = 500000000;//500000000
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

	
	float* sub_absorbed = (float *)calloc(n/nb_proc, sizeof(float));
	int NB_BLOCK=(n+THREADS_PER_BLOCK*nb_proc-1)/(THREADS_PER_BLOCK*nb_proc);

	//Barriere pour décompter temps 
	MPI_Barrier(MPI_COMM_WORLD);
	start = my_gettimeofday();	
	
	//MPI Scatter pour diviser le tableau sur nb_proc mais trop couteux
	//MPI_Scatter(absorbed,n/nb_proc, MPI_FLOAT, sub_absorbed,n/nb_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);
	
	//ALLOCATION GPU
	cudaMalloc((void**)&g_absorbed, (n/nb_proc)*sizeof(float));
  cudaMalloc((void**)&gpu_b, sizeof(int));
  cudaMalloc((void**)&gpu_r, sizeof(int));
  cudaMalloc((void**)&gpu_t, sizeof(int));
	
  //COPIE CPU -> GPU
  cudaMemcpy(gpu_r, &r, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_t, &t, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_b, &b, sizeof(int), cudaMemcpyHostToDevice);  
	cudaMemcpy(g_absorbed, sub_absorbed,(n/nb_proc)*sizeof(float), cudaMemcpyHostToDevice);

  
	//APPEL AU KERNEL  
  neutron_gpu<<<NB_BLOCK,THREADS_PER_BLOCK>>>(n, gpu_r, gpu_t, gpu_b, g_absorbed, c, c_c, c_s, h, my_rank, nb_proc);
  
  cudaDeviceSynchronize();
  // fin du chronometrage
  
  
  //COPIE GPU -> CPU
  cudaMemcpy(&b, gpu_b, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(sub_absorbed, g_absorbed,b*sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(&r, gpu_r, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&t, gpu_t, sizeof(int), cudaMemcpyDeviceToHost);  
  
  //MPI_Gather pour rassembler le tableau et ecrire le tableau dans le fichier plus facilement mais trop couteux aussi
  //MPI_Gather(sub_absorbed,n/nb_proc, MPI_FLOAT, absorbed,n/nb_proc, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  
  //MPI_Reduce pour somme de r b et t
  
  MPI_Reduce(&b, &global_b, 1, MPI_INT, MPI_SUM, 0,MPI_COMM_WORLD);
  MPI_Reduce(&t, &global_t, 1, MPI_INT, MPI_SUM, 0,MPI_COMM_WORLD);
  MPI_Reduce(&r, &global_r, 1, MPI_INT, MPI_SUM, 0,MPI_COMM_WORLD);
  
  MPI_Barrier(MPI_COMM_WORLD);
  //Barriere pour décompter temps
  finish = my_gettimeofday();
  
  
  if(my_rank==0){
  	printf("Nombre neutrons refléchis : %d\n",global_r);
  	printf("Nombre neutrons absorbés : %d\n",global_b);
  	printf("Nombre neutrons transmis : %d\n",global_t);
		printf("\nPourcentage des neutrons refléchis : %4.2g\n", (float) global_r / (float) n);
		printf("Pourcentage des neutrons absorbés : %4.2g\n", (float) global_b / (float) n);
		printf("Pourcentage des neutrons transmis : %4.2g\n", (float) global_t / (float) n);

		printf("\nTemps total de calcul: %.8g sec\n", finish - start);
		printf("Millions de neutrons /s: %.2g\n", (double) n / ((finish - start)*1e6));
	
	}
	// ouverture du fichier pour ecrire les positions des neutrons absorbés
	/*
	MPI_Status status;
    MPI_File fh;
  
  
    MPI_File_open(MPI_COMM_SELF, OUTPUT_FILE,MPI_MODE_CREATE | MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
    MPI_Offset displace=my_rank*sizeof(char)*42*b*sizeof("\n");
	MPI_File_set_view (fh, displace,MPI_FLOAT,MPI_FLOAT, "native" ,MPI_INFO_NULL);

    for (int i=0; i < b; i++){
      char buf[42];
      //fprintf(f,"%d \n",i);
      snprintf(buf,42,"%f \n",sub_absorbed[i]);
      MPI_File_write(fh,buf,strlen(buf), MPI_CHAR,&status);
	}
	
	MPI_File_close(&fh);
	printf("Result written in " OUTPUT_FILE "\n");
	*/
	cudaFree(g_absorbed);
	cudaFree(gpu_r);
	cudaFree(gpu_t);
	cudaFree(gpu_b);

	MPI_Finalize();
  return EXIT_SUCCESS;
}

