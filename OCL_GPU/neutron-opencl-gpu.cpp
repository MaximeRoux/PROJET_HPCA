#include <iostream>
#include <vector>
#include <string>
#include <sys/time.h>
#include <stdio.h>
#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#define OUTPUT_FILE "/tmp/ocl-absorbed.dat"
// Compute c = a + b.
/*
static const char source[] =
    "#if defined(cl_khr_fp64)\n"
    "#  pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
    "#elif defined(cl_amd_fp64)\n"
    "#  pragma OPENCL EXTENSION cl_amd_fp64: enable\n"
    "#else\n"
    "#  error double precision is not supported\n"
    "#endif\n"
    "kernel void add(\n"
    "       ulong n,\n"
    "       global const double *a,\n"
    "       global const double *b,\n"
    "       global double *c\n"
    "       )\n"
    "{\n"
    "    size_t i = get_global_id(0);\n"
    "    if (i < n) {\n"
    "       c[i] = a[i] + b[i];\n"
    "    }\n"
    "}\n";
*/
#define CL_ERR_TO_STR(err) case err: return #err

char const * oclGetErrorString(cl_int const err){
  switch(err) {
      CL_ERR_TO_STR(CL_SUCCESS);
      CL_ERR_TO_STR(CL_DEVICE_NOT_FOUND);
      CL_ERR_TO_STR(CL_DEVICE_NOT_AVAILABLE);
      CL_ERR_TO_STR(CL_COMPILER_NOT_AVAILABLE);
      CL_ERR_TO_STR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
      CL_ERR_TO_STR(CL_OUT_OF_RESOURCES);
      CL_ERR_TO_STR(CL_OUT_OF_HOST_MEMORY);
      CL_ERR_TO_STR(CL_PROFILING_INFO_NOT_AVAILABLE);
      CL_ERR_TO_STR(CL_MEM_COPY_OVERLAP);
      CL_ERR_TO_STR(CL_IMAGE_FORMAT_MISMATCH);
      CL_ERR_TO_STR(CL_IMAGE_FORMAT_NOT_SUPPORTED);
      CL_ERR_TO_STR(CL_BUILD_PROGRAM_FAILURE);
      CL_ERR_TO_STR(CL_MAP_FAILURE);
      CL_ERR_TO_STR(CL_MISALIGNED_SUB_BUFFER_OFFSET);
      CL_ERR_TO_STR(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
      CL_ERR_TO_STR(CL_COMPILE_PROGRAM_FAILURE);
      CL_ERR_TO_STR(CL_LINKER_NOT_AVAILABLE);
      CL_ERR_TO_STR(CL_LINK_PROGRAM_FAILURE);
      CL_ERR_TO_STR(CL_DEVICE_PARTITION_FAILED);
      CL_ERR_TO_STR(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
      CL_ERR_TO_STR(CL_INVALID_VALUE);
      CL_ERR_TO_STR(CL_INVALID_DEVICE_TYPE);
      CL_ERR_TO_STR(CL_INVALID_PLATFORM);
      CL_ERR_TO_STR(CL_INVALID_DEVICE);
      CL_ERR_TO_STR(CL_INVALID_CONTEXT);
      CL_ERR_TO_STR(CL_INVALID_QUEUE_PROPERTIES);
      CL_ERR_TO_STR(CL_INVALID_COMMAND_QUEUE);
      CL_ERR_TO_STR(CL_INVALID_HOST_PTR);
      CL_ERR_TO_STR(CL_INVALID_MEM_OBJECT);
      CL_ERR_TO_STR(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
      CL_ERR_TO_STR(CL_INVALID_IMAGE_SIZE);
      CL_ERR_TO_STR(CL_INVALID_SAMPLER);
      CL_ERR_TO_STR(CL_INVALID_BINARY);
      CL_ERR_TO_STR(CL_INVALID_BUILD_OPTIONS);
      CL_ERR_TO_STR(CL_INVALID_PROGRAM);
      CL_ERR_TO_STR(CL_INVALID_PROGRAM_EXECUTABLE);
      CL_ERR_TO_STR(CL_INVALID_KERNEL_NAME);
      CL_ERR_TO_STR(CL_INVALID_KERNEL_DEFINITION);
      CL_ERR_TO_STR(CL_INVALID_KERNEL);
      CL_ERR_TO_STR(CL_INVALID_ARG_INDEX);
      CL_ERR_TO_STR(CL_INVALID_ARG_VALUE);
      CL_ERR_TO_STR(CL_INVALID_ARG_SIZE);
      CL_ERR_TO_STR(CL_INVALID_KERNEL_ARGS);
      CL_ERR_TO_STR(CL_INVALID_WORK_DIMENSION);
      CL_ERR_TO_STR(CL_INVALID_WORK_GROUP_SIZE);
      CL_ERR_TO_STR(CL_INVALID_WORK_ITEM_SIZE);
      CL_ERR_TO_STR(CL_INVALID_GLOBAL_OFFSET);
      CL_ERR_TO_STR(CL_INVALID_EVENT_WAIT_LIST);
      CL_ERR_TO_STR(CL_INVALID_EVENT);
      CL_ERR_TO_STR(CL_INVALID_OPERATION);
      CL_ERR_TO_STR(CL_INVALID_GL_OBJECT);
      CL_ERR_TO_STR(CL_INVALID_BUFFER_SIZE);
      CL_ERR_TO_STR(CL_INVALID_MIP_LEVEL);
      CL_ERR_TO_STR(CL_INVALID_GLOBAL_WORK_SIZE);
      CL_ERR_TO_STR(CL_INVALID_PROPERTY);
      CL_ERR_TO_STR(CL_INVALID_IMAGE_DESCRIPTOR);
      CL_ERR_TO_STR(CL_INVALID_COMPILER_OPTIONS);
      CL_ERR_TO_STR(CL_INVALID_LINKER_OPTIONS);
      CL_ERR_TO_STR(CL_INVALID_DEVICE_PARTITION_COUNT);
      //CL_ERR_TO_STR(CL_INVALID_PIPE_SIZE); // OCL 2.0
      //CL_ERR_TO_STR(CL_INVALID_DEaVICE_QUEUE); // OCL 2.0

    default:
      return "UNKNOWN ERROR CODE";
    }
}





char* oclLoadSource(const char* filename){
  FILE* fd = NULL;
  size_t size;

  // open the OpenCL source code file
  if((fd = fopen(filename, "rb"))==NULL){
        printf("error: cl file not found\n"); exit(-1);
  }
  fseek(fd, 0, SEEK_END);
  size = ftell(fd);
  fseek(fd, 0, SEEK_SET);

  char* clsource = (char *)malloc(size + 1); 
  if (fread(clsource, size, 1, fd) != 1){
    fclose(fd); free(clsource); return NULL;
  }
  fclose(fd);
  clsource[size] = '\0';

  return clsource;
}

char info[] = "\
Usage:\n\
    neutron-opencl-gpu H Nb C_c C_s\n\
\n\
    H  : épaisseur de la plaque\n\
    Nb : nombre d'échantillons\n\
    C_c: composante absorbante\n\
    C_s: componente diffusante\n\
\n\
Exemple d'execution : \n\
    neutron-seq 1.0 500000000 0.5 0.5\n\
";


double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}




int main(int argc, char *argv[]) {
	char *source;
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
	int j;
	int nb_op;

	nb_op=2500;
	h = 1.0;
	n = 500000000;//500000000
	c_c = 0.5;
	c_s = 0.5;
	if( argc == 1)
		std::cerr << info << std::endl;//fprintf( stderr, "%s\n", info);
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

	std::cout<<"Épaisseur de la plaque : "<< h <<"\n";
	std::cout<<"Nombre d'échantillons : "<< n <<"\n";
	std::cout<<"C_c : "<< c_c <<"\n";
	std::cout<<"C_s : "<< c_s <<"\n";  
  	
	std::vector<float> absorbed(n,0);
	start = my_gettimeofday();
  	try {
		// Get list of OpenCL platforms.
		std::vector<cl::Platform> platform;
		cl::Platform::get(&platform);

		if (platform.empty()) {
			std::cerr << "OpenCL platforms not found." << std::endl;
			return 1;
		}

		// GPU disponible
		cl::Context context;
		std::vector<cl::Device> device;
		for(auto p = platform.begin(); device.empty() && p != platform.end(); p++) {
			std::vector<cl::Device> pldev;

			try {
			p->getDevices(CL_DEVICE_TYPE_GPU, &pldev);

			for(auto d = pldev.begin(); device.empty() && d != pldev.end(); d++) {
				if (!d->getInfo<CL_DEVICE_AVAILABLE>()) continue;

				std::string ext = d->getInfo<CL_DEVICE_EXTENSIONS>();

				if (
					ext.find("cl_khr_fp64") == std::string::npos &&
					ext.find("cl_amd_fp64") == std::string::npos
				   ) continue;

				device.push_back(*d);
				context = cl::Context(device);
			}
			} catch(...) {
			device.clear();
			}
		}

		if (device.empty()) {
			std::cerr << "GPUs with double precision not found." << std::endl;
			return 1;
		}

		std::cout << device[0].getInfo<CL_DEVICE_NAME>() << std::endl;

		// Create command queue.
		cl::CommandQueue queue(context, device[0]);
		source = oclLoadSource("neutron-opencl.cl");
		// Compile OpenCL program for found device.
		cl::Program program(context, cl::Program::Sources(
				1, std::make_pair(source, strlen(source))
				));

		try {
			program.build(device);
		} catch (const cl::Error&) {
			std::cerr
			<< "OpenCL compilation error" << std::endl
				<< program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device[0])
			<< std::endl;
			return 1;
		}

		cl::Kernel add(program, "neutron_gpu");

	
		//start = my_gettimeofday();
		// Allocation et transfert sur gpu 
		cl::Buffer gpu_r(context, CL_MEM_READ_WRITE,
			sizeof(int),&r);

		cl::Buffer gpu_t(context, CL_MEM_READ_WRITE,
			sizeof(int),&t);
	
		cl::Buffer gpu_b(context, CL_MEM_READ_WRITE,
			sizeof(int),&b);

		cl::Buffer g_absorbed(context, CL_MEM_READ_WRITE,
			absorbed.size() * sizeof(float),absorbed.data());
		
		// Ajout des paramètres du kernel
		add.setArg(0, static_cast<cl_int>(n));
		add.setArg(1, gpu_r);
		add.setArg(2, gpu_t);
		add.setArg(3, gpu_b);
		add.setArg(4, g_absorbed);
		add.setArg(5, c);
		add.setArg(6, c_c);
		add.setArg(7, c_s);
		add.setArg(8, h);
		add.setArg(9, nb_op);


		
		//calcul du nombre de thread necessaires
		int nb_thread;
		if(n%nb_op==0)
		  nb_thread=n/nb_op;
		else
		  nb_thread=(n/nb_op)+1;
		//Lancement du kernel
		queue.enqueueNDRangeKernel(add, cl::NullRange, nb_thread, cl::NullRange);

		
		
		queue.enqueueReadBuffer(gpu_r, CL_TRUE, 0, sizeof(int), &r);
		queue.enqueueReadBuffer(gpu_b, CL_TRUE, 0, sizeof(int), &b);
		queue.enqueueReadBuffer(gpu_t, CL_TRUE, 0, sizeof(int), &t);
		queue.enqueueReadBuffer(g_absorbed, CL_TRUE, 0, b*sizeof(float), absorbed.data());
		
		queue.finish();
		
		

	} catch (const cl::Error &err) {
		std::cerr
			<< "OpenCL error: "
			<< err.what() << "(" << err.err() << ")"
			<< std::endl;
		return 1;
    }
      
  finish = my_gettimeofday();
	printf("\nneutrons refléchis : %g\n", (float) r);
	printf("neutrons absorbés : %g\n", (float) b);
	printf("neutrons transmis : %g\n", (float) t);


	printf("\nPourcentage des neutrons refléchis : %4.2g\n", (float) r / (float) n);
	printf("Pourcentage des neutrons absorbés : %4.2g\n", (float) b / (float) n);
	printf("Pourcentage des neutrons transmis : %4.2g\n", (float) t / (float) n);

	printf("\nTemps total de calcul: %.8g sec\n", finish - start);
	printf("Millions de neutrons /s: %.2g\n", (double) n / ((finish - start)*1e6));

	// ouverture du fichier pour ecrire les positions des neutrons absorbés
	/*
	FILE *f_handle = fopen(OUTPUT_FILE, "w");
	if (!f_handle) {
	fprintf(stderr, "Cannot open " OUTPUT_FILE "\n");
	exit(EXIT_FAILURE);
	}

	for (j = 0; j < b; j++){
	fprintf(f_handle, "%f\n", absorbed[j]);
	}

	fclose(f_handle);
	printf("Result written in " OUTPUT_FILE "\n"); 
	*/    
    
    
}

