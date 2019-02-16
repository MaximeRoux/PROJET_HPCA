all:
	make -C cuda
	make -C omp
	make -C hybrid
	make -C OCL_GPU
	make -C OCL_CPU
	make -C MPI_CUDA
	make -C MPI_hybrid
	make -C seq


VERSION ?= cuda

exec : 
	make -C ${VERSION} exec

clean :
	make -C cuda clean
	make -C omp clean
	make -C hybrid clean
	make -C OCL_GPU clean
	make -C OCL_CPU clean
	make -C MPI_CUDA clean
	make -C MPI_hybrid clean
	make -C seq clean
	rm -f neutron-cuda neutron-omp neutron-hybrid neutron-seq neutron-opencl-gpu neutron-mpi-cuda neutron-mpi_hybride 

