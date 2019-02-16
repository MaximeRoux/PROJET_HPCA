/*
#if defined(cl_khr_fp64)
#  pragma OPENCL EXTENSION cl_khr_fp64: enable
#elif defined(cl_amd_fp64)
#  pragma OPENCL EXTENSION cl_amd_fp64: enable
#else
#  error double precision is not supported
#endif
*/

#include "OCL/mwc64x/cl/mwc64x.cl"

__kernel void neutron_gpu(int n, global int *r, global int *t, global int *b, global float *g_absorbed, float c, float c_c, float c_s, float h, int nb_op){
	float L,d,u,x;
	int j,old;
	unsigned int seed;
	//Generation nombre aléatoire
	//TODO

	j = get_global_id(0)*nb_op;
	seed = j;
	mwc64x_state_t rng;
	ulong period = (1 << 10); 
	MWC64X_SeedStreams(&rng, 0, period);
	int i;
	for(i=0;i<nb_op;i++){
		if(j<n){ 
			d = 0.0;
			x = 0.0;
			while (1) {
				while ((u = MWC64X_NextUint(&rng)/(float)UINT_MAX) == 1.0);
				L = -(1 / c) * log(u);
				x = x + L * cos(d);

				if (x < 0) {
		       		atom_add(r, 1);
		   	  		break;
				} else if (x >= h) {
					atom_add(t, 1);
					break;
			 	} 
            	else if ((u = MWC64X_NextUint(&rng)/(float)UINT_MAX) < c_c / c) {
					old = atom_add(b, 1);
					g_absorbed[old] = x;
					break;
    			} else {
			   		while ((u = MWC64X_NextUint(&rng)/(float)UINT_MAX) == 1.0);
		   			d = u * M_PI;
            	}
        	}
		}
    }
  
}

