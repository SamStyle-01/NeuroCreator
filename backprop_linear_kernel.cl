__kernel void backprop_linear(
    __global const float* next_delta,
    __global const float* weights,
    __global float* delta_out,
	int batch, int N_next, int N_curr) {
	
		int n = get_global_id(0);
		int m = get_global_id(1);
		
		if (gid >= N_curr || bid >= batch) return;

		float sum = 0.0f;
		for (int j = 0; j < N_next; j++) {
			sum += next_delta[bid*N_next + j] * weights[j*N_curr + gid];
		}

		delta_out[bid * N_curr + gid] = sum;
    }
}
