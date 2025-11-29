__kernel void backprop_linear(
    __global const float* next_delta,
    __global const float* weights,
    __global float* delta_out,
	int batch, int N_next, int N_curr) {
	
		int bid = get_global_id(0);
		int gid = get_global_id(1);
		
		if (gid >= N_curr || bid >= batch) return;

		float sum = 0.0f;
		for (int j = 0; j < N_next; j++) {
			sum += next_delta[bid * N_next + j] * weights[gid * N_next + j];
		}

		delta_out[bid * N_curr + gid] = sum;
    }
