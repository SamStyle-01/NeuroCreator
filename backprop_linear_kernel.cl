__kernel void backprop_linear(
    __global const float* next_delta,
    __global const float* weights,
    __global float* delta_out,
	__global const float* activation,
	int batch, int N_next, int N_curr,
	int activation_type) {
	
		int bid = get_global_id(0);
		int gid = get_global_id(1);
		
		if (gid >= N_curr || bid >= batch) return;

		float sum = 0.0f;
		for (int j = 0; j < N_next; j++) {
			sum += next_delta[bid * N_next + j] * weights[gid * N_next + j];
		}

		delta_out[bid * N_curr + gid] = sum;
		
		if (activation_type == 1) {
			float v = activation[bid * N_curr + gid];
			delta_out[bid * N_curr + gid] *= v > 0 ? 1 : 0;
		}
		else if (activation_type == 2) {
			float v = activation[bid * N_curr + gid];
			delta_out[bid * N_curr + gid] *= v * (1.0f - v);
		}
		else if (activation_type == 3) {
			float v = activation[bid * N_curr + gid];
			delta_out[bid * N_curr + gid] *= 1 - v * v;
		}
    }
