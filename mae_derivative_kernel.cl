__kernel void mae_deriv_inplace(__global const float* restrict predicted, __global const float* restrict true_vals, __global float* restrict grad, int size) {
  int gid = get_global_id(0);
  
  if (gid < size) {
	float diff = predicted[gid] - true_vals[gid];
    grad[gid] = (diff > 0) ? 1.0f : (diff < 0 ? -1.0f : 0.0f);
  }
}