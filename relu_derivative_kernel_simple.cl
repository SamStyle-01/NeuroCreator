__kernel void relu_deriv_simple_inplace(__global const float* Z, __global float* result, int size) {
  int gid = get_global_id(0);
  if (gid < size) {
    float v = Z[gid];
	result[gid] = v > 0 ? 1 : 0;
  }
}
