__kernel void relu_deriv_inplace(__global const float* Z, __global float* delta, int size) {
  int gid = get_global_id(0);
  if (gid < size) {
    float v = Z[gid];
	delta[gid] *= v > 0 ? 1 : 0;
  }
}
