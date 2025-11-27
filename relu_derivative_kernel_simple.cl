__kernel void relu_deriv_simple_inplace(__global float* Z, int size) {
  int gid = get_global_id(0);
  if (gid < size) {
    float v = Z[gid];
	Z[gid] = v > 0 ? 1 : 0;
  }
}
