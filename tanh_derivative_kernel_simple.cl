__kernel void tanh_deriv_simple_inplace(__global const float* A, __global float* result, int size) {
  int gid = get_global_id(0);
  if (gid < size) {
    float v = A[gid];
	result[gid] = 1 - v * v;
  }
}
