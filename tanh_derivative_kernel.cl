__kernel void tanh_deriv_inplace(__global const float* A, __global float* delta, int size) {
  int gid = get_global_id(0);
  if (gid < size) {
    float v = A[gid];
	delta[gid] *= 1 - v * v;
  }
}
