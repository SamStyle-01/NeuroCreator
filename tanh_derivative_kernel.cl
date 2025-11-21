__kernel void tanh_deriv_inplace(__global float* A, int size) {
  int gid = get_global_id(0);
  if (gid < size) {
    float v = A[gid];
	A[gid] = 1 - v * v;
  }
}
