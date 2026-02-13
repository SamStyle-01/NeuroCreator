__kernel void vector_mult_inplace(__global float* A, __global const float* B, const int size) {
  int gid = get_global_id(0);
  if (gid < size) {
	A[gid] *= B[gid];
  }
}
