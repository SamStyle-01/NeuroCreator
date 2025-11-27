__kernel void sigmoid_deriv_simple_inplace(__global float* A, int size) {
  int gid = get_global_id(0);
  if (gid < size) {
    float v = A[gid];
	A[gid] = v * (1.0f - v);
  }
}
