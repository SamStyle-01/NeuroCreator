__kernel void softmax_deriv_inplace(__global float* restrict predicted, __global float* restrict true_vals, __global float* restrict grad, int size) {
  int gid = get_global_id(0);
  
  if (gid < size) {
    grad[gid] = predicted[gid] - true_vals[gid];
  }
}