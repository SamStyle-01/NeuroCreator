__kernel void sigmoid_inplace(__global float* arr, const int size) {
    int gid = get_global_id(0);
    if (gid < size) {
        float v = arr[gid];
        arr[gid] = 1.0f / (1.0f + exp(-v));
    }
}