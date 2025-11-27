__kernel void relu_inplace(__global float* arr, int size) {
    int gid = get_global_id(0);
    if (gid < size) {
        float v = arr[gid];
        arr[gid] = (v > 0.0f) ? v : 0.0f;
    }
}