__kernel void relu_inplace(__global float* arr, int size) {
    int gid = get_global_id(0);
    if (gid < size) {
        arr[gid] = max(arr[gid], 0.0f);
    }
}