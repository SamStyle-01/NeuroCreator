__kernel void tanh_inplace(__global float* arr, const int size) {
    int gid = get_global_id(0);
    if (gid < size) {
        arr[gid] = tanh(arr[gid]);
    }
}