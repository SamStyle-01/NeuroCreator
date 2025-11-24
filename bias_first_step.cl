__kernel void compute_db(
    __global const float* delta,
    __global float* db,
    int batch,
    int N)
{
    int gid = get_global_id(0);
    if (gid >= N) return;

    float sum = 0.0f;
    for (int b = 0; b < batch; b++)
        sum += delta[b*N + gid];

    db[gid] = sum / batch;
}
