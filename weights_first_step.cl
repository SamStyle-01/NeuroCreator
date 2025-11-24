__kernel void compute_dW(
    __global const float* prev_activ,
    __global const float* delta,
    __global float* dW,
    int batch,
    int N_prev,
    int N_curr)
{
    int i = get_global_id(0); // curr neuron
    int j = get_global_id(1); // prev neuron

    if (i >= N_curr || j >= N_prev) return;

    float sum = 0.0f;
    for (int b = 0; b < batch; b++)
        sum += prev_activ[b * N_prev + j] * delta[b*N_curr + i];

    dW[i*N_prev + j] = sum / batch;
}
