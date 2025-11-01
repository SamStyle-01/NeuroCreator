__kernel void matrixBatchMul(
    __global float* output,  // N ? M
    __global float* input,   // N ? D
    __global float* weights, // D ? M
    __global float* bias,    // M
    int N, int D, int M)
{
    int n = get_global_id(0); // индекс строки (образца)
    int m = get_global_id(1); // индекс нейрона выхода

    if (n < N && m < M) {
        float sum = bias[m];
        for (int k = 0; k < D; ++k)
            sum += input[n * D + k] * weights[k * M + m];
        output[n * M + m] = sum;
    }
}
