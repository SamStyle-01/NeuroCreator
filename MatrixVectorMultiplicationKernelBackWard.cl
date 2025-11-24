__kernel void matrixBatchMul(
    __global float* output,
    __global const float* input,
    __global const float* weights_T,
    __global const float* bias,
    int N, int D, int M)
{
    int n = get_global_id(0);
    int m = get_global_id(1);

    if (n < N && m < M) {
        float sum = bias[m];
        for (int k = 0; k < D; ++k)
            sum += input[n * D + k] * weights_T[m * D + k];

        output[n * M + m] = sum;
    }
}
