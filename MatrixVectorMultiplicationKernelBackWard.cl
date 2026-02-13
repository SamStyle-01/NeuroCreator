__kernel void matrixBatchMulBackward(
    __global float* output,
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
    const int N,
    const int D,
    const int M)
{
    int n = get_global_id(0);
    int m = get_global_id(1);

    if (n < N && m < M) {
        float sum = bias[m];

        for (int k = 0; k < D; ++k) {
            sum += input[n * D + k] * weights[k * M + m];
        }

        output[n * M + m] = sum;
    }
}