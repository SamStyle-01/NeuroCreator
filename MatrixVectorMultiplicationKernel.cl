__kernel void matrixBatchMul(
    __global float* output,
    __global const float* input,
    __global const float* weights_T,
    __global const float* bias,
    const int N, const int D, const int M,
    int activation_type)
{
    int n = get_global_id(0);
    int m = get_global_id(1);

    if (n < N && m < M) {
        float sum = bias[m];
        for (int k = 0; k < D; ++k)
            sum += input[n * D + k] * weights_T[m * D + k];

		if (activation_type == 1 && sum < 0) {
			sum = 0;
		}
        else if (activation_type == 2) {
            sum = 1.0f / (1.0f + exp(-sum));
        }
		else if (activation_type == 3) {
            sum = tanh(sum);
        }

        output[n * M + m] = sum;
    }
}
