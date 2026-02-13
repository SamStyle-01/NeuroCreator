__kernel void bce_deriv_inplace(
    __global const float* restrict logits,
    __global const float* restrict true_vals,
    __global float* restrict grad,
    const int size)
{
    int gid = get_global_id(0);

    if (gid < size) {

        float z = logits[gid];

        float sigmoid;

        if (z >= 0.0f) {
            float exp_neg = exp(-z);
            sigmoid = 1.0f / (1.0f + exp_neg);
        } else {
            float exp_pos = exp(z);
            sigmoid = exp_pos / (1.0f + exp_pos);
        }

        grad[gid] = sigmoid - true_vals[gid];
    }
}
