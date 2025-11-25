__kernel void adam_update_weights(
    __global float* weights,
    __global const float* grads,
    __global float* m,
    __global float* v,
    const float lr,
    const float beta1,
    const float beta2,
    const int t,
    const int N_prev)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    int idx = i * N_prev + j;

    float g = grads[idx];

    // m и v
    float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
    float v_new = beta2 * v[idx] + (1.0f - beta2) * g * g;

    float bc1 = 1.0f - pow(beta1, (float)t);
    float bc2 = 1.0f - pow(beta2, (float)t);

    float m_hat = m_new / bc1;
    float v_hat = v_new / bc2;

    float update = lr * m_hat / (sqrt(v_hat) + 1e-8f);

    weights[idx] -= update;

    m[idx] = m_new;
    v[idx] = v_new;
}
