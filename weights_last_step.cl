__kernel void adam_update_weights(
    __global float* weights,
    __global const float* grads,
    __global float* m,
    __global float* v,
    const float lr,
    const float beta1,
    const float beta2,
    const float beta1_pow_t,
	const float beta2_pow_t,
    const int N_prev) {
	
    int i = get_global_id(0);
    int j = get_global_id(1);

    int idx = i * N_prev + j;

    float g = grads[idx] + 0.01f * weights[idx];

    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

    float m_hat = m[idx] / (1.0f - beta1_pow_t);
    float v_hat = v[idx] / (1.0f - beta2_pow_t);

    weights[idx] -= lr * m_hat / (sqrt(v_hat) + 1e-8f);
}