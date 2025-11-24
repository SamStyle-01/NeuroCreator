__kernel void adam_update_bias(
    __global float* bias,
    __global float* grads,
    __global float* m_b,
    __global float* v_b,
    const float lr,
    const float beta1,
    const float beta2,
    const int t
) {
    int gid = get_global_id(0);
    float g = grads[gid];
    float m_new = beta1 * m_b[gid] + (1.0f - beta1) * g;
    float v_new = beta2 * v_b[gid] + (1.0f - beta2) * g * g;
    float bias_correction1 = 1.0f - pow(beta1, (float)t);
    float bias_correction2 = 1.0f - pow(beta2, (float)t);
    float m_hat = m_new / bias_correction1;
    float v_hat = v_new / bias_correction2;
    float update = lr * m_hat / (sqrt(v_hat) + 1e-8);
    bias[gid] = bias[gid] - update;
    m_b[gid] = m_new;
    v_b[gid] = v_new;
}
