__kernel void adam_update_bias(
    __global float* bias,
    __global const float* grads,
    __global float* m_b,
    __global float* v_b,
    const float lr,
    const float beta1,
    const float beta2,
    const float beta1_pow_t,
	const float beta2_pow_t) {
    int gid = get_global_id(0);
    float g = grads[gid];
    
    m_b[gid] = beta1 * m_b[gid] + (1.0f - beta1) * g;
    v_b[gid] = beta2 * v_b[gid] + (1.0f - beta2) * g * g;
    
    float m_hat = m_b[gid] / (1.0f - beta1_pow_t);
    float v_hat = v_b[gid] / (1.0f - beta2_pow_t);
    
    bias[gid] -= lr * m_hat / (sqrt(v_hat) + 1e-8f);
}