__kernel void adam_update_weights(
    __global float* weights,
    __global float* grads,
    __global float* m,
    __global float* v,
    const float lr,
    const float beta1,
    const float beta2,
    const int t) 
	{
    int gid = get_global_id(0);
    // one element per gid
    float g = grads[gid];
    // update biased first moment estimate
    float m_new = beta1 * m[gid] + (1.0f - beta1) * g;
    // update biased second raw moment estimate
    float v_new = beta2 * v[gid] + (1.0f - beta2) * g * g;
    // bias-corrected estimates
    float bias_correction1 = 1.0f - pow(beta1, (float)t);
    float bias_correction2 = 1.0f - pow(beta2, (float)t);
    // avoid division by zero
    float m_hat = m_new / bias_correction1;
    float v_hat = v_new / bias_correction2;
    // update parameter
    float update = lr * m_hat / (sqrt(v_hat) + 1e-8);
    weights[gid] = weights[gid] - update;
    // write back moments
    m[gid] = m_new;
    v[gid] = v_new;
}
