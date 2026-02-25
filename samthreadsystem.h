#ifndef SAMTHREADSYSTEM_H
#define SAMTHREADSYSTEM_H

#include "pch.h"

class SamThreadSystem : public QObject {
    Q_OBJECT
public:
    SamThreadSystem();
    void set_device(cl_device_id curr_device, cl_context& context, QVector<cl_kernel>& kernels);

    cl_kernel kernel_matrix_mult;

    // Ядро backprop_linear
    cl_kernel kernel_backprop_linear;

    // Ядро vectors_mult
    cl_kernel kernel_vectors_mult;

    // Ядро weights_first_step
    cl_kernel kernel_weights_first_step;

    // Ядро bias_first_step
    cl_kernel kernel_bias_first_step;

    // Ядро weights_last_step
    cl_kernel kernel_weights_last_step;

    // Ядро bias_last_step
    cl_kernel kernel_bias_last_step;

    // ReLU
    cl_kernel kernel_relu;

    // Sigmoid
    cl_kernel kernel_sigmoid;

    // Tanh
    cl_kernel kernel_tanh;

    // Производная ReLU
    cl_kernel kernel_relu_deriv;

    // Производная Sigmoid
    cl_kernel kernel_sigmoid_deriv;

    // Производная Tanh
    cl_kernel kernel_tanh_deriv;

    // Производная MSE
    cl_kernel kernel_mse_deriv;

    // Производная MAE
    cl_kernel kernel_mae_deriv;

    // Производная Softmax
    cl_kernel kernel_softmax_deriv;

    // Производная BCE
    cl_kernel kernel_bce_deriv;

signals:
    void finished(const bool &success, QString log);
};

#endif // SAMTHREADSYSTEM_H
