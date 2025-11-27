#include "backprop.h"
#include "samsystem.h"
#include "dataframe.h"
#include "samtraining.h"
#include "samtest.h"

BackPropagation::BackPropagation(SamSystem *system, QObject *parent) : QObject(parent) {
    this->system = system;
}

void BackPropagation::doWork(cl_context& context) {
    auto temp_layers = system->model->get_layers();
    float eta = this->system->training_view->get_learning_rate();

    // Обработка данных
    cl_int err;

    // Создание командной очереди
    cl_command_queue queue = clCreateCommandQueue(context, system->curr_device, 0, &err);
    OCL_SAFE_CALL(err);

    int final_layer_size = temp_layers.back()->num_neuros;

    auto temp_funcs = system->model->get_funcs();
    QVector<Activation> activations_layers(temp_layers.size(), Activation::LINEAR);
    for (int i = 0; i < temp_funcs.size(); i++) {
        if (temp_funcs[i]->func == "ReLU") {
            activations_layers[temp_funcs[i]->num_layer] = Activation::RELU;
        }
        else if (temp_funcs[i]->func == "SoftMax") {
            activations_layers[temp_funcs[i]->num_layer] = Activation::SOFTMAX;
        }
        else if (temp_funcs[i]->func == "Sigmoid") {
            activations_layers[temp_funcs[i]->num_layer] = Activation::SIGMOID;
        }
        else if (temp_funcs[i]->func == "Tanh") {
            activations_layers[temp_funcs[i]->num_layer] = Activation::TANH;
        }
    }

    int train_share = this->system->training_view->get_train_share();
    int train_cols = system->data->get_cols() - temp_layers.back()->num_neuros;
    auto sharded_data = this->system->data->train_test_split(train_share);
    sharded_data.first->random_shuffle();
    auto& data = sharded_data.first->get_data();

    QVector<cl_mem> bias;
    QVector<cl_mem> m_b;
    QVector<cl_mem> v_b;

    QVector<cl_mem> weights;
    QVector<cl_mem> m_w;
    QVector<cl_mem> v_w;

    for (int l = 1; l < temp_layers.size(); l++) {
        int N_l = temp_layers[l]->num_neuros;
        int N_prev = temp_layers[l - 1]->num_neuros;

        size_t size_bias = N_l * sizeof(float);
        bias.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_bias, system->model->bias[l - 1], &err));
        m_b.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_bias, system->m_b[l - 1].data(), &err));
        v_b.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_bias, system->v_b[l - 1].data(), &err));

        size_t size_weights = N_l * N_prev * sizeof(float);
        weights.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_weights, system->model->weights[l - 1], &err));
        m_w.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_weights, system->m_w[l - 1].data(), &err));
        v_w.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_weights, system->v_w[l - 1].data(), &err));
    }

    int common_size_batch = this->system->training_view->get_batch_size();
    while (this->system->training_view->get_epochs() > 0 && this->system->get_is_training()) {
        sharded_data.first->random_shuffle();
        for (int i = 0; i < sharded_data.first->get_rows(); i += common_size_batch) {
            const int size_batch = std::min(common_size_batch, sharded_data.first->get_rows() - i);

            QVector<float> input_vector(size_batch * temp_layers[0]->num_neuros);;

            for (int j = 0; j < size_batch; j++)
                for (int k = 0; k < train_cols; k++)
                    input_vector[j * train_cols + k] = data[k][i + j];

            QVector<cl_mem> pre_activations;
            QVector<cl_mem> activations;
            pre_activations.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                     input_vector.size() * sizeof(float), input_vector.data(), &err));
            activations.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                 input_vector.size() * sizeof(float), input_vector.data(), &err));
            for (int c = 0; c < temp_layers.size() - 1; c++) {

                // Создание буферов (память на устройстве)
                size_t size_R = size_batch * temp_layers[c + 1]->num_neuros * sizeof(float);

                cl_mem cl_result_vector = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_R, nullptr, &err);
                OCL_SAFE_CALL(err);

                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 0, sizeof(cl_mem), &cl_result_vector));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 1, sizeof(cl_mem), &activations.back()));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 2, sizeof(cl_mem), &weights[c]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 3, sizeof(cl_mem), &bias[c]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 4, sizeof(cl_int), &size_batch));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 5, sizeof(cl_int), &temp_layers[c]->num_neuros));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 6, sizeof(cl_int), &temp_layers[c + 1]->num_neuros));

                // Запуск ядра
                size_t global_work_size[] = { (size_t)size_batch, (size_t)temp_layers[c + 1]->num_neuros };

                err = clEnqueueNDRangeKernel(queue, system->kernel_matrix_mult, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
                clFinish(queue);

                pre_activations.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE, size_R, nullptr, &err));
                err = clEnqueueCopyBuffer(queue, cl_result_vector, pre_activations.back(),
                                          0, 0, size_R,
                                          0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                if (activations_layers[c] == Activation::RELU) {
                    int size = size_batch * temp_layers[c + 1]->num_neuros;
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_relu, 0, sizeof(cl_mem), &cl_result_vector));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_relu, 1, sizeof(cl_int), &size));

                    // Запуск ядра
                    size_t global_work_size[] = { (size_t)size };

                    err = clEnqueueNDRangeKernel(queue, system->kernel_relu, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);
                }
                else if (activations_layers[c] == Activation::SIGMOID) {
                    int size = size_batch * temp_layers[c + 1]->num_neuros;
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_sigmoid, 0, sizeof(cl_mem), &cl_result_vector));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_sigmoid, 1, sizeof(cl_int), &size));

                    // Запуск ядра
                    size_t global_work_size[] = { (size_t)size };

                    err = clEnqueueNDRangeKernel(queue, system->kernel_sigmoid, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);
                }
                else if (activations_layers[c] == Activation::TANH) {
                    int size = size_batch * temp_layers[c + 1]->num_neuros;
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_tanh, 0, sizeof(cl_mem), &cl_result_vector));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_tanh, 1, sizeof(cl_int), &size));

                    // Запуск ядра
                    size_t global_work_size[] = { (size_t)size };

                    err = clEnqueueNDRangeKernel(queue, system->kernel_tanh, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);
                }
                clFinish(queue);

                activations.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE, size_R, nullptr, &err));
                err = clEnqueueCopyBuffer(queue, cl_result_vector, activations.back(),
                                          0, 0, size_R,
                                          0, nullptr, nullptr);

                // Очистка ресурсов
                clReleaseMemObject(cl_result_vector);
            }

            // Обратное распространение ошибки
            QVector<float> true_vals;
            true_vals.reserve(final_layer_size * size_batch);
            for (int j = 0; j < size_batch; j++) {
                for (int k = train_cols; k < data.size(); k++) {
                    true_vals.push_back(data[k][i + j]);
                }
            }

            auto loss_func = this->system->training_view->get_loss_func();
            int size_A_int = final_layer_size;
            size_t size_A = size_A_int * sizeof(float);
            cl_mem cl_delta_vector = clCreateBuffer(context, CL_MEM_READ_WRITE, size_A, nullptr, &err);
            OCL_SAFE_CALL(err);
            switch (loss_func) {
                case LossFunc::MSE: {
                    cl_mem cl_matrix_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_A, true_vals.data(), &err);
                    OCL_SAFE_CALL(err);
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_mse_deriv, 0, sizeof(cl_mem), &activations.back()));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_mse_deriv, 1, sizeof(cl_mem), &cl_matrix_B));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_mse_deriv, 2, sizeof(cl_mem), &cl_delta_vector));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_mse_deriv, 3, sizeof(cl_int), &size_A_int));

                    size_t global_work_size[] = { (size_t)size_A_int };

                    err = clEnqueueNDRangeKernel(queue, system->kernel_mse_deriv, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);

                    clReleaseMemObject(cl_matrix_B);
                    break;
                }
                case LossFunc::MAE: {
                    cl_mem cl_matrix_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_A, true_vals.data(), &err);
                    OCL_SAFE_CALL(err);
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_mae_deriv, 0, sizeof(cl_mem), &activations.back()));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_mae_deriv, 1, sizeof(cl_mem), &cl_matrix_B));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_mae_deriv, 2, sizeof(cl_mem), &cl_delta_vector));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_mae_deriv, 3, sizeof(cl_int), &size_A_int));

                    size_t global_work_size[] = { (size_t)size_A_int };

                    err = clEnqueueNDRangeKernel(queue, system->kernel_mae_deriv, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);

                    clReleaseMemObject(cl_matrix_B);
                    break;
                }
                default: {
                    break;
                }
            }

            if (activations_layers.back() == Activation::RELU) {
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_relu_deriv, 0, sizeof(cl_mem), &pre_activations[pre_activations.size() - 1]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_relu_deriv, 1, sizeof(cl_mem), &cl_delta_vector));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_relu_deriv, 2, sizeof(cl_int), &size_A_int));

                size_t global_work_size[] = { (size_t)size_A_int };

                err = clEnqueueNDRangeKernel(queue, system->kernel_relu_deriv, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }
            else if (activations_layers.back() == Activation::SIGMOID) {
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_sigmoid_deriv, 0, sizeof(cl_mem), &activations[activations.size() - 1]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_sigmoid_deriv, 1, sizeof(cl_mem), &cl_delta_vector));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_sigmoid_deriv, 2, sizeof(cl_int), &size_A_int));

                size_t global_work_size[] = { (size_t)size_A_int };

                err = clEnqueueNDRangeKernel(queue, system->kernel_sigmoid_deriv, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }
            else if (activations_layers.back() == Activation::TANH) {
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_tanh_deriv, 0, sizeof(cl_mem), &activations[activations.size() - 1]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_tanh_deriv, 1, sizeof(cl_mem), &cl_delta_vector));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_tanh_deriv, 2, sizeof(cl_int), &size_A_int));

                size_t global_work_size[] = { (size_t)size_A_int };

                err = clEnqueueNDRangeKernel(queue, system->kernel_tanh_deriv, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }
            else if (activations_layers.back() == Activation::SOFTMAX) {
                cl_mem cl_matrix_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_A, activations.back(), &err);
                OCL_SAFE_CALL(err);
                cl_mem cl_matrix_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_A, true_vals.data(), &err);
                OCL_SAFE_CALL(err);
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_softmax_deriv, 0, sizeof(cl_mem), &cl_matrix_A));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_softmax_deriv, 1, sizeof(cl_mem), &cl_matrix_B));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_softmax_deriv, 2, sizeof(cl_mem), &cl_delta_vector));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_softmax_deriv, 3, sizeof(cl_int), &size_A_int));

                size_t global_work_size[] = { (size_t)size_A_int };

                err = clEnqueueNDRangeKernel(queue, system->kernel_softmax_deriv, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                clReleaseMemObject(cl_matrix_A);
                clReleaseMemObject(cl_matrix_B);
            }

            for (int l = 0; l < temp_layers.size() - 1; l++) {
                if (activations_layers[l] == Activation::RELU) {
                    int size = size_batch * temp_layers[l + 1]->num_neuros;
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_relu_deriv_simple, 0, sizeof(cl_mem), &pre_activations[l]));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_relu_deriv_simple, 1, sizeof(cl_int), &size));

                    size_t global_work_size[] = { (size_t)size_batch * (size_t)temp_layers[l + 1]->num_neuros };

                    err = clEnqueueNDRangeKernel(queue, system->kernel_relu_deriv_simple, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);
                }
            }

            for (int l = 0; l < temp_layers.size() - 1; l++) {
                if (activations_layers[l] == Activation::SIGMOID) {
                    int size = size_batch * temp_layers[l + 1]->num_neuros;
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_sigmoid_deriv_simple, 0, sizeof(cl_mem), &activations[l]));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_sigmoid_deriv_simple, 1, sizeof(cl_int), &size));

                    size_t global_work_size[] = { (size_t)size_batch * (size_t)temp_layers[l + 1]->num_neuros };

                    err = clEnqueueNDRangeKernel(queue, system->kernel_sigmoid_deriv_simple, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);
                }
                else if (activations_layers[l] == Activation::TANH) {
                    int size = size_batch * temp_layers[l + 1]->num_neuros;
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_tanh_deriv_simple, 0, sizeof(cl_mem), &activations[l]));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_tanh_deriv_simple, 1, sizeof(cl_int), &size));

                    size_t global_work_size[] = { (size_t)size_batch * (size_t)temp_layers[l + 1]->num_neuros };

                    err = clEnqueueNDRangeKernel(queue, system->kernel_tanh_deriv_simple, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);
                }
            }

            QVector<cl_mem> hidden_delta(temp_layers.size());
            hidden_delta[temp_layers.size() - 1] = cl_delta_vector;

            for (int l = temp_layers.size() - 2; l >= 0; l--) {
                hidden_delta[l] = clCreateBuffer(context, CL_MEM_READ_WRITE, size_batch * temp_layers[l]->num_neuros * sizeof(float), nullptr, &err);

                OCL_SAFE_CALL(clSetKernelArg(system->kernel_backprop_linear, 0, sizeof(cl_mem), &hidden_delta[l + 1]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_backprop_linear, 1, sizeof(cl_mem), &weights[l]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_backprop_linear, 2, sizeof(cl_mem), &hidden_delta[l]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_backprop_linear, 3, sizeof(cl_int), &size_batch));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_backprop_linear, 4, sizeof(cl_int), &temp_layers[l + 1]->num_neuros));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_backprop_linear, 5, sizeof(cl_int), &temp_layers[l]->num_neuros));

                size_t global_work_size[] = { (size_t)size_batch, (size_t)temp_layers[l]->num_neuros };

                err = clEnqueueNDRangeKernel(queue, system->kernel_backprop_linear, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                if (activations_layers[l] == Activation::RELU) {
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_vectors_mult, 0, sizeof(cl_mem), &hidden_delta[l]));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_vectors_mult, 1, sizeof(cl_mem), &pre_activations[l]));
                    const int size_final = size_batch * temp_layers[l]->num_neuros;
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_vectors_mult, 2, sizeof(cl_int), &size_final));

                    size_t global_work_size[] = { (size_t)size_final };

                    err = clEnqueueNDRangeKernel(queue, system->kernel_vectors_mult, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);
                }
                else if (activations_layers[l] == Activation::SIGMOID || activations_layers[l] == Activation::TANH) {
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_vectors_mult, 0, sizeof(cl_mem), &hidden_delta[l]));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_vectors_mult, 1, sizeof(cl_mem), &activations[l]));
                    const int size_final = size_batch * temp_layers[l]->num_neuros;
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_vectors_mult, 2, sizeof(cl_int), &size_final));

                    size_t global_work_size[] = { (size_t)size_final };

                    err = clEnqueueNDRangeKernel(queue, system->kernel_vectors_mult, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);
                }
            }

            QVector<cl_mem> db;
            QVector<cl_mem> dW;

            for (int l = 1; l < temp_layers.size(); l++) {
                const int N_l = temp_layers[l]->num_neuros;
                const int N_prev = temp_layers[l - 1]->num_neuros;

                // Смещения
                db[l] = clCreateBuffer(context, CL_MEM_READ_WRITE, N_l * sizeof(float), nullptr, &err);

                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_first_step, 0, sizeof(cl_mem), &hidden_delta[l]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_first_step, 1, sizeof(cl_mem), &db[l]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_first_step, 2, sizeof(cl_int), &size_batch));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_first_step, 3, sizeof(cl_int), &N_l));

                size_t global_work_size_bias[] = { (size_t)N_l };

                err = clEnqueueNDRangeKernel(queue, system->kernel_bias_first_step, 1, nullptr, global_work_size_bias, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                // Веса
                dW[l] = clCreateBuffer(context, CL_MEM_READ_WRITE, N_l * N_prev * sizeof(float), nullptr, &err);

                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_first_step, 0, sizeof(cl_mem), &activations[l - 1]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_first_step, 1, sizeof(cl_mem), &hidden_delta[l]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_first_step, 2, sizeof(cl_mem), &dW[l]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_first_step, 3, sizeof(cl_int), &size_batch));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_first_step, 4, sizeof(cl_int), &N_prev));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_first_step, 5, sizeof(cl_int), &N_l));

                size_t global_work_size[] = { (size_t)N_l, (size_t)N_prev };

                err = clEnqueueNDRangeKernel(queue, system->kernel_weights_first_step, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }

            this->system->t++;

            for (int l = 1; l < temp_layers.size(); l++) {
                int N_l = temp_layers[l]->num_neuros;
                int N_prev = temp_layers[l - 1]->num_neuros;
                // Смещения
                size_t size_bias = N_l * sizeof(float);

                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_last_step, 0, sizeof(cl_mem), &bias[l - 1]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_last_step, 1, sizeof(cl_mem), &db[l]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_last_step, 2, sizeof(cl_mem), &m_b[l - 1]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_last_step, 3, sizeof(cl_mem), &v_b[l - 1]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_last_step, 4, sizeof(cl_float), &eta));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_last_step, 5, sizeof(cl_float), &this->system->beta1));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_last_step, 6, sizeof(cl_float), &this->system->beta2));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_last_step, 7, sizeof(cl_int), &this->system->t));

                size_t global_work_size_bias[] = { (size_t)N_l };

                err = clEnqueueNDRangeKernel(queue, system->kernel_bias_last_step, 1, nullptr, global_work_size_bias, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                clFinish(queue);
                err = clEnqueueReadBuffer(queue, bias[l - 1], CL_TRUE, 0, size_bias, this->system->model->bias[l - 1], 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                err = clEnqueueReadBuffer(queue, m_b[l - 1], CL_TRUE, 0, size_bias, this->system->m_b[l - 1].data(), 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                err = clEnqueueReadBuffer(queue, v_b[l - 1], CL_TRUE, 0, size_bias, this->system->v_b[l - 1].data(), 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                // Веса
                size_t size_weights = N_l * N_prev * sizeof(float);

                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_last_step, 0, sizeof(cl_mem), &weights[l - 1]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_last_step, 1, sizeof(cl_mem), &dW[l]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_last_step, 2, sizeof(cl_mem), &m_w[l - 1]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_last_step, 3, sizeof(cl_mem), &v_w[l - 1]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_last_step, 4, sizeof(cl_float), &eta));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_last_step, 5, sizeof(cl_float), &this->system->beta1));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_last_step, 6, sizeof(cl_float), &this->system->beta2));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_last_step, 7, sizeof(cl_int), &this->system->t));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_last_step, 8, sizeof(cl_int), &N_prev));

                size_t global_work_size[] = { (size_t)N_l, (size_t)N_prev };

                err = clEnqueueNDRangeKernel(queue, system->kernel_weights_last_step, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                clFinish(queue);
                err = clEnqueueReadBuffer(queue, weights[l - 1], CL_TRUE, 0, size_weights, this->system->model->weights[l - 1], 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                err = clEnqueueReadBuffer(queue, m_w[l - 1], CL_TRUE, 0, size_weights, this->system->m_w[l - 1].data(), 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                err = clEnqueueReadBuffer(queue, v_w[l - 1], CL_TRUE, 0, size_weights, this->system->v_w[l - 1].data(), 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }

            for (int ind = 0; ind < db.size(); ind++) {
                clReleaseMemObject(db[ind]);
                clReleaseMemObject(dW[ind]);
            }

            for (int ind = 0; ind < hidden_delta.size(); ind++) {
                clReleaseMemObject(hidden_delta[ind]);
            }

            for (int activs = 0; activs < activations.size(); activs++) {
                clReleaseMemObject(activations[activs]);
                clReleaseMemObject(pre_activations[activs]);
            }
        }
        this->system->model->update_weights();

        SamTest* test = new SamTest(this->system);
        float train_loss = -1;
        float test_loss = -1;
        auto reply = test->doWork(sharded_data.first, context);
        if (reply.first == "")
            train_loss = reply.second;
        else {
            emit finished(false, reply.first);
        }

        reply = test->doWork(sharded_data.second, context);
        if (reply.first == "")
            test_loss = reply.second;
        else {
            emit finished(false, reply.first);
        }
        delete test;

        if (train_share != 100) {
            if (system->best_loss > test_loss) {
                system->best_loss = test_loss;
                this->system->steal_weights_bias(system->model->weights, system->model->bias);
                this->system->best_epoch = this->system->curr_epochs;
            }
        }
        else {
            if (system->best_loss > train_loss) {
                system->best_loss = train_loss;
                qDebug() << this->system->best_epoch << " " << train_loss;
                this->system->steal_weights_bias(system->model->weights, system->model->bias);
                this->system->best_epoch = this->system->curr_epochs;
            }
        }
        emit epoch_done(train_loss, test_loss);

        this->system->training_view->set_epochs(this->system->training_view->get_epochs() - 1);
    }

    for (int l = 0; l < bias.size(); l++) {
        clReleaseMemObject(bias[l]);
        clReleaseMemObject(m_b[l]);
        clReleaseMemObject(v_b[l]);

        clReleaseMemObject(weights[l]);
        clReleaseMemObject(m_w[l]);
        clReleaseMemObject(v_w[l]);
    }

    clReleaseCommandQueue(queue);

    delete sharded_data.first;
    delete sharded_data.second;

    this->system->set_is_training(false);

    emit finished(true, "");
}
