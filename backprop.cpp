#include "backprop.h"
#include "samsystem.h"
#include "dataframe.h"
#include "samtraining.h"
#include "samtest.h"

BackPropagation::BackPropagation(SamSystem *system, QObject *parent) : QObject(parent) {
    this->system = system;
}

constexpr qint64 MIN_EPOCH_TIME_MS = 5;

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
    int train_cols = system->data->get_cols() - final_layer_size;
    auto sharded_data = this->system->data->train_test_split(train_share);
    sharded_data.first->random_shuffle();
    auto& data = sharded_data.first->get_data();

    QVector<SamArray> bias;
    QVector<SamArray> m_b;
    QVector<SamArray> v_b;

    QVector<SamArray> weights;
    QVector<SamArray> m_w;
    QVector<SamArray> v_w;

    for (int l = 1; l < temp_layers.size(); l++) {
        int N_l = temp_layers[l]->num_neuros;
        int N_prev = temp_layers[l - 1]->num_neuros;

        size_t size_bias = N_l * sizeof(float);
        bias.emplace_back(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_bias, system->model->bias[l - 1], &err), size_bias);
        m_b.emplace_back(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_bias, system->m_b[l - 1].data(), &err), size_bias);
        v_b.emplace_back(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_bias, system->v_b[l - 1].data(), &err), size_bias);

        size_t size_weights = N_l * N_prev * sizeof(float);
        weights.emplace_back(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_weights, system->model->weights[l - 1], &err), size_weights);
        m_w.emplace_back(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_weights, system->m_w[l - 1].data(), &err), size_weights);
        v_w.emplace_back(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_weights, system->v_w[l - 1].data(), &err), size_weights);
    }

    auto loss_func = this->system->training_view->get_loss_func();
    int common_size_batch = this->system->training_view->get_batch_size();
    while (this->system->training_view->get_epochs() > 0 && this->system->get_is_training()) {
        QElapsedTimer epochTimer;
        epochTimer.start();
        sharded_data.first->random_shuffle();
        for (int i = 0; i < sharded_data.first->get_rows(); i += common_size_batch) {
            const int size_batch = std::min(common_size_batch, sharded_data.first->get_rows() - i);

            QVector<float> input_vector(size_batch * temp_layers[0]->num_neuros);

            for (int j = 0; j < size_batch; j++)
                for (int k = 0; k < train_cols; k++)
                    input_vector[j * train_cols + k] = data[k][i + j];

            QVector<SamArray> activations;
            SamArray cl_result_vector;
            for (int c = 0; c < temp_layers.size(); c++) {

                // Создание буферов (память на устройстве)
                size_t size_R = size_batch * temp_layers[c]->num_neuros * sizeof(float);

                OCL_SAFE_CALL(err);
                if (!activations.size()) {
                    cl_result_vector = SamArray(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                      input_vector.size() * sizeof(float), input_vector.data(), &err), input_vector.size());
                }

                if (activations_layers[c] == Activation::RELU) {
                    int size = size_batch * temp_layers[c]->num_neuros;
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_relu, 0, sizeof(cl_mem), &cl_result_vector.memory));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_relu, 1, sizeof(cl_int), &size));

                    // Запуск ядра
                    size_t global_work_size[] = { (size_t)size };

                    err = clEnqueueNDRangeKernel(queue, system->kernel_relu, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);
                }
                else if (activations_layers[c] == Activation::SIGMOID && (loss_func != LossFunc::B_CROSSENTROPY && c != temp_layers.size())) {
                    int size = size_batch * temp_layers[c]->num_neuros;
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_sigmoid, 0, sizeof(cl_mem), &cl_result_vector.memory));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_sigmoid, 1, sizeof(cl_int), &size));

                    // Запуск ядра
                    size_t global_work_size[] = { (size_t)size };

                    err = clEnqueueNDRangeKernel(queue, system->kernel_sigmoid, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);
                }
                else if (activations_layers[c] == Activation::TANH) {
                    int size = size_batch * temp_layers[c]->num_neuros;
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_tanh, 0, sizeof(cl_mem), &cl_result_vector.memory));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_tanh, 1, sizeof(cl_int), &size));

                    // Запуск ядра
                    size_t global_work_size[] = { (size_t)size };

                    err = clEnqueueNDRangeKernel(queue, system->kernel_tanh, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);
                }
                else if (activations_layers[c] == Activation::SOFTMAX) {
                    int size = size_batch * final_layer_size;

                    QVector<float> vec(size);
                    err = clEnqueueReadBuffer(queue, cl_result_vector.memory, CL_TRUE, 0, size * sizeof(float), vec.data(), 0, nullptr, nullptr);
                    for (int el = 0; el < vec.size(); el += final_layer_size) {
                        QVector<float>::Iterator it = vec.begin() + el;
                        this->system->SoftMax_func(it, it + final_layer_size);
                    }
                    err = clEnqueueWriteBuffer(queue, cl_result_vector.memory, CL_TRUE, 0, size * sizeof(float), vec.data(), 0, nullptr, nullptr);

                    OCL_SAFE_CALL(err);
                }

                activations.emplace_back(clCreateBuffer(context, CL_MEM_READ_WRITE, size_R, nullptr, &err), size_R / sizeof(float));
                err = clEnqueueCopyBuffer(queue, cl_result_vector.memory, activations.back().memory, 0, 0, size_R, 0, nullptr, nullptr);

                if (c != temp_layers.size() - 1) {
                    cl_result_vector.clear();
                    int size_R2 = size_batch * temp_layers[c + 1]->num_neuros * sizeof(float);
                    cl_result_vector = SamArray(clCreateBuffer(context, CL_MEM_READ_WRITE, size_R2, nullptr, &err), size_R2 / sizeof(float));
                    OCL_SAFE_CALL(err);

                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 0, sizeof(cl_mem), &cl_result_vector.memory));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 1, sizeof(cl_mem), &activations.back().memory));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 2, sizeof(cl_mem), &weights[c].memory));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 3, sizeof(cl_mem), &bias[c].memory));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 4, sizeof(cl_int), &size_batch));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 5, sizeof(cl_int), &temp_layers[c]->num_neuros));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 6, sizeof(cl_int), &temp_layers[c + 1]->num_neuros));

                    // Запуск ядра
                    size_t global_work_size[] = { (size_t)size_batch, (size_t)temp_layers[c + 1]->num_neuros };

                    err = clEnqueueNDRangeKernel(queue, system->kernel_matrix_mult, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);
                }
                // Очистка ресурсов
            }
            cl_result_vector.clear();

            // Обратное распространение ошибки
            QVector<float> true_vals(final_layer_size * size_batch);
            for (int j = 0; j < size_batch; j++)
                for (int k = train_cols; k < data.size(); k++)
                    true_vals[j * final_layer_size + (k - train_cols)] = data[k][i + j];

            size_t size_A = final_layer_size * size_batch * sizeof(float);
            SamArray cl_delta_vector = SamArray(clCreateBuffer(context, CL_MEM_READ_WRITE, size_A, nullptr, &err), size_A / sizeof(float));
            OCL_SAFE_CALL(err);
            switch (loss_func) {
                case LossFunc::MSE: {
                    int size = final_layer_size * size_batch;
                    SamArray cl_matrix_B = SamArray(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_A, true_vals.data(), &err), size_A / sizeof(float));
                    OCL_SAFE_CALL(err);
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_mse_deriv, 0, sizeof(cl_mem), &activations.back().memory));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_mse_deriv, 1, sizeof(cl_mem), &cl_matrix_B.memory));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_mse_deriv, 2, sizeof(cl_mem), &cl_delta_vector.memory));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_mse_deriv, 3, sizeof(cl_int), &size));

                    size_t global_work_size[] = { (size_t)size };

                    err = clEnqueueNDRangeKernel(queue, system->kernel_mse_deriv, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);

                    cl_matrix_B.clear();
                    break;
                }
                case LossFunc::MAE: {
                    int size = final_layer_size * size_batch;
                    SamArray cl_matrix_B = SamArray(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_A, true_vals.data(), &err), size_A / sizeof(float));
                    OCL_SAFE_CALL(err);
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_mae_deriv, 0, sizeof(cl_mem), &activations.back().memory));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_mae_deriv, 1, sizeof(cl_mem), &cl_matrix_B.memory));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_mae_deriv, 2, sizeof(cl_mem), &cl_delta_vector.memory));
                    OCL_SAFE_CALL(clSetKernelArg(system->kernel_mae_deriv, 3, sizeof(cl_int), &size));

                    size_t global_work_size[] = { (size_t)size };

                    err = clEnqueueNDRangeKernel(queue, system->kernel_mae_deriv, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);

                    cl_matrix_B.clear();
                    break;
                }
                default: {
                    break;
                }
            }

            if (activations_layers.back() == Activation::RELU) {
                int size = final_layer_size * size_batch;
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_relu_deriv, 0, sizeof(cl_mem), &activations.back().memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_relu_deriv, 1, sizeof(cl_mem), &cl_delta_vector.memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_relu_deriv, 2, sizeof(cl_int), &size));

                size_t global_work_size[] = { (size_t)size };

                err = clEnqueueNDRangeKernel(queue, system->kernel_relu_deriv, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }
            else if (activations_layers.back() == Activation::SIGMOID && this->system->training_view->get_loss_func() == LossFunc::B_CROSSENTROPY) {
                SamArray cl_matrix_B = SamArray(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_A, true_vals.data(), &err), size_A / sizeof(float));
                OCL_SAFE_CALL(err);
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bce_deriv, 0, sizeof(cl_mem), &activations.back().memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bce_deriv, 1, sizeof(cl_mem), &cl_matrix_B.memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bce_deriv, 2, sizeof(cl_mem), &cl_delta_vector.memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bce_deriv, 3, sizeof(cl_int), &size_batch));

                size_t global_work_size[] = { (size_t)size_batch };

                err = clEnqueueNDRangeKernel(queue, system->kernel_bce_deriv, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                cl_matrix_B.clear();
            }
            else if (activations_layers.back() == Activation::SIGMOID) {
                int size = final_layer_size * size_batch;
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_sigmoid_deriv, 0, sizeof(cl_mem), &activations.back().memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_sigmoid_deriv, 1, sizeof(cl_mem), &cl_delta_vector.memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_sigmoid_deriv, 2, sizeof(cl_int), &size));

                size_t global_work_size[] = { (size_t)size };

                err = clEnqueueNDRangeKernel(queue, system->kernel_sigmoid_deriv, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }
            else if (activations_layers.back() == Activation::TANH) {
                int size = final_layer_size * size_batch;
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_tanh_deriv, 0, sizeof(cl_mem), &activations.back().memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_tanh_deriv, 1, sizeof(cl_mem), &cl_delta_vector.memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_tanh_deriv, 2, sizeof(cl_int), &size));

                size_t global_work_size[] = { (size_t)size };

                err = clEnqueueNDRangeKernel(queue, system->kernel_tanh_deriv, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }
            else if (activations_layers.back() == Activation::SOFTMAX) {
                int size = final_layer_size * size_batch;
                SamArray cl_matrix_B = SamArray(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_A, true_vals.data(), &err), size_A / sizeof(float));
                OCL_SAFE_CALL(err);
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_softmax_deriv, 0, sizeof(cl_mem), &activations.back().memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_softmax_deriv, 1, sizeof(cl_mem), &cl_matrix_B.memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_softmax_deriv, 2, sizeof(cl_mem), &cl_delta_vector.memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_softmax_deriv, 3, sizeof(cl_int), &size));

                size_t global_work_size[] = { (size_t)size };

                err = clEnqueueNDRangeKernel(queue, system->kernel_softmax_deriv, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                cl_matrix_B.clear();
            }

            QVector<SamArray> hidden_delta(temp_layers.size() - 1);
            hidden_delta[hidden_delta.size() - 1] = std::move(cl_delta_vector);

            for (int l = temp_layers.size() - 2; l >= 1; l--) {

                int activation_type = 0;
                if (activations_layers[l] == Activation::RELU) {
                    activation_type = 1;
                }
                else if (activations_layers[l] == Activation::SIGMOID) {
                    activation_type = 2;
                }
                else if (activations_layers[l] == Activation::TANH) {
                    activation_type = 3;
                }

                hidden_delta[l - 1] = SamArray(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                            size_batch * temp_layers[l]->num_neuros * sizeof(float), nullptr, &err), size_batch * temp_layers[l]->num_neuros);

                OCL_SAFE_CALL(clSetKernelArg(system->kernel_backprop_linear, 0, sizeof(cl_mem), &hidden_delta[l].memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_backprop_linear, 1, sizeof(cl_mem), &weights[l].memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_backprop_linear, 2, sizeof(cl_mem), &hidden_delta[l - 1].memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_backprop_linear, 3, sizeof(cl_mem), &activations[l].memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_backprop_linear, 4, sizeof(cl_int), &size_batch));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_backprop_linear, 5, sizeof(cl_int), &temp_layers[l + 1]->num_neuros));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_backprop_linear, 6, sizeof(cl_int), &temp_layers[l]->num_neuros));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_backprop_linear, 7, sizeof(cl_int), &activation_type));

                size_t global_work_size[] = { (size_t)size_batch, (size_t)temp_layers[l]->num_neuros };

                err = clEnqueueNDRangeKernel(queue, system->kernel_backprop_linear, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }

            QVector<SamArray> db(temp_layers.size() - 1);
            QVector<SamArray> dW(temp_layers.size() - 1);

            for (int l = 1; l < temp_layers.size(); l++) {
                const int N_l = temp_layers[l]->num_neuros;
                const int N_prev = temp_layers[l - 1]->num_neuros;

                // Смещения
                db[l - 1] = SamArray(clCreateBuffer(context, CL_MEM_READ_WRITE, N_l * sizeof(float), nullptr, &err), N_l);
                OCL_SAFE_CALL(err);

                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_first_step, 0, sizeof(cl_mem), &hidden_delta[l - 1].memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_first_step, 1, sizeof(cl_mem), &db[l - 1].memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_first_step, 2, sizeof(cl_int), &size_batch));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_first_step, 3, sizeof(cl_int), &N_l));

                size_t global_work_size_bias[] = { (size_t)N_l };

                err = clEnqueueNDRangeKernel(queue, system->kernel_bias_first_step, 1, nullptr, global_work_size_bias, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                // Веса
                dW[l - 1] = SamArray(clCreateBuffer(context, CL_MEM_READ_WRITE, N_l * N_prev * sizeof(float), nullptr, &err), N_l * N_prev);
                OCL_SAFE_CALL(err);

                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_first_step, 0, sizeof(cl_mem), &activations[l - 1].memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_first_step, 1, sizeof(cl_mem), &hidden_delta[l - 1].memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_first_step, 2, sizeof(cl_mem), &dW[l - 1].memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_first_step, 3, sizeof(cl_int), &size_batch));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_first_step, 4, sizeof(cl_int), &N_prev));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_first_step, 5, sizeof(cl_int), &N_l));

                size_t global_work_size[] = { (size_t)N_l, (size_t)N_prev };

                err = clEnqueueNDRangeKernel(queue, system->kernel_weights_first_step, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }

            this->system->t++;
            this->system->t = std::min(this->system->t, 1000000);

            for (int l = 1; l < temp_layers.size(); l++) {
                int N_l = temp_layers[l]->num_neuros;
                int N_prev = temp_layers[l - 1]->num_neuros;

                float pow_beta1_t = pow(this->system->beta1, this->system->t);
                float pow_beta2_t = pow(this->system->beta2, this->system->t);

                // Смещения
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_last_step, 0, sizeof(cl_mem), &bias[l - 1].memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_last_step, 1, sizeof(cl_mem), &db[l - 1].memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_last_step, 2, sizeof(cl_mem), &m_b[l - 1].memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_last_step, 3, sizeof(cl_mem), &v_b[l - 1].memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_last_step, 4, sizeof(cl_float), &eta));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_last_step, 5, sizeof(cl_float), &this->system->beta1));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_last_step, 6, sizeof(cl_float), &this->system->beta2));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_last_step, 7, sizeof(cl_float), &pow_beta1_t));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_bias_last_step, 8, sizeof(cl_float), &pow_beta2_t));

                size_t global_work_size_bias[] = { (size_t)N_l };

                err = clEnqueueNDRangeKernel(queue, system->kernel_bias_last_step, 1, nullptr, global_work_size_bias, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                // Веса
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_last_step, 0, sizeof(cl_mem), &weights[l - 1].memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_last_step, 1, sizeof(cl_mem), &dW[l - 1].memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_last_step, 2, sizeof(cl_mem), &m_w[l - 1].memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_last_step, 3, sizeof(cl_mem), &v_w[l - 1].memory));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_last_step, 4, sizeof(cl_float), &eta));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_last_step, 5, sizeof(cl_float), &this->system->beta1));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_last_step, 6, sizeof(cl_float), &this->system->beta2));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_last_step, 7, sizeof(cl_float), &pow_beta1_t));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_last_step, 8, sizeof(cl_float), &pow_beta2_t));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_weights_last_step, 9, sizeof(cl_int), &N_prev));

                size_t global_work_size[] = { (size_t)N_l, (size_t)N_prev };

                err = clEnqueueNDRangeKernel(queue, system->kernel_weights_last_step, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }
        }

        for (int l = 1; l < temp_layers.size(); l++) {
            int N_l = temp_layers[l]->num_neuros;
            int N_prev = temp_layers[l - 1]->num_neuros;

            size_t size_bias = N_l * sizeof(float);
            size_t size_weights = N_l * N_prev * sizeof(float);
            clFinish(queue);
            err = clEnqueueReadBuffer(queue, bias[l - 1].memory, CL_TRUE, 0, size_bias, this->system->model->bias[l - 1], 0, nullptr, nullptr);
            OCL_SAFE_CALL(err);

            clFinish(queue);
            err = clEnqueueReadBuffer(queue, weights[l - 1].memory, CL_TRUE, 0, size_weights, this->system->model->weights[l - 1], 0, nullptr, nullptr);
            OCL_SAFE_CALL(err);

            clFinish(queue);
            err = clEnqueueReadBuffer(queue, m_w[l - 1].memory, CL_TRUE, 0, size_weights, this->system->m_w[l - 1].data(), 0, nullptr, nullptr);
            OCL_SAFE_CALL(err);

            clFinish(queue);
            err = clEnqueueReadBuffer(queue, v_w[l - 1].memory, CL_TRUE, 0, size_weights, this->system->v_w[l - 1].data(), 0, nullptr, nullptr);
            OCL_SAFE_CALL(err);

            clFinish(queue);
            err = clEnqueueReadBuffer(queue, m_b[l - 1].memory, CL_TRUE, 0, size_bias, this->system->m_b[l - 1].data(), 0, nullptr, nullptr);
            OCL_SAFE_CALL(err);

            clFinish(queue);
            err = clEnqueueReadBuffer(queue, v_b[l - 1].memory, CL_TRUE, 0, size_bias, this->system->v_b[l - 1].data(), 0, nullptr, nullptr);
            OCL_SAFE_CALL(err);
        }

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
        if (system->best_loss > train_loss) {
            system->best_loss = train_loss;
            this->system->steal_weights_bias(system->model->weights, system->model->bias);
            this->system->best_epoch = this->system->curr_epochs;
        }
        this->system->training_view->set_epochs(this->system->training_view->get_epochs() - 1);
        emit epoch_done(train_loss, test_loss);

        qint64 elapsed = epochTimer.elapsed();

        if (elapsed < MIN_EPOCH_TIME_MS) {
            QThread::msleep(MIN_EPOCH_TIME_MS - elapsed);
        }

    }

    clReleaseCommandQueue(queue);

    delete sharded_data.first;
    delete sharded_data.second;

    this->system->set_is_training(false);

    emit finished(true, "");
}

SamArray::SamArray(cl_mem mem, size_t s) noexcept
    : memory(mem), size(s), is_inited(mem != nullptr) {}

SamArray::SamArray(SamArray&& other) noexcept
    : memory(other.memory), size(other.size), is_inited(other.is_inited)
{
    other.memory = nullptr;
    other.size = 0;
    other.is_inited = false;
}

SamArray& SamArray::operator=(SamArray&& other) noexcept {
    if (this != &other) {
        release();
        memory = other.memory;
        size = other.size;
        is_inited = other.is_inited;

        other.memory = nullptr;
        other.size = 0;
        other.is_inited = false;
    }
    return *this;
}

SamArray::~SamArray() {
    release();
}

void SamArray::clear() { release(); }

void SamArray::release() noexcept {
    if (is_inited && memory) {
        clReleaseMemObject(memory);
    }
    memory = nullptr;
    size = 0;
    is_inited = false;
}

SamArray::SamArray(const SamArray& other) {
    memory = other.memory;
    size = other.size;
    is_inited = other.is_inited;
    if (is_inited && memory)
        clRetainMemObject(memory);
}

SamArray& SamArray::operator=(const SamArray& other) {
    if (this != &other) {
        release();
        memory = other.memory;
        size = other.size;
        is_inited = other.is_inited;
        if (is_inited && memory)
            clRetainMemObject(memory);
    }
    return *this;
}
