#include "backward.h"
#include "samsystem.h"
#include "dataframe.h"
#include "samsystem.h"
#include "samtraining.h"
#include "samtest.h"
#include <fstream>

BackWard::BackWard(SamSystem *system, QObject *parent) : QObject(parent) {
    this->system = system;
}

void BackWard::doWork(cl_context& context) {
    auto temp_layers = system->model->get_layers();
    float eta = this->system->training_view->get_learning_rate();

    // Обработка данных
    cl_int err;

    // Создание командной очереди
    cl_command_queue queue = clCreateCommandQueue(context, system->curr_device, 0, &err);
    OCL_SAFE_CALL(err);

    // Загрузка исходного кода ядра
    std::ifstream sourceFile("../../MatrixVectorMultiplicationKernelBackWard.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    if(sourceCode.empty()) {
        emit finished(false, "Не удалось считать файл ядра");
        return;
    }
    const char *source_str = sourceCode.c_str();
    size_t source_len = sourceCode.length();

    // Создание программы и ее компиляция
    cl_program program = clCreateProgramWithSource(context, 1, &source_str, &source_len, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(program, 1, &system->curr_device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, system->curr_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        QVector<char> build_log(log_size);
        clGetProgramBuildInfo(program, system->curr_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        OCL_SAFE_CALL(err);
        emit finished(false, "Ошибка компиляции ядра");
        return;
    }

    // Создание и настройка ядра
    cl_kernel kernel = clCreateKernel(program, "matrixBatchMul", &err);
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

    int common_size_batch = this->system->training_view->get_batch_size();
    while (this->system->training_view->get_epochs() > 0 && this->system->get_is_training()) {
        sharded_data.first->random_shuffle();
        for (int i = 0; i < sharded_data.first->get_rows(); i += common_size_batch) {
            const int size_batch = std::min(common_size_batch, sharded_data.first->get_rows() - i);

            QVector<float> input_vector(size_batch * temp_layers[0]->num_neuros);;

            for (int j = 0; j < size_batch; j++)
                for (int k = 0; k < train_cols; k++)
                    input_vector[j * train_cols + k] = data[k][i + j];

            QVector<QVector<float>> pre_activations;
            QVector<QVector<float>> activations;
            pre_activations.push_back(input_vector);
            activations.push_back(input_vector);
            for (int c = 0; c < temp_layers.size() - 1; c++) {
                QVector<float> result_vector(size_batch * temp_layers[c + 1]->num_neuros, 0.0f);

                // Создание буферов (память на устройстве)
                size_t size_A = temp_layers[c + 1]->num_neuros * temp_layers[c]->num_neuros * sizeof(float);
                size_t size_B = size_batch * temp_layers[c]->num_neuros * sizeof(float);
                size_t size_R = size_batch * temp_layers[c + 1]->num_neuros * sizeof(float);
                size_t size_bias = temp_layers[c + 1]->num_neuros * sizeof(float);

                cl_mem cl_matrix_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_A, system->model->get_weight_T(c), &err);
                OCL_SAFE_CALL(err);
                cl_mem cl_vector_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_B, input_vector.data(), &err);
                OCL_SAFE_CALL(err);
                cl_mem cl_result_vector = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_R, nullptr, &err);
                OCL_SAFE_CALL(err);
                cl_mem cl_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_bias, system->model->get_bias(c), &err);
                OCL_SAFE_CALL(err);

                OCL_SAFE_CALL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_result_vector));
                OCL_SAFE_CALL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_vector_B));
                OCL_SAFE_CALL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &cl_matrix_A));
                OCL_SAFE_CALL(clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_bias));
                OCL_SAFE_CALL(clSetKernelArg(kernel, 4, sizeof(cl_int), &size_batch));
                OCL_SAFE_CALL(clSetKernelArg(kernel, 5, sizeof(cl_int), &temp_layers[c]->num_neuros));
                OCL_SAFE_CALL(clSetKernelArg(kernel, 6, sizeof(cl_int), &temp_layers[c + 1]->num_neuros));

                // Запуск ядра
                size_t global_work_size[] = { (size_t)size_batch, (size_t)temp_layers[c + 1]->num_neuros };

                err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
                clFinish(queue);

                // Чтение результата
                err = clEnqueueReadBuffer(queue, cl_result_vector, CL_TRUE, 0, size_R, result_vector.data(), 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                pre_activations.push_back(result_vector);

                if (activations_layers[c] == Activation::SOFTMAX) {
                    system->SoftMax_func(result_vector);
                }
                else if (activations_layers[c] == Activation::RELU) {
                    system->ReLU_func(result_vector);
                }
                else if (activations_layers[c] == Activation::SIGMOID) {
                    system->Sigmoid_func(result_vector);
                }
                else if (activations_layers[c] == Activation::TANH) {
                    system->Tanh_func(result_vector);
                }

                activations.push_back(result_vector);
                input_vector = result_vector;

                // Очистка ресурсов
                clReleaseMemObject(cl_matrix_A);
                clReleaseMemObject(cl_vector_B);
                clReleaseMemObject(cl_result_vector);
            }

            // Преобразование результатов в удобный матричный формат
            QVector<QVector<float>> output(final_layer_size);
            for (auto &vec : output)
                vec.reserve(size_batch);
            for (int l = 0; l < size_batch; l++)
                for (int d = 0; d < final_layer_size; d++)
                    output[d].push_back(input_vector[l * final_layer_size + d]);

            QVector<QVector<float>> test_data;
            for (int q = train_cols; q < data.size(); q++) {
                test_data.push_back(QVector<float>(size_batch));
                for (int z = i; z < i + size_batch; z++) {
                    test_data[q - train_cols][z - i] = data[q][z];
                }
            }

            // Обратное распространение ошибки
            auto loss_func = this->system->training_view->get_loss_func();
            QVector<float> delta;
            switch (loss_func) {
                case LossFunc::MSE: {
                    delta = MSE_deriv(output, test_data);
                    break;
                }
                case LossFunc::MAE: {
                    delta = MAE_deriv(output, test_data);
                    break;
                }
                case LossFunc::CROSSENTROPY: {
                    delta = CrossEntropy_deriv(output, test_data);
                    break;
                }
            }

            if (activations_layers.back() == Activation::RELU) {
                QVector<float> temp = pre_activations[pre_activations.size() - 1];
                ReLU_func_deriv(temp);
                for (int i1 = 0; i1 < delta.size(); i1++) {
                    delta[i1] *= temp[i1];
                }
            }
            else if (activations_layers.back() == Activation::SIGMOID) {
                QVector<float> temp = activations[activations.size() - 1];
                Sigmoid_func_deriv(temp);
                for (int i1 = 0; i1 < delta.size(); i1++) {
                    delta[i1] *= temp[i1];
                }
            }
            else if (activations_layers.back() == Activation::TANH) {
                QVector<float> temp = activations[activations.size() - 1];
                Tanh_func_deriv(temp);
                for (int i1 = 0; i1 < delta.size(); i1++) {
                    delta[i1] *= temp[i1];
                }
            }
            else if (activations_layers.back() == Activation::SOFTMAX) {
                delta = SoftMax_func_deriv(output, test_data);
            }

            QVector<QVector<float>> hidden_delta(temp_layers.size());

            hidden_delta[temp_layers.size() - 1] = delta;

            for (int l = temp_layers.size() - 2; l >= 0; l--) {
                hidden_delta[l] = QVector<float>(size_batch * temp_layers[l]->num_neuros, 0.0f);
                for (int m = 0; m < size_batch; m++) {
                    for (int ne = 0; ne < temp_layers[l]->num_neuros; ne++) {
                        for (int ne2 = 0; ne2 < temp_layers[l + 1]->num_neuros; ne2++) {
                            hidden_delta[l][m * temp_layers[l]->num_neuros + ne] += hidden_delta[l + 1][m * temp_layers[l + 1]->num_neuros + ne2]
                                              * system->model->weights[l][ne2 * temp_layers[l]->num_neuros + ne];
                        }
                        if (activations_layers[l] == Activation::RELU) {
                            auto temp = pre_activations[l];
                            ReLU_func_deriv(temp);
                            hidden_delta[l][m * temp_layers[l]->num_neuros + ne] *= temp[m * temp_layers[l]->num_neuros + ne];
                        }
                        else if (activations_layers[l] == Activation::SIGMOID) {
                            auto temp = activations[l];
                            Sigmoid_func_deriv(temp);
                            hidden_delta[l][m * temp_layers[l]->num_neuros + ne] *= temp[m * temp_layers[l]->num_neuros + ne];
                        }
                        else if (activations_layers[l] == Activation::TANH) {
                            auto temp = activations[l];
                            Tanh_func_deriv(temp);
                            hidden_delta[l][m * temp_layers[l]->num_neuros + ne] *= temp[m * temp_layers[l]->num_neuros + ne];
                        }
                    }
                }
            }

            QVector<QVector<float>> db(temp_layers.size());
            QVector<QVector<float>> dW(temp_layers.size());

            for (int l = 1; l < temp_layers.size(); l++) {
                int N_l = temp_layers[l]->num_neuros;
                int N_prev = temp_layers[l - 1]->num_neuros;

                db[l] = QVector<float>(N_l, 0.0f);
                for (int ne = 0; ne < N_l; ne++) {
                    for (int b = 0; b < size_batch; b++) {
                        db[l][ne] += hidden_delta[l][b * N_l + ne];
                    }
                    db[l][ne] /= (float)size_batch;
                }

                dW[l] = QVector<float>(N_l * N_prev, 0.0f);
                for (int ne = 0; ne < N_l; ne++) {
                    for (int ne_prev = 0; ne_prev < N_prev; ne_prev++) {
                        for (int b = 0; b < size_batch; b++) {
                            dW[l][ne * N_prev + ne_prev] +=
                                activations[l - 1][b * N_prev + ne_prev] * hidden_delta[l][b * N_l + ne];
                        }
                        dW[l][ne * N_prev + ne_prev] /= (float)size_batch;
                    }
                }
            }

            this->system->t++;

            for (int l = 1; l < temp_layers.size(); l++) {
                int N_l = temp_layers[l]->num_neuros;
                int N_prev = temp_layers[l - 1]->num_neuros;

                for (int b = 0; b < N_l; b++) {
                    this->system->m_b[l - 1][b] = this->system->beta1 * this->system->m_b[l - 1][b] + (1 - this->system->beta1) * db[l][b];
                    this->system->v_b[l - 1][b] = this->system->beta2 * this->system->v_b[l - 1][b] + (1 - this->system->beta2) * pow(db[l][b], 2);
                    float m_hat = this->system->m_b[l - 1][b] / (1 - pow(this->system->beta1, this->system->t));
                    float v_hat = this->system->v_b[l - 1][b] / (1 - pow(this->system->beta2, this->system->t));

                    this->system->model->bias[l - 1][b] -= eta * m_hat / (sqrt(v_hat) + this->system->eps);
                }

                for (int w = 0; w < N_l; w++) {
                    for (int w2 = 0; w2 < N_prev; w2++) {
                        int index = w * N_prev + w2;
                        this->system->m_w[l - 1][index] = this->system->beta1 * this->system->m_w[l - 1][index] + (1 - this->system->beta1) * dW[l][index];
                        this->system->v_w[l - 1][index] = this->system->beta2 * this->system->v_w[l - 1][index] + (1 - this->system->beta2) * pow(dW[l][index], 2);

                        float m_hat = this->system->m_w[l - 1][index] / (1 - pow(this->system->beta1, this->system->t));
                        float v_hat = this->system->v_w[l - 1][index] / (1 - pow(this->system->beta2, this->system->t));

                        this->system->model->weights[l - 1][index] -= eta * m_hat / (sqrt(v_hat) + this->system->eps);
                    }
                }
            }
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
        double maxY = -std::numeric_limits<double>::infinity();
        qDebug() << this->system->curr_epochs << " " << train_loss << " " << std::isfinite(train_loss) << " " << (maxY < train_loss);

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
                this->system->steal_weights_bias(system->model->weights, system->model->bias);
                this->system->best_epoch = this->system->curr_epochs;
            }
        }

        this->system->model->update_weights();
        emit epoch_done(train_loss, test_loss);

        this->system->training_view->set_epochs(this->system->training_view->get_epochs() - 1);
    }
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);

    delete sharded_data.first;
    delete sharded_data.second;

    this->system->set_is_training(false);

    emit finished(true, "");
}

void BackWard::clip_gradients(QVector<float>& grad, float clip_value) {
    for (float& g : grad) {
        if (g > clip_value) g = clip_value;
        else if (g < -clip_value) g = -clip_value;
    }
}

void BackWard::ReLU_func_deriv(QVector<float>& vector) {
    for (int i = 0; i < vector.size(); i++) {
        vector[i] = vector[i] > 0 ? 1 : 0;
    }
}

QVector<float> BackWard::SoftMax_func_deriv(const QVector<QVector<float>>& predicted, const QVector<QVector<float>>& target) {
    int batch = predicted.size();
    int outputs = predicted[0].size();
    QVector<float> grad(outputs * batch);

    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < outputs; j++) {
            grad[i * outputs + j] = predicted[i][j] - target[i][j];
        }
    }

    return grad;
}

void BackWard::Sigmoid_func_deriv(QVector<float>& vector) {
    for (int i = 0; i < vector.size(); i++) {
        vector[i] = vector[i] * (1 - vector[i]);
    }
}

void BackWard::Tanh_func_deriv(QVector<float>& vector) {
    for (int i = 0; i < vector.size(); i++) {
        vector[i] = 1 - pow(vector[i], 2);
    }
}

QVector<float> BackWard::MSE_deriv(const QVector<QVector<float>>& predicted, const QVector<QVector<float>>& true_vals) {
    int batch = predicted.size();
    int outputs = predicted[0].size();
    QVector<float> grad(outputs * batch);

    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < outputs; j++) {
            grad[i * outputs + j] = predicted[i][j] - true_vals[i][j];
        }
    }

    return grad;
}

QVector<float> BackWard::MAE_deriv(const QVector<QVector<float>>& predicted, const QVector<QVector<float>>& true_vals) {
    int batch = predicted.size();
    int outputs = predicted[0].size();
    QVector<float> grad(outputs * batch);

    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < outputs; j++) {
            float diff = predicted[i][j] - true_vals[i][j];
            grad[i * outputs + j] = (diff > 0) ? 1.0f : (diff < 0 ? -1.0f : 0.0f);
        }
    }

    return grad;
}

QVector<float> BackWard::CrossEntropy_deriv(const QVector<QVector<float>>& predicted, const QVector<QVector<float>>& true_vals) {
    int batch = predicted.size();
    int outputs = predicted[0].size();
    QVector<float> grad(outputs * batch);

    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < outputs; j++) {
            grad[i * outputs + j] = -true_vals[i][j] / (predicted[i][j] + 1e-8f);
        }
    }

    return grad;
}
