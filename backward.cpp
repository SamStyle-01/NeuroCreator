#include "backward.h"
#include "samsystem.h"
#include "dataframe.h"
#include "samtraining.h"
#include <fstream>

BackWard::BackWard(SamSystem *system, QObject *parent) : QObject(parent) {
    this->system = system;
}

void BackWard::doWork() {
    auto temp_layers = system->model->get_layers();

    // Обработка данных
    cl_int err;
    // Создание контекста
    cl_context context = clCreateContext(0, 1, &system->curr_device, nullptr, nullptr, &err);
    OCL_SAFE_CALL(err);

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

    int train_cols = system->data->get_cols() - temp_layers.back()->num_neuros;
    auto& data = system->data->get_data();

    int common_size_batch = 512;
    while (this->system->training_view->get_epochs() > 0) {
        for (int i = 0; i < system->data->get_rows(); i += common_size_batch) {
            const int size_batch = std::min(common_size_batch, system->data->get_rows() - i);

            QVector<float> input_vector(size_batch * temp_layers[0]->num_neuros);;

            for (int j = 0; j < size_batch; j++)
                for (int k = 0; k < train_cols; k++)
                    input_vector[j * train_cols + k] = data[k][i + j];

            QVector<QVector<float>> pre_activations;
            QVector<QVector<float>> activations;
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
                    test_data[q - train_cols][z] = data[q][z];
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
                for (int i = 0; i < delta.size(); i++) {
                    delta[i] *= temp[i];
                }
            }
            else if (activations_layers.back() == Activation::SIGMOID) {
                QVector<float> temp = activations[activations.size() - 1];
                Sigmoid_func_deriv(temp);
                for (int i = 0; i < delta.size(); i++) {
                    delta[i] *= temp[i];
                }
            }
            else if (activations_layers.back() == Activation::TANH) {
                QVector<float> temp = activations[activations.size() - 1];
                Tanh_func_deriv(temp);
                for (int i = 0; i < delta.size(); i++) {
                    delta[i] *= temp[i];
                }
            }
            else if (activations_layers.back() == Activation::SOFTMAX) {
                delta = SoftMax_func_deriv(output, test_data);
            }
        }
        this->system->training_view->set_epochs(this->system->training_view->get_epochs() - 1);
    }
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    emit finished(true, "");
}

void BackWard::ReLU_func_deriv(QVector<float>& vector) {
    for (int i = 0; i < vector.size(); i++) {
        vector[i] = vector[i] > 0 ? 1 : 0;
    }
}

QVector<float> BackWard::SoftMax_func_deriv(const QVector<QVector<float>>& predicted, const QVector<QVector<float>>& target) {
    int batch_size = predicted.size();
    int num_outputs = predicted[0].size();

    QVector<float> grad(num_outputs, 0.0f);

    for (int b = 0; b < batch_size; b++) {
        for (int j = 0; j < num_outputs; j++) {
            grad[j] += predicted[b][j] - target[b][j];
        }
    }

    for (float& g : grad)
        g /= static_cast<float>(batch_size);

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
    QVector<float> grad(outputs, 0);

    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < outputs; j++) {
            grad[j] += predicted[i][j] - true_vals[i][j];
        }
    }

    for (float &g : grad) g /= (float)batch;
    return grad;
}

QVector<float> BackWard::MAE_deriv(const QVector<QVector<float>>& predicted, const QVector<QVector<float>>& true_vals) {
    int batch = predicted.size();
    int outputs = predicted[0].size();
    QVector<float> grad(outputs, 0);

    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < outputs; j++) {
            float diff = predicted[i][j] - true_vals[i][j];
            grad[j] += (diff > 0) ? 1.0f : (diff < 0 ? -1.0f : 0.0f);
        }
    }

    for (float &g : grad) g /= (float)batch;
    return grad;
}

QVector<float> BackWard::CrossEntropy_deriv(const QVector<QVector<float>>& predicted, const QVector<QVector<float>>& true_vals) {
    int batch = predicted.size();
    int outputs = predicted[0].size();
    QVector<float> grad(outputs, 0);

    for (int i = 0; i < batch; i++) {
        for (int j = 0; j < outputs; j++) {
            grad[j] -= true_vals[i][j] / (predicted[i][j] + 1e-8f);
        }
    }

    for (float &g : grad) g /= (float)batch;
    return grad;
}
