#include "samtest.h"
#include "samsystem.h"
#include "dataframe.h"
#include "samtraining.h"

SamTest::SamTest(SamSystem *system, QObject *parent) : QObject(parent) {
    this->system = system;
}

void SamTest::doWork(DataFrame* processing_data, bool delete_data, cl_context& context) {
    auto temp_layers = system->model->get_layers();

    // Обработка данных
    cl_int err;

    // Создание командной очереди
    cl_command_queue queue = clCreateCommandQueue(context, system->curr_device, 0, &err);
    OCL_SAFE_CALL(err);

    int final_layer_size = temp_layers.back()->num_neuros;

    QVector<QVector<float>> output(final_layer_size);
    for (auto &vec : output)
        vec.reserve(processing_data->get_rows());

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
    auto& data = processing_data->get_data();

    QVector<cl_mem> bias;
    QVector<cl_mem> weights;

    for (int l = 1; l < temp_layers.size(); l++) {
        int N_l = temp_layers[l]->num_neuros;
        int N_prev = temp_layers[l - 1]->num_neuros;

        size_t size_bias = N_l * sizeof(float);
        bias.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_bias, system->model->bias[l - 1], &err));

        size_t size_weights = N_l * N_prev * sizeof(float);
        weights.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_weights, system->model->weights[l - 1], &err));
    }

    QVector<QVector<float>> test_data;
    for (int q = train_cols; q < data.size(); q++) {
        test_data.push_back(QVector<float>(data[q]));
    }

    for (int i = 0; i < processing_data->get_rows(); i += 512) {
        const int size_batch = std::min(512, processing_data->get_rows() - i);
        QVector<float> input_vector(size_batch * temp_layers[0]->num_neuros);;

        for (int j = 0; j < size_batch; j++)
            for (int k = 0; k < train_cols; k++)
                input_vector[j * train_cols + k] = data[k][i + j];

        // Создание буферов (память на устройстве)
        size_t size_R = input_vector.size() * sizeof(float);
        cl_mem cl_result_vector = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_R, input_vector.data(), &err);
        OCL_SAFE_CALL(err);
        for (int c = 0; c < temp_layers.size(); c++) {

            int size = size_batch * temp_layers[c]->num_neuros;
            if (activations_layers[c] == Activation::RELU) {
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_relu, 0, sizeof(cl_mem), &cl_result_vector));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_relu, 1, sizeof(cl_int), &size));

                // Запуск ядра
                size_t global_work_size[] = { (size_t)size };

                err = clEnqueueNDRangeKernel(queue, system->kernel_relu, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }
            else if (activations_layers[c] == Activation::SIGMOID) {
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_sigmoid, 0, sizeof(cl_mem), &cl_result_vector));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_sigmoid, 1, sizeof(cl_int), &size));

                // Запуск ядра
                size_t global_work_size[] = { (size_t)size };

                err = clEnqueueNDRangeKernel(queue, system->kernel_sigmoid, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }
            else if (activations_layers[c] == Activation::TANH) {
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_tanh, 0, sizeof(cl_mem), &cl_result_vector));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_tanh, 1, sizeof(cl_int), &size));

                // Запуск ядра
                size_t global_work_size[] = { (size_t)size };

                err = clEnqueueNDRangeKernel(queue, system->kernel_tanh, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }

            if (c != temp_layers.size() - 1) {
                // Создание буферов (память на устройстве)

                int size_R2 = size_batch * temp_layers[c + 1]->num_neuros * sizeof(float);
                cl_mem cl_result_vector_post = clCreateBuffer(context, CL_MEM_READ_WRITE, size_R2, nullptr, &err);
                OCL_SAFE_CALL(err);

                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 0, sizeof(cl_mem), &cl_result_vector_post));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 1, sizeof(cl_mem), &cl_result_vector));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 2, sizeof(cl_mem), &weights[c]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 3, sizeof(cl_mem), &bias[c]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 4, sizeof(cl_int), &size_batch));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 5, sizeof(cl_int), &temp_layers[c]->num_neuros));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 6, sizeof(cl_int), &temp_layers[c + 1]->num_neuros));

                // Запуск ядра
                size_t global_work_size[] = { (size_t)size_batch, (size_t)temp_layers[c + 1]->num_neuros };

                err = clEnqueueNDRangeKernel(queue, system->kernel_matrix_mult, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                OCL_SAFE_CALL(err);
                clReleaseMemObject(cl_result_vector);
                cl_result_vector = cl_result_vector_post;
            }
        }
        input_vector.clear();
        input_vector.resize(size_batch * temp_layers.back()->num_neuros, 0.0f);
        clFinish(queue);
        err = clEnqueueReadBuffer(queue, cl_result_vector, CL_TRUE, 0, size_batch * temp_layers.back()->num_neuros * sizeof(float),
                                  input_vector.data(), 0, nullptr, nullptr);
        OCL_SAFE_CALL(err);

        // Очистка ресурсов
        clReleaseMemObject(cl_result_vector);
        for (int l = 0; l < size_batch; l++)
            for (int d = 0; d < final_layer_size; d++)
                output[d].push_back(input_vector[l * final_layer_size + d]);
    }
    if (delete_data)
        delete processing_data;

    for (int l = 0; l < bias.size(); l++) {
        clReleaseMemObject(bias[l]);
        clReleaseMemObject(weights[l]);
    }
    clReleaseCommandQueue(queue);

    auto loss_func = this->system->training_view->get_loss_func();
    float loss;
    switch (loss_func) {
        case LossFunc::MSE: {
            loss = this->system->MSE_loss(output, test_data);
            break;
        }
        case LossFunc::MAE: {
            loss = this->system->MAE_loss(output, test_data);
            break;
        }
        case LossFunc::CROSSENTROPY: {
            loss = this->system->CrossEntropy_loss(output, test_data);
            break;
        }
    }

    emit finished(true, "", loss);
}


QPair<QString, float> SamTest::doWork(DataFrame* processing_data, cl_context& context) {
    auto temp_layers = system->model->get_layers();

    // Обработка данных
    cl_int err;

    // Создание командной очереди
    cl_command_queue queue = clCreateCommandQueue(context, system->curr_device, 0, &err);
    OCL_SAFE_CALL(err);

    int final_layer_size = temp_layers.back()->num_neuros;

    QVector<QVector<float>> output(final_layer_size);
    for (auto &vec : output)
        vec.reserve(processing_data->get_rows());

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
    auto& data = processing_data->get_data();

    QVector<cl_mem> bias;
    QVector<cl_mem> weights;

    for (int l = 1; l < temp_layers.size(); l++) {
        int N_l = temp_layers[l]->num_neuros;
        int N_prev = temp_layers[l - 1]->num_neuros;

        size_t size_bias = N_l * sizeof(float);
        bias.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_bias, system->model->bias[l - 1], &err));

        size_t size_weights = N_l * N_prev * sizeof(float);
        weights.push_back(clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_weights, system->model->weights[l - 1], &err));
    }

    for (int i = 0; i < processing_data->get_rows(); i += 512) {
        const int size_batch = std::min(512, processing_data->get_rows() - i);
        QVector<float> input_vector(size_batch * temp_layers[0]->num_neuros);;

        for (int j = 0; j < size_batch; j++)
            for (int k = 0; k < train_cols; k++)
                input_vector[j * train_cols + k] = data[k][i + j];

        size_t size_R = input_vector.size() * sizeof(float);
        cl_mem cl_result_vector = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, size_R, input_vector.data(), &err);
        OCL_SAFE_CALL(err);
        for (int c = 0; c < temp_layers.size(); c++) {
            // Создание буферов (память на устройстве)

            int size = size_batch * temp_layers[c]->num_neuros;
            if (activations_layers[c] == Activation::RELU) {
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_relu, 0, sizeof(cl_mem), &cl_result_vector));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_relu, 1, sizeof(cl_int), &size));

                // Запуск ядра
                size_t global_work_size[] = { (size_t)size };

                err = clEnqueueNDRangeKernel(queue, system->kernel_relu, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }
            else if (activations_layers[c] == Activation::SIGMOID) {
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_sigmoid, 0, sizeof(cl_mem), &cl_result_vector));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_sigmoid, 1, sizeof(cl_int), &size));

                // Запуск ядра
                size_t global_work_size[] = { (size_t)size };

                err = clEnqueueNDRangeKernel(queue, system->kernel_sigmoid, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }
            else if (activations_layers[c] == Activation::TANH) {
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_tanh, 0, sizeof(cl_mem), &cl_result_vector));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_tanh, 1, sizeof(cl_int), &size));

                // Запуск ядра
                size_t global_work_size[] = { (size_t)size };

                err = clEnqueueNDRangeKernel(queue, system->kernel_tanh, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }

            if (c != temp_layers.size() - 1) {
                int size_R2 = size_batch * temp_layers[c + 1]->num_neuros * sizeof(float);
                cl_mem cl_result_vector_post = clCreateBuffer(context, CL_MEM_READ_WRITE, size_R2, nullptr, &err);
                OCL_SAFE_CALL(err);

                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 0, sizeof(cl_mem), &cl_result_vector_post));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 1, sizeof(cl_mem), &cl_result_vector));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 2, sizeof(cl_mem), &weights[c]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 3, sizeof(cl_mem), &bias[c]));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 4, sizeof(cl_int), &size_batch));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 5, sizeof(cl_int), &temp_layers[c]->num_neuros));
                OCL_SAFE_CALL(clSetKernelArg(system->kernel_matrix_mult, 6, sizeof(cl_int), &temp_layers[c + 1]->num_neuros));

                // Запуск ядра
                size_t global_work_size[] = { (size_t)size_batch, (size_t)temp_layers[c + 1]->num_neuros };

                err = clEnqueueNDRangeKernel(queue, system->kernel_matrix_mult, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                OCL_SAFE_CALL(err);
                clReleaseMemObject(cl_result_vector);
                cl_result_vector = cl_result_vector_post;
            }
        }
        input_vector.clear();
        input_vector.resize(size_batch * temp_layers.back()->num_neuros, 0.0f);
        clFinish(queue);
        err = clEnqueueReadBuffer(queue, cl_result_vector, CL_TRUE, 0, size_batch * temp_layers.back()->num_neuros * sizeof(float),
                                  input_vector.data(), 0, nullptr, nullptr);
        OCL_SAFE_CALL(err);

        // Очистка ресурсов
        clReleaseMemObject(cl_result_vector);
        for (int l = 0; l < size_batch; l++)
            for (int d = 0; d < final_layer_size; d++)
                output[d].push_back(input_vector[l * final_layer_size + d]);
    }

    for (int l = 0; l < bias.size(); l++) {
        clReleaseMemObject(bias[l]);
        clReleaseMemObject(weights[l]);
    }
    clReleaseCommandQueue(queue);

    QVector<QVector<float>> test_data;
    for (int q = train_cols; q < data.size(); q++) {
        test_data.push_back(QVector<float>(data[q]));
    }

    auto loss_func = this->system->training_view->get_loss_func();
    float loss;
    switch (loss_func) {
    case LossFunc::MSE: {
        loss = this->system->MSE_loss(output, test_data);
        break;
    }
    case LossFunc::MAE: {
        loss = this->system->MAE_loss(output, test_data);
        break;
    }
    case LossFunc::CROSSENTROPY: {
        loss = this->system->CrossEntropy_loss(output, test_data);
        break;
    }
    }

    return qMakePair("", loss);
}
