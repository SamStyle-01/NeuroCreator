#include "samthreadsystem.h"
#include "samsystem.h"

SamThreadSystem::SamThreadSystem() {}

void SamThreadSystem::set_device(cl_device_id curr_device, cl_context& context, QVector<cl_kernel>& kernels) {
    cl_int err;
    // Загрузка исходного кода ядра
    // Умножение матриц
    std::ifstream source_file_matrix_mult("./MatrixVectorMultiplicationKernelBackWard.cl");
    std::string source_code_matrix_mult(std::istreambuf_iterator<char>(source_file_matrix_mult), (std::istreambuf_iterator<char>()));
    const char *source_str = source_code_matrix_mult.c_str();
    size_t source_len = source_code_matrix_mult.length();

    // Ядро backprop_linear
    std::ifstream source_file_backprop_linear("./backprop_linear_kernel.cl");
    std::string source_code_backprop_linear(std::istreambuf_iterator<char>(source_file_backprop_linear), (std::istreambuf_iterator<char>()));
    const char *source_str_backprop_linear = source_code_backprop_linear.c_str();
    size_t source_len_backprop_linear = source_code_backprop_linear.length();

    // Ядро vectors_mult
    std::ifstream source_file_vectors_mult("./vectors_mult.cl");
    std::string source_code_vectors_mult(std::istreambuf_iterator<char>(source_file_vectors_mult), (std::istreambuf_iterator<char>()));
    const char *source_str_vectors_mult = source_code_vectors_mult.c_str();
    size_t source_len_vectors_mult = source_code_vectors_mult.length();

    // Ядро weights_first_step
    std::ifstream source_file_weights_first_step("./weights_first_step.cl");
    std::string source_code_weights_first_step(std::istreambuf_iterator<char>(source_file_weights_first_step), (std::istreambuf_iterator<char>()));
    const char *source_str_weights_first_step = source_code_weights_first_step.c_str();
    size_t source_len_weights_first_step = source_code_weights_first_step.length();

    // Ядро bias_first_step
    std::ifstream source_file_bias_first_step("./bias_first_step.cl");
    std::string source_code_bias_first_step(std::istreambuf_iterator<char>(source_file_bias_first_step), (std::istreambuf_iterator<char>()));
    const char *source_str_bias_first_step = source_code_bias_first_step.c_str();
    size_t source_len_bias_first_step = source_code_bias_first_step.length();

    // Ядро weights_last_step
    std::ifstream source_file_weights_last_step("./weights_last_step.cl");
    std::string source_code_weights_last_step(std::istreambuf_iterator<char>(source_file_weights_last_step), (std::istreambuf_iterator<char>()));
    const char *source_str_weights_last_step = source_code_weights_last_step.c_str();
    size_t source_len_weights_last_step = source_code_weights_last_step.length();

    // Ядро bias_last_step
    std::ifstream source_file_bias_last_step("./bias_last_step.cl");
    std::string source_code_bias_last_step(std::istreambuf_iterator<char>(source_file_bias_last_step), (std::istreambuf_iterator<char>()));
    const char *source_str_bias_last_step = source_code_bias_last_step.c_str();
    size_t source_len_bias_last_step = source_code_bias_last_step.length();

    // ReLU
    std::ifstream source_file_relu("./relu_func.cl");
    std::string source_code_relu(std::istreambuf_iterator<char>(source_file_relu), (std::istreambuf_iterator<char>()));
    const char *source_str_relu = source_code_relu.c_str();
    size_t source_len_relu = source_code_relu.length();

    // Sigmoid
    std::ifstream source_sigmoid("./sigmoid_func.cl");
    std::string source_code_sigmoid(std::istreambuf_iterator<char>(source_sigmoid), (std::istreambuf_iterator<char>()));
    const char *source_str_sigmoid = source_code_sigmoid.c_str();
    size_t source_len_sigmoid = source_code_sigmoid.length();

    // Tanh
    std::ifstream source_tanh("./tanh_func.cl");
    std::string source_code_tanh(std::istreambuf_iterator<char>(source_tanh), (std::istreambuf_iterator<char>()));
    const char *source_str_tanh = source_code_tanh.c_str();
    size_t source_len_tanh = source_code_tanh.length();

    // Производная ReLU
    std::ifstream source_file_relu_deriv("./relu_derivative_kernel.cl");
    std::string source_code_relu_deriv(std::istreambuf_iterator<char>(source_file_relu_deriv), (std::istreambuf_iterator<char>()));
    const char *source_str_relu_deriv = source_code_relu_deriv.c_str();
    size_t source_len_relu_deriv = source_code_relu_deriv.length();

    // Произодная Sigmoid
    std::ifstream source_sigmoid_deriv("./sigmoid_derivative_kernel.cl");
    std::string source_code_sigmoid_deriv(std::istreambuf_iterator<char>(source_sigmoid_deriv), (std::istreambuf_iterator<char>()));
    const char *source_str_sigmoid_deriv = source_code_sigmoid_deriv.c_str();
    size_t source_len_sigmoid_deriv = source_code_sigmoid_deriv.length();

    // Произодная Tanh
    std::ifstream source_tanh_deriv("./tanh_derivative_kernel.cl");
    std::string source_code_tanh_deriv(std::istreambuf_iterator<char>(source_tanh_deriv), (std::istreambuf_iterator<char>()));
    const char *source_str_tanh_deriv = source_code_tanh_deriv.c_str();
    size_t source_len_tanh_deriv = source_code_tanh_deriv.length();

    // Произодная MSE облегчённая версия
    std::ifstream source_mse_deriv("./mse_derivative_kernel.cl");
    std::string source_code_mse_deriv(std::istreambuf_iterator<char>(source_mse_deriv), (std::istreambuf_iterator<char>()));
    const char *source_str_mse_deriv = source_code_mse_deriv.c_str();
    size_t source_len_mse_deriv = source_code_mse_deriv.length();

    // Произодная MAE
    std::ifstream source_mae_deriv("./mae_derivative_kernel.cl");
    std::string source_code_mae_deriv(std::istreambuf_iterator<char>(source_mae_deriv), (std::istreambuf_iterator<char>()));
    const char *source_str_mae_deriv = source_code_mae_deriv.c_str();
    size_t source_len_mae_deriv = source_code_mae_deriv.length();

    // Произодная Softmax
    std::ifstream source_softmax_deriv("./softmax_derivative_kernel.cl");
    std::string source_code_softmax_deriv(std::istreambuf_iterator<char>(source_softmax_deriv), (std::istreambuf_iterator<char>()));
    const char *source_str_softmax_deriv = source_code_softmax_deriv.c_str();
    size_t source_len_softmax_deriv = source_code_softmax_deriv.length();

    // Произодная BCE
    std::ifstream source_bce_deriv("./bce_derivative_kernel.cl");
    std::string source_code_bce_deriv(std::istreambuf_iterator<char>(source_bce_deriv), (std::istreambuf_iterator<char>()));
    const char *source_str_bce_deriv = source_code_bce_deriv.c_str();
    size_t source_len_bce_deriv = source_code_bce_deriv.length();

    // Создание программ и их компиляция
    // Умножение матриц
    cl_program program_matrix_mult = clCreateProgramWithSource(context, 1, &source_str, &source_len, &err);
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    err = clBuildProgram(program_matrix_mult, 1, &curr_device, nullptr, nullptr, nullptr);

    // Ядро backprop_linear
    cl_program backprop_linear_program = clCreateProgramWithSource(context, 1, &source_str_backprop_linear, &source_len_backprop_linear, &err);
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    err = clBuildProgram(backprop_linear_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Ядро vectors_mult
    cl_program vectors_mult_program = clCreateProgramWithSource(context, 1, &source_str_vectors_mult, &source_len_vectors_mult, &err);
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    err = clBuildProgram(vectors_mult_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Ядро weights_first_step
    cl_program weights_first_step_program = clCreateProgramWithSource(context, 1, &source_str_weights_first_step, &source_len_weights_first_step, &err);
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    err = clBuildProgram(weights_first_step_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Ядро bias_first_step
    cl_program bias_first_step_program = clCreateProgramWithSource(context, 1, &source_str_bias_first_step, &source_len_bias_first_step, &err);
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    err = clBuildProgram(bias_first_step_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Ядро weights_last_step
    cl_program weights_last_step_program = clCreateProgramWithSource(context, 1, &source_str_weights_last_step, &source_len_weights_last_step, &err);
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    err = clBuildProgram(weights_last_step_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Ядро bias_last_step
    cl_program bias_last_step_program = clCreateProgramWithSource(context, 1, &source_str_bias_last_step, &source_len_bias_last_step, &err);
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    err = clBuildProgram(bias_last_step_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // ReLU
    cl_program relu_program = clCreateProgramWithSource(context, 1, &source_str_relu, &source_len_relu, &err);
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    err = clBuildProgram(relu_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Sigmoid
    cl_program sigmoid_program = clCreateProgramWithSource(context, 1, &source_str_sigmoid, &source_len_sigmoid, &err);
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    err = clBuildProgram(sigmoid_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Tanh
    cl_program tanh_program = clCreateProgramWithSource(context, 1, &source_str_tanh, &source_len_tanh, &err);
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    err = clBuildProgram(tanh_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Производная ReLU
    cl_program relu_deriv_program = clCreateProgramWithSource(context, 1, &source_str_relu_deriv, &source_len_relu_deriv, &err);
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    err = clBuildProgram(relu_deriv_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Произодная Sigmoid
    cl_program sigmoid_deriv_program = clCreateProgramWithSource(context, 1, &source_str_sigmoid_deriv, &source_len_sigmoid_deriv, &err);
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    err = clBuildProgram(sigmoid_deriv_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Произодная Tanh
    cl_program tanh_deriv_program = clCreateProgramWithSource(context, 1, &source_str_tanh_deriv, &source_len_tanh_deriv, &err);
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    err = clBuildProgram(tanh_deriv_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Произодная MSE
    cl_program mse_deriv_program = clCreateProgramWithSource(context, 1, &source_str_mse_deriv, &source_len_mse_deriv, &err);
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    err = clBuildProgram(mse_deriv_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Произодная MAE
    cl_program mae_deriv_program = clCreateProgramWithSource(context, 1, &source_str_mae_deriv, &source_len_mae_deriv, &err);
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    err = clBuildProgram(mae_deriv_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Произодная Softmax
    cl_program softmax_deriv_program = clCreateProgramWithSource(context, 1, &source_str_softmax_deriv, &source_len_softmax_deriv, &err);
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    err = clBuildProgram(softmax_deriv_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Произодная BCE
    cl_program bce_deriv_program = clCreateProgramWithSource(context, 1, &source_str_bce_deriv, &source_len_bce_deriv, &err);
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    err = clBuildProgram(bce_deriv_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Создание и настройка ядер
    // Умножение матриц
    kernels.push_back(clCreateKernel(program_matrix_mult, "matrixBatchMulBackward", &err));
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    // Ядро backprop_linear
    kernels.push_back(clCreateKernel(backprop_linear_program, "backprop_linear", &err));
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    // Ядро vectors_mult
    kernels.push_back(clCreateKernel(vectors_mult_program, "vector_mult_inplace", &err));
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    // Ядро weights_first_step
    kernels.push_back(clCreateKernel(weights_first_step_program, "compute_dW", &err));
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    // Ядро bias_first_step
    kernels.push_back(clCreateKernel(bias_first_step_program, "compute_db", &err));
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    // Ядро weights_last_step
    kernels.push_back(clCreateKernel(weights_last_step_program, "adam_update_weights", &err));
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    // Ядро bias_last_step
    kernels.push_back(clCreateKernel(bias_last_step_program, "adam_update_bias", &err));
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    // ReLU
    kernels.push_back(clCreateKernel(relu_program, "relu_inplace", &err));
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    // Sigmoid
    kernels.push_back(clCreateKernel(sigmoid_program, "sigmoid_inplace", &err));
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    // Tanh
    kernels.push_back(clCreateKernel(tanh_program, "tanh_inplace", &err));
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    // Производная ReLU
    kernels.push_back(clCreateKernel(relu_deriv_program, "relu_deriv_inplace", &err));
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    // Производная Sigmoid
    kernels.push_back(clCreateKernel(sigmoid_deriv_program, "sigmoid_deriv_inplace", &err));
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    // Производная Tanh
    kernels.push_back(clCreateKernel(tanh_deriv_program, "tanh_deriv_inplace", &err));
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    // Производная MSE
    kernels.push_back(clCreateKernel(mse_deriv_program, "mse_deriv_inplace", &err));
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    // Производная MAE
    kernels.push_back(clCreateKernel(mae_deriv_program, "mae_deriv_inplace", &err));
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    // Производная Softmax
    kernels.push_back(clCreateKernel(softmax_deriv_program, "softmax_deriv_inplace", &err));
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    // Производная BCE
    kernels.push_back(clCreateKernel(bce_deriv_program, "bce_deriv_inplace", &err));
    if (err != CL_SUCCESS) {
        emit finished(false, QString("OpenCL код ошибки: %1")
                                 .arg(err));
        return;
    }

    clReleaseProgram(program_matrix_mult);
    clReleaseProgram(relu_deriv_program);
    clReleaseProgram(softmax_deriv_program);
    clReleaseProgram(sigmoid_deriv_program);
    clReleaseProgram(tanh_deriv_program);
    clReleaseProgram(mae_deriv_program);
    clReleaseProgram(mse_deriv_program);
    clReleaseProgram(vectors_mult_program);
    clReleaseProgram(backprop_linear_program);
    clReleaseProgram(bias_first_step_program);
    clReleaseProgram(weights_first_step_program);
    clReleaseProgram(bias_last_step_program);
    clReleaseProgram(weights_last_step_program);
    clReleaseProgram(relu_program);
    clReleaseProgram(sigmoid_program);
    clReleaseProgram(tanh_program);
    clReleaseProgram(bce_deriv_program);

    emit finished(true, "");
}
