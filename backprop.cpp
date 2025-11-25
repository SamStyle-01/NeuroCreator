#include "backprop.h"
#include "samsystem.h"
#include "dataframe.h"
#include "samsystem.h"
#include "samtraining.h"
#include "samtest.h"
#include <fstream>

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

    // Загрузка исходного кода ядра
    // Умножение матриц
    std::ifstream source_file_matrix_mult("../../MatrixVectorMultiplicationKernelBackWard.cl");
    std::string source_code_matrix_mult(std::istreambuf_iterator<char>(source_file_matrix_mult), (std::istreambuf_iterator<char>()));
    if(source_code_matrix_mult.empty()) {
        emit finished(false, "Не удалось считать файл ядра");
        return;
    }
    const char *source_str = source_code_matrix_mult.c_str();
    size_t source_len = source_code_matrix_mult.length();

    // Ядро backprop_linear
    std::ifstream source_file_backprop_linear("../../backprop_linear_kernel.cl");
    std::string source_code_backprop_linear(std::istreambuf_iterator<char>(source_file_backprop_linear), (std::istreambuf_iterator<char>()));
    if(source_code_backprop_linear.empty()) {
        emit finished(false, "Не удалось считать файл ядра");
        return;
    }
    const char *source_str_backprop_linear = source_code_backprop_linear.c_str();
    size_t source_len_backprop_linear = source_code_backprop_linear.length();

    // Ядро vectors_mult
    std::ifstream source_file_vectors_mult("../../vectors_mult.cl");
    std::string source_code_vectors_mult(std::istreambuf_iterator<char>(source_file_vectors_mult), (std::istreambuf_iterator<char>()));
    if(source_code_vectors_mult.empty()) {
        emit finished(false, "Не удалось считать файл ядра");
        return;
    }
    const char *source_str_vectors_mult = source_code_vectors_mult.c_str();
    size_t source_len_vectors_mult = source_code_vectors_mult.length();

    // Ядро weights_first_step
    std::ifstream source_file_weights_first_step("../../weights_first_step.cl");
    std::string source_code_weights_first_step(std::istreambuf_iterator<char>(source_file_weights_first_step), (std::istreambuf_iterator<char>()));
    if(source_code_weights_first_step.empty()) {
        emit finished(false, "Не удалось считать файл ядра");
        return;
    }
    const char *source_str_weights_first_step = source_code_weights_first_step.c_str();
    size_t source_len_weights_first_step = source_code_weights_first_step.length();

    // Ядро bias_first_step
    std::ifstream source_file_bias_first_step("../../bias_first_step.cl");
    std::string source_code_bias_first_step(std::istreambuf_iterator<char>(source_file_bias_first_step), (std::istreambuf_iterator<char>()));
    if(source_code_bias_first_step.empty()) {
        emit finished(false, "Не удалось считать файл ядра");
        return;
    }
    const char *source_str_bias_first_step = source_code_bias_first_step.c_str();
    size_t source_len_bias_first_step = source_code_bias_first_step.length();

    // Ядро weights_last_step
    std::ifstream source_file_weights_last_step("../../weights_last_step.cl");
    std::string source_code_weights_last_step(std::istreambuf_iterator<char>(source_file_weights_last_step), (std::istreambuf_iterator<char>()));
    if(source_code_weights_last_step.empty()) {
        emit finished(false, "Не удалось считать файл ядра");
        return;
    }
    const char *source_str_weights_last_step = source_code_weights_last_step.c_str();
    size_t source_len_weights_last_step = source_code_weights_last_step.length();

    // Ядро bias_last_step
    std::ifstream source_file_bias_last_step("../../bias_last_step.cl");
    std::string source_code_bias_last_step(std::istreambuf_iterator<char>(source_file_bias_last_step), (std::istreambuf_iterator<char>()));
    if(source_code_bias_last_step.empty()) {
        emit finished(false, "Не удалось считать файл ядра");
        return;
    }
    const char *source_str_bias_last_step = source_code_bias_last_step.c_str();
    size_t source_len_bias_last_step = source_code_bias_last_step.length();

    // Производная ReLU
    std::ifstream source_file_relu_deriv("../../relu_derivative_kernel.cl");
    std::string source_code_relu_deriv(std::istreambuf_iterator<char>(source_file_relu_deriv), (std::istreambuf_iterator<char>()));
    if(source_code_relu_deriv.empty()) {
        emit finished(false, "Не удалось считать файл ядра");
        return;
    }
    const char *source_str_relu_deriv = source_code_relu_deriv.c_str();
    size_t source_len_relu_deriv = source_code_relu_deriv.length();

    // Произодная Sigmoid
    std::ifstream source_sigmoid_deriv("../../sigmoid_derivative_kernel.cl");
    std::string source_code_sigmoid_deriv(std::istreambuf_iterator<char>(source_sigmoid_deriv), (std::istreambuf_iterator<char>()));
    if(source_code_sigmoid_deriv.empty()) {
        emit finished(false, "Не удалось считать файл ядра");
        return;
    }
    const char *source_str_sigmoid_deriv = source_code_sigmoid_deriv.c_str();
    size_t source_len_sigmoid_deriv = source_code_sigmoid_deriv.length();

    // Произодная Tanh
    std::ifstream source_tanh_deriv("../../tanh_derivative_kernel.cl");
    std::string source_code_tanh_deriv(std::istreambuf_iterator<char>(source_tanh_deriv), (std::istreambuf_iterator<char>()));
    if(source_code_tanh_deriv.empty()) {
        emit finished(false, "Не удалось считать файл ядра");
        return;
    }
    const char *source_str_tanh_deriv = source_code_tanh_deriv.c_str();
    size_t source_len_tanh_deriv = source_code_tanh_deriv.length();

    // Производная ReLU облегчённая версия
    std::ifstream source_file_relu_deriv_simple("../../relu_derivative_kernel_simple.cl");
    std::string source_code_relu_deriv_simple(std::istreambuf_iterator<char>(source_file_relu_deriv_simple), (std::istreambuf_iterator<char>()));
    if(source_code_relu_deriv_simple.empty()) {
        emit finished(false, "Не удалось считать файл ядра");
        return;
    }
    const char *source_str_relu_deriv_simple = source_code_relu_deriv_simple.c_str();
    size_t source_len_relu_deriv_simple = source_code_relu_deriv_simple.length();

    // Произодная Sigmoid облегчённая версия
    std::ifstream source_sigmoid_deriv_simple("../../sigmoid_derivative_kernel_simple.cl");
    std::string source_code_sigmoid_deriv_simple(std::istreambuf_iterator<char>(source_sigmoid_deriv_simple), (std::istreambuf_iterator<char>()));
    if(source_code_sigmoid_deriv_simple.empty()) {
        emit finished(false, "Не удалось считать файл ядра");
        return;
    }
    const char *source_str_sigmoid_deriv_simple = source_code_sigmoid_deriv_simple.c_str();
    size_t source_len_sigmoid_deriv_simple = source_code_sigmoid_deriv_simple.length();

    // Произодная Tanh облегчённая версия
    std::ifstream source_tanh_deriv_simple("../../tanh_derivative_kernel_simple.cl");
    std::string source_code_tanh_deriv_simple(std::istreambuf_iterator<char>(source_tanh_deriv_simple), (std::istreambuf_iterator<char>()));
    if(source_code_tanh_deriv_simple.empty()) {
        emit finished(false, "Не удалось считать файл ядра");
        return;
    }
    const char *source_str_tanh_deriv_simple = source_code_tanh_deriv_simple.c_str();
    size_t source_len_tanh_deriv_simple = source_code_tanh_deriv_simple.length();

    // Произодная MSE облегчённая версия
    std::ifstream source_mse_deriv("../../mse_derivative_kernel.cl");
    std::string source_code_mse_deriv(std::istreambuf_iterator<char>(source_mse_deriv), (std::istreambuf_iterator<char>()));
    if(source_code_mse_deriv.empty()) {
        emit finished(false, "Не удалось считать файл ядра");
        return;
    }
    const char *source_str_mse_deriv = source_code_mse_deriv.c_str();
    size_t source_len_mse_deriv = source_code_mse_deriv.length();

    // Произодная MAE
    std::ifstream source_mae_deriv("../../mae_derivative_kernel.cl");
    std::string source_code_mae_deriv(std::istreambuf_iterator<char>(source_mae_deriv), (std::istreambuf_iterator<char>()));
    if(source_code_mae_deriv.empty()) {
        emit finished(false, "Не удалось считать файл ядра");
        return;
    }
    const char *source_str_mae_deriv = source_code_mae_deriv.c_str();
    size_t source_len_mae_deriv = source_code_mae_deriv.length();

    // Произодная softmax
    std::ifstream source_softmax_deriv("../../softmax_derivative_kernel.cl");
    std::string source_code_softmax_deriv(std::istreambuf_iterator<char>(source_softmax_deriv), (std::istreambuf_iterator<char>()));
    if(source_code_softmax_deriv.empty()) {
        emit finished(false, "Не удалось считать файл ядра");
        return;
    }
    const char *source_str_softmax_deriv = source_code_softmax_deriv.c_str();
    size_t source_len_softmax_deriv = source_code_softmax_deriv.length();

    // Создание программ и их компиляция
    // Умножение матриц
    cl_program program_matrix_mult = clCreateProgramWithSource(context, 1, &source_str, &source_len, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(program_matrix_mult, 1, &system->curr_device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program_matrix_mult, system->curr_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        QVector<char> build_log(log_size);
        clGetProgramBuildInfo(program_matrix_mult, system->curr_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        OCL_SAFE_CALL(err);
        emit finished(false, "Ошибка компиляции ядра");
        return;
    }

    // Ядро backprop_linear
    cl_program backprop_linear_program = clCreateProgramWithSource(context, 1, &source_str_backprop_linear, &source_len_backprop_linear, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(backprop_linear_program, 1, &system->curr_device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(backprop_linear_program, system->curr_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        QVector<char> build_log(log_size);
        clGetProgramBuildInfo(backprop_linear_program, system->curr_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        OCL_SAFE_CALL(err);
        emit finished(false, "Ошибка компиляции ядра");
        return;
    }

    // Ядро vectors_mult
    cl_program vectors_mult_program = clCreateProgramWithSource(context, 1, &source_str_vectors_mult, &source_len_vectors_mult, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(vectors_mult_program, 1, &system->curr_device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(vectors_mult_program, system->curr_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        QVector<char> build_log(log_size);
        clGetProgramBuildInfo(vectors_mult_program, system->curr_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        OCL_SAFE_CALL(err);
        emit finished(false, "Ошибка компиляции ядра");
        return;
    }

    // Ядро weights_first_step
    cl_program weights_first_step_program = clCreateProgramWithSource(context, 1, &source_str_weights_first_step, &source_len_weights_first_step, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(weights_first_step_program, 1, &system->curr_device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(weights_first_step_program, system->curr_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        QVector<char> build_log(log_size);
        clGetProgramBuildInfo(weights_first_step_program, system->curr_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        OCL_SAFE_CALL(err);
        emit finished(false, "Ошибка компиляции ядра");
        return;
    }

    // Ядро bias_first_step
    cl_program bias_first_step_program = clCreateProgramWithSource(context, 1, &source_str_bias_first_step, &source_len_bias_first_step, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(bias_first_step_program, 1, &system->curr_device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(bias_first_step_program, system->curr_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        QVector<char> build_log(log_size);
        clGetProgramBuildInfo(bias_first_step_program, system->curr_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        OCL_SAFE_CALL(err);
        emit finished(false, "Ошибка компиляции ядра");
        return;
    }

    // Ядро weights_last_step
    cl_program weights_last_step_program = clCreateProgramWithSource(context, 1, &source_str_weights_last_step, &source_len_weights_last_step, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(weights_last_step_program, 1, &system->curr_device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(weights_last_step_program, system->curr_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        QVector<char> build_log(log_size);
        clGetProgramBuildInfo(weights_last_step_program, system->curr_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        OCL_SAFE_CALL(err);
        emit finished(false, "Ошибка компиляции ядра");
        return;
    }

    // Ядро bias_last_step
    cl_program bias_last_step_program = clCreateProgramWithSource(context, 1, &source_str_bias_last_step, &source_len_bias_last_step, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(bias_last_step_program, 1, &system->curr_device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(bias_last_step_program, system->curr_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        QVector<char> build_log(log_size);
        clGetProgramBuildInfo(bias_last_step_program, system->curr_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        OCL_SAFE_CALL(err);
        emit finished(false, "Ошибка компиляции ядра");
        return;
    }

    // Производная ReLU
    cl_program relu_deriv_program = clCreateProgramWithSource(context, 1, &source_str_relu_deriv, &source_len_relu_deriv, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(relu_deriv_program, 1, &system->curr_device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(relu_deriv_program, system->curr_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        QVector<char> build_log(log_size);
        clGetProgramBuildInfo(relu_deriv_program, system->curr_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        OCL_SAFE_CALL(err);
        emit finished(false, "Ошибка компиляции ядра");
        return;
    }

    // Произодная Sigmoid
    cl_program sigmoid_deriv_program = clCreateProgramWithSource(context, 1, &source_str_sigmoid_deriv, &source_len_sigmoid_deriv, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(sigmoid_deriv_program, 1, &system->curr_device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(sigmoid_deriv_program, system->curr_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        QVector<char> build_log(log_size);
        clGetProgramBuildInfo(sigmoid_deriv_program, system->curr_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        OCL_SAFE_CALL(err);
        emit finished(false, "Ошибка компиляции ядра");
        return;
    }

    // Произодная Tanh
    cl_program tanh_deriv_program = clCreateProgramWithSource(context, 1, &source_str_tanh_deriv, &source_len_tanh_deriv, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(tanh_deriv_program, 1, &system->curr_device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(tanh_deriv_program, system->curr_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        QVector<char> build_log(log_size);
        clGetProgramBuildInfo(tanh_deriv_program, system->curr_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        OCL_SAFE_CALL(err);
        emit finished(false, "Ошибка компиляции ядра");
        return;
    }

    // Производная ReLU облегчённая версия
    cl_program relu_deriv_simple_program = clCreateProgramWithSource(context, 1, &source_str_relu_deriv_simple, &source_len_relu_deriv_simple, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(relu_deriv_simple_program, 1, &system->curr_device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(relu_deriv_simple_program, system->curr_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        QVector<char> build_log(log_size);
        clGetProgramBuildInfo(relu_deriv_simple_program, system->curr_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        OCL_SAFE_CALL(err);
        emit finished(false, "Ошибка компиляции ядра");
        return;
    }

    // Произодная Sigmoid облегчённая версия
    cl_program sigmoid_deriv_simple_program = clCreateProgramWithSource(context, 1, &source_str_sigmoid_deriv_simple, &source_len_sigmoid_deriv_simple, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(sigmoid_deriv_simple_program, 1, &system->curr_device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(sigmoid_deriv_simple_program, system->curr_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        QVector<char> build_log(log_size);
        clGetProgramBuildInfo(sigmoid_deriv_simple_program, system->curr_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        OCL_SAFE_CALL(err);
        emit finished(false, "Ошибка компиляции ядра");
        return;
    }

    // Произодная Tanh облегчённая версия
    cl_program tanh_deriv_simple_program = clCreateProgramWithSource(context, 1, &source_str_tanh_deriv_simple, &source_len_tanh_deriv_simple, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(tanh_deriv_simple_program, 1, &system->curr_device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(tanh_deriv_simple_program, system->curr_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        QVector<char> build_log(log_size);
        clGetProgramBuildInfo(tanh_deriv_simple_program, system->curr_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        OCL_SAFE_CALL(err);
        emit finished(false, "Ошибка компиляции ядра");
        return;
    }

    // Произодная MSE
    cl_program mse_deriv_program = clCreateProgramWithSource(context, 1, &source_str_mse_deriv, &source_len_mse_deriv, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(mse_deriv_program, 1, &system->curr_device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(mse_deriv_program, system->curr_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        QVector<char> build_log(log_size);
        clGetProgramBuildInfo(mse_deriv_program, system->curr_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        OCL_SAFE_CALL(err);
        emit finished(false, "Ошибка компиляции ядра");
        return;
    }

    // Произодная MAE
    cl_program mae_deriv_program = clCreateProgramWithSource(context, 1, &source_str_mae_deriv, &source_len_mae_deriv, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(mae_deriv_program, 1, &system->curr_device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(mae_deriv_program, system->curr_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        QVector<char> build_log(log_size);
        clGetProgramBuildInfo(mae_deriv_program, system->curr_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        OCL_SAFE_CALL(err);
        emit finished(false, "Ошибка компиляции ядра");
        return;
    }

    // Произодная softmax
    cl_program softmax_deriv_program = clCreateProgramWithSource(context, 1, &source_str_softmax_deriv, &source_len_softmax_deriv, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(softmax_deriv_program, 1, &system->curr_device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(softmax_deriv_program, system->curr_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        QVector<char> build_log(log_size);
        clGetProgramBuildInfo(softmax_deriv_program, system->curr_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        OCL_SAFE_CALL(err);
        emit finished(false, "Ошибка компиляции ядра");
        return;
    }

    // Создание и настройка ядер
    // Умножение матриц
    cl_kernel kernel_matrix_mult = clCreateKernel(program_matrix_mult, "matrixBatchMul", &err);
    OCL_SAFE_CALL(err);

    // Ядро backprop_linear
    cl_kernel kernel_backprop_linear = clCreateKernel(backprop_linear_program, "backprop_linear", &err);
    OCL_SAFE_CALL(err);

    // Ядро vectors_mult
    cl_kernel kernel_vectors_mult = clCreateKernel(vectors_mult_program, "vector_mult_inplace", &err);
    OCL_SAFE_CALL(err);

    // Ядро weights_first_step
    cl_kernel kernel_weights_first_step = clCreateKernel(weights_first_step_program, "compute_dW", &err);
    OCL_SAFE_CALL(err);

    // Ядро bias_first_step
    cl_kernel kernel_bias_first_step = clCreateKernel(bias_first_step_program, "compute_db", &err);
    OCL_SAFE_CALL(err);

    // Ядро weights_last_step
    cl_kernel kernel_weights_last_step = clCreateKernel(weights_last_step_program, "adam_update_weights", &err);
    OCL_SAFE_CALL(err);

    // Ядро bias_last_step
    cl_kernel kernel_bias_last_step = clCreateKernel(bias_last_step_program, "adam_update_bias", &err);
    OCL_SAFE_CALL(err);

    // Производная ReLU
    cl_kernel kernel_relu_deriv = clCreateKernel(relu_deriv_program, "relu_deriv_inplace", &err);
    OCL_SAFE_CALL(err);

    // Производная Sigmoid
    cl_kernel kernel_sigmoid_deriv = clCreateKernel(sigmoid_deriv_program, "sigmoid_deriv_inplace", &err);
    OCL_SAFE_CALL(err);

    // Производная Tanh
    cl_kernel kernel_tanh_deriv = clCreateKernel(tanh_deriv_program, "tanh_deriv_inplace", &err);
    OCL_SAFE_CALL(err);

    // Производная ReLU облегчённая версия
    cl_kernel kernel_relu_deriv_simple = clCreateKernel(relu_deriv_simple_program, "relu_deriv_simple_inplace", &err);
    OCL_SAFE_CALL(err);

    // Производная Sigmoid облегчённая версия
    cl_kernel kernel_sigmoid_deriv_simple = clCreateKernel(sigmoid_deriv_simple_program, "sigmoid_deriv_simple_inplace", &err);
    OCL_SAFE_CALL(err);

    // Производная Tanh облегчённая версия
    cl_kernel kernel_tanh_deriv_simple = clCreateKernel(tanh_deriv_simple_program, "tanh_deriv_simple_inplace", &err);
    OCL_SAFE_CALL(err);

    // Производная MSE
    cl_kernel kernel_mse_deriv = clCreateKernel(mse_deriv_program, "mse_deriv_inplace", &err);
    OCL_SAFE_CALL(err);

    // Производная MAE
    cl_kernel kernel_mae_deriv = clCreateKernel(mae_deriv_program, "mae_deriv_inplace", &err);
    OCL_SAFE_CALL(err);

    // Производная Softmax
    cl_kernel kernel_softmax_deriv = clCreateKernel(softmax_deriv_program, "softmax_deriv_inplace", &err);
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
            QVector<cl_mem> activations_derived;
            pre_activations.push_back(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                     input_vector.size() * sizeof(float), input_vector.data(), &err));
            activations.push_back(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                 input_vector.size() * sizeof(float), input_vector.data(), &err));
            activations_derived.push_back(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                 input_vector.size() * sizeof(float), input_vector.data(), &err));
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

                OCL_SAFE_CALL(clSetKernelArg(kernel_matrix_mult, 0, sizeof(cl_mem), &cl_result_vector));
                OCL_SAFE_CALL(clSetKernelArg(kernel_matrix_mult, 1, sizeof(cl_mem), &cl_vector_B));
                OCL_SAFE_CALL(clSetKernelArg(kernel_matrix_mult, 2, sizeof(cl_mem), &cl_matrix_A));
                OCL_SAFE_CALL(clSetKernelArg(kernel_matrix_mult, 3, sizeof(cl_mem), &cl_bias));
                OCL_SAFE_CALL(clSetKernelArg(kernel_matrix_mult, 4, sizeof(cl_int), &size_batch));
                OCL_SAFE_CALL(clSetKernelArg(kernel_matrix_mult, 5, sizeof(cl_int), &temp_layers[c]->num_neuros));
                OCL_SAFE_CALL(clSetKernelArg(kernel_matrix_mult, 6, sizeof(cl_int), &temp_layers[c + 1]->num_neuros));

                // Запуск ядра
                size_t global_work_size[] = { (size_t)size_batch, (size_t)temp_layers[c + 1]->num_neuros };

                err = clEnqueueNDRangeKernel(queue, kernel_matrix_mult, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
                clFinish(queue);

                // Чтение результата
                err = clEnqueueReadBuffer(queue, cl_result_vector, CL_TRUE, 0, size_R, result_vector.data(), 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                pre_activations.push_back(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                         result_vector.size() * sizeof(float), result_vector.data(), &err));

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

                activations.push_back(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                     result_vector.size() * sizeof(float), result_vector.data(), &err));
                activations_derived.push_back(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                     result_vector.size() * sizeof(float), result_vector.data(), &err));
                input_vector = result_vector;

                // Очистка ресурсов
                clReleaseMemObject(cl_matrix_A);
                clReleaseMemObject(cl_vector_B);
                clReleaseMemObject(cl_result_vector);
            }

            // Обратное распространение ошибки
            QVector<float> true_vals;
            true_vals.reserve(final_layer_size * size_batch);
            for (int k1 = train_cols; k1 < data.size(); k1++) {
                for (int n1 = i; n1 < size_batch + i; n1++) {
                    true_vals.push_back(data[k1][n1]);
                }
            }

            auto loss_func = this->system->training_view->get_loss_func();
            int size_A_int = input_vector.size();
            size_t size_A = size_A_int * sizeof(float);
            cl_mem cl_delta_vector = clCreateBuffer(context, CL_MEM_READ_WRITE, size_A, nullptr, &err);
            OCL_SAFE_CALL(err);
            switch (loss_func) {
                case LossFunc::MSE: {
                    cl_mem cl_matrix_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_A, input_vector.data(), &err);
                    OCL_SAFE_CALL(err);
                    cl_mem cl_matrix_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_A, true_vals.data(), &err);
                    OCL_SAFE_CALL(err);
                    OCL_SAFE_CALL(clSetKernelArg(kernel_mse_deriv, 0, sizeof(cl_mem), &cl_matrix_A));
                    OCL_SAFE_CALL(clSetKernelArg(kernel_mse_deriv, 1, sizeof(cl_mem), &cl_matrix_B));
                    OCL_SAFE_CALL(clSetKernelArg(kernel_mse_deriv, 2, sizeof(cl_mem), &cl_delta_vector));
                    OCL_SAFE_CALL(clSetKernelArg(kernel_mse_deriv, 3, sizeof(cl_int), &size_A_int));

                    size_t global_work_size[] = { (size_t)size_A_int };

                    err = clEnqueueNDRangeKernel(queue, kernel_mse_deriv, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);

                    clReleaseMemObject(cl_matrix_A);
                    clReleaseMemObject(cl_matrix_B);
                    break;
                }
                case LossFunc::MAE: {
                    cl_mem cl_matrix_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_A, input_vector.data(), &err);
                    OCL_SAFE_CALL(err);
                    cl_mem cl_matrix_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_A, true_vals.data(), &err);
                    OCL_SAFE_CALL(err);
                    OCL_SAFE_CALL(clSetKernelArg(kernel_mae_deriv, 0, sizeof(cl_mem), &cl_matrix_A));
                    OCL_SAFE_CALL(clSetKernelArg(kernel_mae_deriv, 1, sizeof(cl_mem), &cl_matrix_B));
                    OCL_SAFE_CALL(clSetKernelArg(kernel_mae_deriv, 2, sizeof(cl_mem), &cl_delta_vector));
                    OCL_SAFE_CALL(clSetKernelArg(kernel_mae_deriv, 3, sizeof(cl_int), &size_A_int));

                    size_t global_work_size[] = { (size_t)size_A_int };

                    err = clEnqueueNDRangeKernel(queue, kernel_mae_deriv, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);

                    clReleaseMemObject(cl_matrix_A);
                    clReleaseMemObject(cl_matrix_B);
                    break;
                }
                default: {
                    break;
                }
            }

            if (activations_layers.back() == Activation::RELU) {
                OCL_SAFE_CALL(clSetKernelArg(kernel_relu_deriv, 0, sizeof(cl_mem), &pre_activations[pre_activations.size() - 1]));
                OCL_SAFE_CALL(clSetKernelArg(kernel_relu_deriv, 1, sizeof(cl_mem), &cl_delta_vector));
                OCL_SAFE_CALL(clSetKernelArg(kernel_relu_deriv, 2, sizeof(cl_int), &size_A_int));

                size_t global_work_size[] = { (size_t)size_A_int };

                err = clEnqueueNDRangeKernel(queue, kernel_relu_deriv, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }
            else if (activations_layers.back() == Activation::SIGMOID) {
                OCL_SAFE_CALL(clSetKernelArg(kernel_sigmoid_deriv, 0, sizeof(cl_mem), &activations[activations.size() - 1]));
                OCL_SAFE_CALL(clSetKernelArg(kernel_sigmoid_deriv, 1, sizeof(cl_mem), &cl_delta_vector));
                OCL_SAFE_CALL(clSetKernelArg(kernel_sigmoid_deriv, 2, sizeof(cl_int), &size_A_int));

                size_t global_work_size[] = { (size_t)size_A_int };

                err = clEnqueueNDRangeKernel(queue, kernel_sigmoid_deriv, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }
            else if (activations_layers.back() == Activation::TANH) {
                OCL_SAFE_CALL(clSetKernelArg(kernel_tanh_deriv, 0, sizeof(cl_mem), &activations[activations.size() - 1]));
                OCL_SAFE_CALL(clSetKernelArg(kernel_tanh_deriv, 1, sizeof(cl_mem), &cl_delta_vector));
                OCL_SAFE_CALL(clSetKernelArg(kernel_tanh_deriv, 2, sizeof(cl_int), &size_A_int));

                size_t global_work_size[] = { (size_t)size_A_int };

                err = clEnqueueNDRangeKernel(queue, kernel_tanh_deriv, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }
            else if (activations_layers.back() == Activation::SOFTMAX) {
                cl_mem cl_matrix_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_A, input_vector.data(), &err);
                OCL_SAFE_CALL(err);
                cl_mem cl_matrix_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_A, true_vals.data(), &err);
                OCL_SAFE_CALL(err);
                OCL_SAFE_CALL(clSetKernelArg(kernel_softmax_deriv, 0, sizeof(cl_mem), &cl_matrix_A));
                OCL_SAFE_CALL(clSetKernelArg(kernel_softmax_deriv, 1, sizeof(cl_mem), &cl_matrix_B));
                OCL_SAFE_CALL(clSetKernelArg(kernel_softmax_deriv, 2, sizeof(cl_mem), &cl_delta_vector));
                OCL_SAFE_CALL(clSetKernelArg(kernel_softmax_deriv, 3, sizeof(cl_int), &size_A_int));

                size_t global_work_size[] = { (size_t)size_A_int };

                err = clEnqueueNDRangeKernel(queue, kernel_softmax_deriv, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                clReleaseMemObject(cl_matrix_A);
                clReleaseMemObject(cl_matrix_B);
            }

            for (int l = 0; l < temp_layers.size() - 1; l++) {
                if (activations_layers[l] == Activation::RELU) {
                    int size = size_batch * temp_layers[l + 1]->num_neuros;
                    OCL_SAFE_CALL(clSetKernelArg(kernel_relu_deriv_simple, 0, sizeof(cl_mem), &pre_activations[l]));
                    OCL_SAFE_CALL(clSetKernelArg(kernel_relu_deriv_simple, 1, sizeof(cl_int), &size));

                    size_t global_work_size[] = { (size_t)temp_layers[l + 1]->num_neuros };

                    err = clEnqueueNDRangeKernel(queue, kernel_relu_deriv_simple, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);
                }
            }

            for (int l = 0; l < temp_layers.size() - 1; l++) {
                if (activations_layers[l] == Activation::SIGMOID) {
                    int size = size_batch * temp_layers[l + 1]->num_neuros;
                    OCL_SAFE_CALL(clSetKernelArg(kernel_sigmoid_deriv_simple, 0, sizeof(cl_mem), &activations_derived[l]));
                    OCL_SAFE_CALL(clSetKernelArg(kernel_sigmoid_deriv_simple, 1, sizeof(cl_int), &size));

                    size_t global_work_size[] = { (size_t)temp_layers[l + 1]->num_neuros };

                    err = clEnqueueNDRangeKernel(queue, kernel_sigmoid_deriv_simple, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);
                }
                else if (activations_layers[l] == Activation::TANH) {
                    int size = size_batch * temp_layers[l + 1]->num_neuros;
                    OCL_SAFE_CALL(clSetKernelArg(kernel_tanh_deriv_simple, 0, sizeof(cl_mem), &activations_derived[l]));
                    OCL_SAFE_CALL(clSetKernelArg(kernel_tanh_deriv_simple, 1, sizeof(cl_int), &size));

                    size_t global_work_size[] = { (size_t)temp_layers[l + 1]->num_neuros };

                    err = clEnqueueNDRangeKernel(queue, kernel_tanh_deriv_simple, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);
                }
            }

            QVector<cl_mem> hidden_delta(temp_layers.size());
            hidden_delta[temp_layers.size() - 1] = cl_delta_vector;

            for (int l = temp_layers.size() - 2; l >= 0; l--) {
                hidden_delta[l] = clCreateBuffer(context, CL_MEM_READ_WRITE, size_batch * temp_layers[l]->num_neuros * sizeof(float), nullptr, &err);

                OCL_SAFE_CALL(clSetKernelArg(kernel_backprop_linear, 0, sizeof(cl_mem), &hidden_delta[l + 1]));
                OCL_SAFE_CALL(clSetKernelArg(kernel_backprop_linear, 1, sizeof(cl_mem), &weights[l]));
                OCL_SAFE_CALL(clSetKernelArg(kernel_backprop_linear, 2, sizeof(cl_mem), &hidden_delta[l]));
                OCL_SAFE_CALL(clSetKernelArg(kernel_backprop_linear, 3, sizeof(cl_int), &size_batch));
                OCL_SAFE_CALL(clSetKernelArg(kernel_backprop_linear, 4, sizeof(cl_int), &temp_layers[l + 1]->num_neuros));
                OCL_SAFE_CALL(clSetKernelArg(kernel_backprop_linear, 5, sizeof(cl_int), &temp_layers[l]->num_neuros));

                size_t global_work_size[] = { (size_t)size_batch, (size_t)temp_layers[l]->num_neuros };

                err = clEnqueueNDRangeKernel(queue, kernel_backprop_linear, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                if (activations_layers[l] == Activation::RELU) {
                    OCL_SAFE_CALL(clSetKernelArg(kernel_vectors_mult, 0, sizeof(cl_mem), &hidden_delta[l]));
                    OCL_SAFE_CALL(clSetKernelArg(kernel_vectors_mult, 1, sizeof(cl_mem), &pre_activations[l]));
                    OCL_SAFE_CALL(clSetKernelArg(kernel_vectors_mult, 2, sizeof(cl_int), &size_A_int));

                    size_t global_work_size[] = { (size_t)size_batch };

                    err = clEnqueueNDRangeKernel(queue, kernel_vectors_mult, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);
                }
                else if (activations_layers[l] == Activation::SIGMOID || activations_layers[l] == Activation::TANH) {
                    OCL_SAFE_CALL(clSetKernelArg(kernel_vectors_mult, 0, sizeof(cl_mem), &hidden_delta[l]));
                    OCL_SAFE_CALL(clSetKernelArg(kernel_vectors_mult, 1, sizeof(cl_mem), &activations_derived[l]));
                    OCL_SAFE_CALL(clSetKernelArg(kernel_vectors_mult, 2, sizeof(cl_int), &size_A_int));

                    size_t global_work_size[] = { (size_t)size_batch };

                    err = clEnqueueNDRangeKernel(queue, kernel_vectors_mult, 1, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                    OCL_SAFE_CALL(err);
                }
            }

            QVector<cl_mem> db(temp_layers.size());
            QVector<cl_mem> dW(temp_layers.size());

            for (int l = 1; l < temp_layers.size(); l++) {
                const int N_l = temp_layers[l]->num_neuros;
                const int N_prev = temp_layers[l - 1]->num_neuros;

                // Смещения
                db[l] = clCreateBuffer(context, CL_MEM_READ_WRITE, N_l * sizeof(float), nullptr, &err);

                OCL_SAFE_CALL(clSetKernelArg(kernel_bias_first_step, 0, sizeof(cl_mem), &hidden_delta[l]));
                OCL_SAFE_CALL(clSetKernelArg(kernel_bias_first_step, 1, sizeof(cl_mem), &db[l]));
                OCL_SAFE_CALL(clSetKernelArg(kernel_bias_first_step, 2, sizeof(cl_int), &size_batch));
                OCL_SAFE_CALL(clSetKernelArg(kernel_bias_first_step, 3, sizeof(cl_int), &N_l));

                size_t global_work_size_bias[] = { (size_t)N_l };

                err = clEnqueueNDRangeKernel(queue, kernel_bias_first_step, 1, nullptr, global_work_size_bias, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                // Веса
                dW[l] = clCreateBuffer(context, CL_MEM_READ_WRITE, N_l * N_prev * sizeof(float), nullptr, &err);

                OCL_SAFE_CALL(clSetKernelArg(kernel_weights_first_step, 0, sizeof(cl_mem), &activations[l - 1]));
                OCL_SAFE_CALL(clSetKernelArg(kernel_weights_first_step, 1, sizeof(cl_mem), &hidden_delta[l]));
                OCL_SAFE_CALL(clSetKernelArg(kernel_weights_first_step, 2, sizeof(cl_mem), &dW[l]));
                OCL_SAFE_CALL(clSetKernelArg(kernel_weights_first_step, 3, sizeof(cl_int), &size_batch));
                OCL_SAFE_CALL(clSetKernelArg(kernel_weights_first_step, 4, sizeof(cl_int), &N_prev));
                OCL_SAFE_CALL(clSetKernelArg(kernel_weights_first_step, 5, sizeof(cl_int), &N_l));

                size_t global_work_size[] = { (size_t)size_batch, (size_t)std::max(N_l, N_prev) };

                err = clEnqueueNDRangeKernel(queue, kernel_weights_first_step, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }

            this->system->t++;

            for (int l = 1; l < temp_layers.size(); l++) {
                int N_l = temp_layers[l]->num_neuros;
                int N_prev = temp_layers[l - 1]->num_neuros;
                // Смещения
                size_t size_bias = N_l * sizeof(float);

                OCL_SAFE_CALL(clSetKernelArg(kernel_bias_last_step, 0, sizeof(cl_mem), &bias[l - 1]));
                OCL_SAFE_CALL(clSetKernelArg(kernel_bias_last_step, 1, sizeof(cl_mem), &db[l]));
                OCL_SAFE_CALL(clSetKernelArg(kernel_bias_last_step, 2, sizeof(cl_mem), &m_b[l - 1]));
                OCL_SAFE_CALL(clSetKernelArg(kernel_bias_last_step, 3, sizeof(cl_mem), &v_b[l - 1]));
                OCL_SAFE_CALL(clSetKernelArg(kernel_bias_last_step, 4, sizeof(cl_float), &eta));
                OCL_SAFE_CALL(clSetKernelArg(kernel_bias_last_step, 5, sizeof(cl_float), &this->system->beta1));
                OCL_SAFE_CALL(clSetKernelArg(kernel_bias_last_step, 6, sizeof(cl_float), &this->system->beta2));
                OCL_SAFE_CALL(clSetKernelArg(kernel_bias_last_step, 7, sizeof(cl_int), &this->system->t));

                size_t global_work_size_bias[] = { (size_t)N_l };

                err = clEnqueueNDRangeKernel(queue, kernel_bias_last_step, 1, nullptr, global_work_size_bias, nullptr, 0, nullptr, nullptr);
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

                OCL_SAFE_CALL(clSetKernelArg(kernel_weights_last_step, 0, sizeof(cl_mem), &weights[l - 1]));
                OCL_SAFE_CALL(clSetKernelArg(kernel_weights_last_step, 1, sizeof(cl_mem), &dW[l]));
                OCL_SAFE_CALL(clSetKernelArg(kernel_weights_last_step, 2, sizeof(cl_mem), &m_w[l - 1]));
                OCL_SAFE_CALL(clSetKernelArg(kernel_weights_last_step, 3, sizeof(cl_mem), &v_w[l - 1]));
                OCL_SAFE_CALL(clSetKernelArg(kernel_weights_last_step, 4, sizeof(cl_float), &eta));
                OCL_SAFE_CALL(clSetKernelArg(kernel_weights_last_step, 5, sizeof(cl_float), &this->system->beta1));
                OCL_SAFE_CALL(clSetKernelArg(kernel_weights_last_step, 6, sizeof(cl_float), &this->system->beta2));
                OCL_SAFE_CALL(clSetKernelArg(kernel_weights_last_step, 7, sizeof(cl_int), &this->system->t));
                OCL_SAFE_CALL(clSetKernelArg(kernel_weights_last_step, 8, sizeof(cl_int), &N_prev));

                size_t global_work_size[] = { (size_t)N_l, (size_t)N_prev };

                err = clEnqueueNDRangeKernel(queue, kernel_weights_last_step, 2, nullptr, global_work_size, nullptr, 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                clFinish(queue);
                err = clEnqueueReadBuffer(queue, weights[l - 1], CL_TRUE, 0, size_weights, this->system->model->weights[l - 1], 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                err = clEnqueueReadBuffer(queue, m_w[l - 1], CL_TRUE, 0, size_weights, this->system->m_w[l - 1].data(), 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);

                err = clEnqueueReadBuffer(queue, v_w[l - 1], CL_TRUE, 0, size_weights, this->system->v_w[l - 1].data(), 0, nullptr, nullptr);
                OCL_SAFE_CALL(err);
            }
            this->system->model->update_weights();

            for (int ind = 0; ind < db.size(); ind++) {
                clReleaseMemObject(db[ind]);
                clReleaseMemObject(dW[ind]);
            }

            for (int ind = 0; ind < hidden_delta.size(); ind++) {
                clReleaseMemObject(hidden_delta[ind]);
            }

            for (int activs = 0; activs < activations.size(); activs++) {
                clReleaseMemObject(activations[activs]);
                clReleaseMemObject(activations_derived[activs]);
                clReleaseMemObject(pre_activations[activs]);
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
        qDebug() << this->system->best_epoch << " " << train_loss;
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

    clReleaseKernel(kernel_matrix_mult);
    clReleaseKernel(kernel_relu_deriv);
    clReleaseKernel(kernel_softmax_deriv);
    clReleaseKernel(kernel_sigmoid_deriv);
    clReleaseKernel(kernel_tanh_deriv);
    clReleaseKernel(kernel_mae_deriv);
    clReleaseKernel(kernel_mse_deriv);
    clReleaseKernel(kernel_vectors_mult);
    clReleaseKernel(kernel_backprop_linear);
    clReleaseKernel(kernel_relu_deriv_simple);
    clReleaseKernel(kernel_sigmoid_deriv_simple);
    clReleaseKernel(kernel_tanh_deriv_simple);
    clReleaseKernel(kernel_bias_first_step);
    clReleaseKernel(kernel_weights_first_step);
    clReleaseKernel(kernel_bias_last_step);
    clReleaseKernel(kernel_weights_last_step);

    clReleaseProgram(program_matrix_mult);
    clReleaseProgram(relu_deriv_program);
    clReleaseProgram(softmax_deriv_program);
    clReleaseProgram(sigmoid_deriv_program);
    clReleaseProgram(tanh_deriv_program);
    clReleaseProgram(mae_deriv_program);
    clReleaseProgram(mse_deriv_program);
    clReleaseProgram(vectors_mult_program);
    clReleaseProgram(backprop_linear_program);
    clReleaseProgram(relu_deriv_simple_program);
    clReleaseProgram(sigmoid_deriv_simple_program);
    clReleaseProgram(tanh_deriv_simple_program);
    clReleaseProgram(bias_first_step_program);
    clReleaseProgram(weights_first_step_program);
    clReleaseProgram(bias_last_step_program);
    clReleaseProgram(weights_last_step_program);

    clReleaseCommandQueue(queue);

    delete sharded_data.first;
    delete sharded_data.second;

    this->system->set_is_training(false);

    emit finished(true, "");
}

void BackPropagation::clip_gradients(QVector<float>& grad, float clip_value) {
    for (float& g : grad) {
        if (g > clip_value) g = clip_value;
        else if (g < -clip_value) g = -clip_value;
    }
}

void BackPropagation::ReLU_func_deriv(QVector<float>& vector) {
    for (int i = 0; i < vector.size(); i++) {
        vector[i] = vector[i] > 0 ? 1 : 0;
    }
}

QVector<float> BackPropagation::SoftMax_func_deriv(const QVector<float>& predicted, const QVector<QVector<float>>& target, int outputs,
                                            int col_first, int curr_el) {
    int batch = predicted.size() / outputs;
    QVector<float> grad(outputs * batch);

    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < batch; j++) {
            grad[i * outputs + j] = predicted[i * outputs + j] - target[col_first + i][curr_el + j];
        }
    }

    return grad;
}

void BackPropagation::Sigmoid_func_deriv(QVector<float>& vector) {
    for (int i = 0; i < vector.size(); i++) {
        vector[i] = vector[i] * (1 - vector[i]);
    }
}

void BackPropagation::Tanh_func_deriv(QVector<float>& vector) {
    for (int i = 0; i < vector.size(); i++) {
        vector[i] = 1 - pow(vector[i], 2);
    }
}

QVector<float> BackPropagation::MSE_deriv(const QVector<float>& predicted, const QVector<QVector<float>>& true_vals, int outputs,
                                   int col_first, int curr_el) {
    int batch = predicted.size() / outputs;
    QVector<float> grad(outputs * batch);

    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < batch; j++) {
            grad[i * outputs + j] = predicted[i * outputs + j] - true_vals[col_first + i][curr_el + j];
        }
    }

    return grad;
}

QVector<float> BackPropagation::MAE_deriv(const QVector<float>& predicted, const QVector<QVector<float>>& true_vals, int outputs,
                                   int col_first, int curr_el) {
    int batch = predicted.size() / outputs;
    QVector<float> grad(outputs * batch);

    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < batch; j++) {
            float diff = predicted[i * outputs + j] - true_vals[col_first + i][curr_el + j];
            grad[i * outputs + j] = (diff > 0) ? 1.0f : (diff < 0 ? -1.0f : 0.0f);
        }
    }

    return grad;
}

QVector<float> BackPropagation::CrossEntropy_deriv(const QVector<float>& predicted, const QVector<QVector<float>>& true_vals, int outputs,
                                            int col_first, int curr_el) {
    int batch = predicted.size() / outputs;
    QVector<float> grad(outputs * batch);

    for (int i = 0; i < outputs; i++) {
        for (int j = 0; j < batch; j++) {
            grad[i * outputs + j] = -true_vals[col_first + i][curr_el + j] / (predicted[i * outputs + j] + 1e-8f);
        }
    }

    return grad;
}
