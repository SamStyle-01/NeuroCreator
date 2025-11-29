#include "samsystem.h"
#include "forwardpass.h"
#include "backprop.h"
#include "samtraining.h"
#include "samtest.h"
#include <math.h>
#include <fstream>

extern QString radio_button_style_disabled;
extern QString radio_button_style;

SamSystem::SamSystem(SamView* main_window) {
    this->data = new DataFrame(main_window);
    this->model = new SamModel(main_window, this);
    this->main_window = main_window;
    this->is_standartized = false;
    this->is_inited = false;
    this->curr_epochs = 0;
    this->training_now = false;

    this->best_loss = INFINITY;
    this->best_epoch = -1;

    this->beta1 = 0.9f;
    this->beta2 = 0.999f;

    this->ocl_inited = true;

    if(!ocl_init()) {
        this->ocl_inited = false;
        QMessageBox::warning(this->main_window, "Ошибка", "Не удалось инициализировать драйвер OpenCL");
    }

    this->first_activation = true;

    // Поиск платформ
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));

    if(platformsCount == 0) {
        this->ocl_inited = false;
        QMessageBox::warning(this->main_window, "Ошибка", "Не удалось найти платформы OpenCL");
    }

    QVector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    // Поиск устройств
    for(unsigned int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        cl_platform_id platform = platforms[platformIndex];

        size_t nameSize;
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &nameSize);
        QVector<char> platformName(nameSize);
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, nameSize, platformName.data(), nullptr);

        // Устройства на платформе
        cl_uint currentDeviceCount = 0;
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &currentDeviceCount);
        QVector<cl_device_id> currentDevices(currentDeviceCount);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, currentDeviceCount, currentDevices.data(), nullptr);

        for(unsigned int deviceIndex = 0; deviceIndex < currentDeviceCount; ++deviceIndex) {
            cl_device_id device = currentDevices[deviceIndex];

            // Получаем имя устройства
            clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &nameSize);
            QVector<char> deviceName(nameSize);
            clGetDeviceInfo(device, CL_DEVICE_NAME, nameSize, deviceName.data(), nullptr);

            this->devices.push_back(qMakePair(device, deviceName.data()));
        }
    }
}

void SamSystem::steal_weights_bias(QVector<float*> best_weights, QVector<float*> best_bias) {
    if (best_bias.size()) {
    }
    auto temp_layers = model->get_layers();
    for (int l = 1; l < temp_layers.size(); l++) {
        int N_l = temp_layers[l]->num_neuros;
        int N_prev = temp_layers[l - 1]->num_neuros;
        for (int w = 0; w < N_l; w++) {
            for (int w2 = 0; w2 < N_prev; w2++) {
                int index = w * N_prev + w2;
                this->best_weights[l - 1][index] = best_weights[l - 1][index];
            }
        }

        for (int b = 0; b < N_l; b++) {
            this->best_bias[l - 1][b] = best_bias[l - 1][b];
        }
    }
}

void SamSystem::set_curr_epochs(int epoch) {
    this->curr_epochs = epoch;
}

void SamSystem::set_best_model() {
    this->model->set_model(this->best_weights, this->best_bias);
}

SamSystem::~SamSystem() {
    delete data;
    delete model;
    if (best_bias.size()) {
        for (int i = 0; i < best_bias.size(); i++) {
            delete[] best_bias[i];
            delete[] best_weights[i];
        }
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
    clReleaseKernel(kernel_bias_first_step);
    clReleaseKernel(kernel_weights_first_step);
    clReleaseKernel(kernel_bias_last_step);
    clReleaseKernel(kernel_weights_last_step);
    clReleaseKernel(kernel_relu);
    clReleaseKernel(kernel_sigmoid);
    clReleaseKernel(kernel_tanh);

    if (!first_activation) {
        clReleaseContext(context);
    }
}

bool SamSystem::backpropagation() {
    QThread* thread = new QThread();

    BackPropagation* worker = new BackPropagation(this);

    worker->moveToThread(thread);

    connect(thread, &QThread::started, worker, [worker, this](){
        worker->doWork(this->context);
    });
    connect(worker, &BackPropagation::finished, this, [this](bool success, QString log) {
        if (success) QMessageBox::information(this->main_window, "Выполнено", "Обучение выполнено успешно");
        else QMessageBox::warning(this->main_window, "Ошибка", log);
        this->training_view->training_done();
    });
    connect(worker, &BackPropagation::epoch_done, this,
            [this](float train_loss, float valid_loss) {
        this->curr_epochs++;
        this->training_view->set_epochs_view(this->curr_epochs);

        if (this->training_view->get_train_share() != 100)
            this->training_view->add_loss(train_loss, valid_loss);
        else
            this->training_view->add_loss(train_loss);
    });

    connect(worker, &BackPropagation::finished, thread, &QThread::quit);
    connect(thread, &QThread::finished, worker, &QObject::deleteLater);
    connect(thread, &QThread::finished, thread, &QObject::deleteLater);
    thread->start();
    return true;
}

bool SamSystem::get_ocl_inited() const {
    return ocl_inited;
}

void reportError(cl_int err, const QString &filename, int line) {
    if (err == CL_SUCCESS)
        return;

    QString message = QString("OpenCL код ошибки: %1\nФайл: %2\nСтрока: %3")
                          .arg(err)
                          .arg(filename)
                          .arg(line);

    qDebug() << message;

    QMessageBox::critical(nullptr, "Ошибка", "Ошибка OpenCL");
}

bool SamSystem::load_data() {
    QString fileName = QFileDialog::getOpenFileName(main_window, "Выберите файл", "", "CSV файлы (*.csv)");
    if (!fileName.isEmpty()) {
        QFile file(fileName);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QMessageBox::warning(main_window, "Ошибка", "Не удалось открыть файл");
            return false;
        }
    }
    else {
        return false;
    }
    bool ok = data->load_data(fileName, true);
    if (ok) data->random_shuffle();
    return ok;
}

void SamSystem::set_device(cl_device_id index) {
    this->curr_device = index;

    if (!first_activation) {
        clReleaseKernel(kernel_matrix_mult);
        clReleaseKernel(kernel_relu_deriv);
        clReleaseKernel(kernel_softmax_deriv);
        clReleaseKernel(kernel_sigmoid_deriv);
        clReleaseKernel(kernel_tanh_deriv);
        clReleaseKernel(kernel_mae_deriv);
        clReleaseKernel(kernel_mse_deriv);
        clReleaseKernel(kernel_vectors_mult);
        clReleaseKernel(kernel_backprop_linear);
        clReleaseKernel(kernel_bias_first_step);
        clReleaseKernel(kernel_weights_first_step);
        clReleaseKernel(kernel_bias_last_step);
        clReleaseKernel(kernel_weights_last_step);
        clReleaseKernel(kernel_relu);
        clReleaseKernel(kernel_sigmoid);
        clReleaseKernel(kernel_tanh);
        clReleaseContext(context);
    }

    cl_int err;
    context = clCreateContext(0, 1, &curr_device, nullptr, nullptr, &err);
    OCL_SAFE_CALL(err);

    // Загрузка исходного кода ядра
    // Умножение матриц
    std::ifstream source_file_matrix_mult("../../MatrixVectorMultiplicationKernelBackWard.cl");
    std::string source_code_matrix_mult(std::istreambuf_iterator<char>(source_file_matrix_mult), (std::istreambuf_iterator<char>()));
    const char *source_str = source_code_matrix_mult.c_str();
    size_t source_len = source_code_matrix_mult.length();

    // Ядро backprop_linear
    std::ifstream source_file_backprop_linear("../../backprop_linear_kernel.cl");
    std::string source_code_backprop_linear(std::istreambuf_iterator<char>(source_file_backprop_linear), (std::istreambuf_iterator<char>()));
    const char *source_str_backprop_linear = source_code_backprop_linear.c_str();
    size_t source_len_backprop_linear = source_code_backprop_linear.length();

    // Ядро vectors_mult
    std::ifstream source_file_vectors_mult("../../vectors_mult.cl");
    std::string source_code_vectors_mult(std::istreambuf_iterator<char>(source_file_vectors_mult), (std::istreambuf_iterator<char>()));
    const char *source_str_vectors_mult = source_code_vectors_mult.c_str();
    size_t source_len_vectors_mult = source_code_vectors_mult.length();

    // Ядро weights_first_step
    std::ifstream source_file_weights_first_step("../../weights_first_step.cl");
    std::string source_code_weights_first_step(std::istreambuf_iterator<char>(source_file_weights_first_step), (std::istreambuf_iterator<char>()));
    const char *source_str_weights_first_step = source_code_weights_first_step.c_str();
    size_t source_len_weights_first_step = source_code_weights_first_step.length();

    // Ядро bias_first_step
    std::ifstream source_file_bias_first_step("../../bias_first_step.cl");
    std::string source_code_bias_first_step(std::istreambuf_iterator<char>(source_file_bias_first_step), (std::istreambuf_iterator<char>()));
    const char *source_str_bias_first_step = source_code_bias_first_step.c_str();
    size_t source_len_bias_first_step = source_code_bias_first_step.length();

    // Ядро weights_last_step
    std::ifstream source_file_weights_last_step("../../weights_last_step.cl");
    std::string source_code_weights_last_step(std::istreambuf_iterator<char>(source_file_weights_last_step), (std::istreambuf_iterator<char>()));
    const char *source_str_weights_last_step = source_code_weights_last_step.c_str();
    size_t source_len_weights_last_step = source_code_weights_last_step.length();

    // Ядро bias_last_step
    std::ifstream source_file_bias_last_step("../../bias_last_step.cl");
    std::string source_code_bias_last_step(std::istreambuf_iterator<char>(source_file_bias_last_step), (std::istreambuf_iterator<char>()));
    const char *source_str_bias_last_step = source_code_bias_last_step.c_str();
    size_t source_len_bias_last_step = source_code_bias_last_step.length();

    // ReLU
    std::ifstream source_file_relu("../../relu_func.cl");
    std::string source_code_relu(std::istreambuf_iterator<char>(source_file_relu), (std::istreambuf_iterator<char>()));
    const char *source_str_relu = source_code_relu.c_str();
    size_t source_len_relu = source_code_relu.length();

    // Sigmoid
    std::ifstream source_sigmoid("../../sigmoid_func.cl");
    std::string source_code_sigmoid(std::istreambuf_iterator<char>(source_sigmoid), (std::istreambuf_iterator<char>()));
    const char *source_str_sigmoid = source_code_sigmoid.c_str();
    size_t source_len_sigmoid = source_code_sigmoid.length();

    // Tanh
    std::ifstream source_tanh("../../tanh_func.cl");
    std::string source_code_tanh(std::istreambuf_iterator<char>(source_tanh), (std::istreambuf_iterator<char>()));
    const char *source_str_tanh = source_code_tanh.c_str();
    size_t source_len_tanh = source_code_tanh.length();

    // Производная ReLU
    std::ifstream source_file_relu_deriv("../../relu_derivative_kernel.cl");
    std::string source_code_relu_deriv(std::istreambuf_iterator<char>(source_file_relu_deriv), (std::istreambuf_iterator<char>()));
    const char *source_str_relu_deriv = source_code_relu_deriv.c_str();
    size_t source_len_relu_deriv = source_code_relu_deriv.length();

    // Произодная Sigmoid
    std::ifstream source_sigmoid_deriv("../../sigmoid_derivative_kernel.cl");
    std::string source_code_sigmoid_deriv(std::istreambuf_iterator<char>(source_sigmoid_deriv), (std::istreambuf_iterator<char>()));
    const char *source_str_sigmoid_deriv = source_code_sigmoid_deriv.c_str();
    size_t source_len_sigmoid_deriv = source_code_sigmoid_deriv.length();

    // Произодная Tanh
    std::ifstream source_tanh_deriv("../../tanh_derivative_kernel.cl");
    std::string source_code_tanh_deriv(std::istreambuf_iterator<char>(source_tanh_deriv), (std::istreambuf_iterator<char>()));
    const char *source_str_tanh_deriv = source_code_tanh_deriv.c_str();
    size_t source_len_tanh_deriv = source_code_tanh_deriv.length();

    // Произодная MSE облегчённая версия
    std::ifstream source_mse_deriv("../../mse_derivative_kernel.cl");
    std::string source_code_mse_deriv(std::istreambuf_iterator<char>(source_mse_deriv), (std::istreambuf_iterator<char>()));
    const char *source_str_mse_deriv = source_code_mse_deriv.c_str();
    size_t source_len_mse_deriv = source_code_mse_deriv.length();

    // Произодная MAE
    std::ifstream source_mae_deriv("../../mae_derivative_kernel.cl");
    std::string source_code_mae_deriv(std::istreambuf_iterator<char>(source_mae_deriv), (std::istreambuf_iterator<char>()));
    const char *source_str_mae_deriv = source_code_mae_deriv.c_str();
    size_t source_len_mae_deriv = source_code_mae_deriv.length();

    // Произодная softmax
    std::ifstream source_softmax_deriv("../../softmax_derivative_kernel.cl");
    std::string source_code_softmax_deriv(std::istreambuf_iterator<char>(source_softmax_deriv), (std::istreambuf_iterator<char>()));
    const char *source_str_softmax_deriv = source_code_softmax_deriv.c_str();
    size_t source_len_softmax_deriv = source_code_softmax_deriv.length();

    // Создание программ и их компиляция
    // Умножение матриц
    cl_program program_matrix_mult = clCreateProgramWithSource(context, 1, &source_str, &source_len, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(program_matrix_mult, 1, &curr_device, nullptr, nullptr, nullptr);

    // Ядро backprop_linear
    cl_program backprop_linear_program = clCreateProgramWithSource(context, 1, &source_str_backprop_linear, &source_len_backprop_linear, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(backprop_linear_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Ядро vectors_mult
    cl_program vectors_mult_program = clCreateProgramWithSource(context, 1, &source_str_vectors_mult, &source_len_vectors_mult, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(vectors_mult_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Ядро weights_first_step
    cl_program weights_first_step_program = clCreateProgramWithSource(context, 1, &source_str_weights_first_step, &source_len_weights_first_step, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(weights_first_step_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Ядро bias_first_step
    cl_program bias_first_step_program = clCreateProgramWithSource(context, 1, &source_str_bias_first_step, &source_len_bias_first_step, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(bias_first_step_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Ядро weights_last_step
    cl_program weights_last_step_program = clCreateProgramWithSource(context, 1, &source_str_weights_last_step, &source_len_weights_last_step, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(weights_last_step_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Ядро bias_last_step
    cl_program bias_last_step_program = clCreateProgramWithSource(context, 1, &source_str_bias_last_step, &source_len_bias_last_step, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(bias_last_step_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // ReLU
    cl_program relu_program = clCreateProgramWithSource(context, 1, &source_str_relu, &source_len_relu, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(relu_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Sigmoid
    cl_program sigmoid_program = clCreateProgramWithSource(context, 1, &source_str_sigmoid, &source_len_sigmoid, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(sigmoid_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Tanh
    cl_program tanh_program = clCreateProgramWithSource(context, 1, &source_str_tanh, &source_len_tanh, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(tanh_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Производная ReLU
    cl_program relu_deriv_program = clCreateProgramWithSource(context, 1, &source_str_relu_deriv, &source_len_relu_deriv, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(relu_deriv_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Произодная Sigmoid
    cl_program sigmoid_deriv_program = clCreateProgramWithSource(context, 1, &source_str_sigmoid_deriv, &source_len_sigmoid_deriv, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(sigmoid_deriv_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Произодная Tanh
    cl_program tanh_deriv_program = clCreateProgramWithSource(context, 1, &source_str_tanh_deriv, &source_len_tanh_deriv, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(tanh_deriv_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Произодная MSE
    cl_program mse_deriv_program = clCreateProgramWithSource(context, 1, &source_str_mse_deriv, &source_len_mse_deriv, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(mse_deriv_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Произодная MAE
    cl_program mae_deriv_program = clCreateProgramWithSource(context, 1, &source_str_mae_deriv, &source_len_mae_deriv, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(mae_deriv_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Произодная softmax
    cl_program softmax_deriv_program = clCreateProgramWithSource(context, 1, &source_str_softmax_deriv, &source_len_softmax_deriv, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(softmax_deriv_program, 1, &curr_device, nullptr, nullptr, nullptr);

    // Создание и настройка ядер
    // Умножение матриц
    kernel_matrix_mult = clCreateKernel(program_matrix_mult, "matrixBatchMulBackward", &err);
    OCL_SAFE_CALL(err);

    // Ядро backprop_linear
    kernel_backprop_linear = clCreateKernel(backprop_linear_program, "backprop_linear", &err);
    OCL_SAFE_CALL(err);

    // Ядро vectors_mult
    kernel_vectors_mult = clCreateKernel(vectors_mult_program, "vector_mult_inplace", &err);
    OCL_SAFE_CALL(err);

    // Ядро weights_first_step
    kernel_weights_first_step = clCreateKernel(weights_first_step_program, "compute_dW", &err);
    OCL_SAFE_CALL(err);

    // Ядро bias_first_step
    kernel_bias_first_step = clCreateKernel(bias_first_step_program, "compute_db", &err);
    OCL_SAFE_CALL(err);

    // Ядро weights_last_step
    kernel_weights_last_step = clCreateKernel(weights_last_step_program, "adam_update_weights", &err);
    OCL_SAFE_CALL(err);

    // Ядро bias_last_step
    kernel_bias_last_step = clCreateKernel(bias_last_step_program, "adam_update_bias", &err);
    OCL_SAFE_CALL(err);

    // ReLU
    kernel_relu = clCreateKernel(relu_program, "relu_inplace", &err);
    OCL_SAFE_CALL(err);

    // Sigmoid
    kernel_sigmoid = clCreateKernel(sigmoid_program, "sigmoid_inplace", &err);
    OCL_SAFE_CALL(err);

    // Tanh
    kernel_tanh = clCreateKernel(tanh_program, "tanh_inplace", &err);
    OCL_SAFE_CALL(err);

    // Производная ReLU
    kernel_relu_deriv = clCreateKernel(relu_deriv_program, "relu_deriv_inplace", &err);
    OCL_SAFE_CALL(err);

    // Производная Sigmoid
    kernel_sigmoid_deriv = clCreateKernel(sigmoid_deriv_program, "sigmoid_deriv_inplace", &err);
    OCL_SAFE_CALL(err);

    // Производная Tanh
    kernel_tanh_deriv = clCreateKernel(tanh_deriv_program, "tanh_deriv_inplace", &err);
    OCL_SAFE_CALL(err);

    // Производная MSE
    kernel_mse_deriv = clCreateKernel(mse_deriv_program, "mse_deriv_inplace", &err);
    OCL_SAFE_CALL(err);

    // Производная MAE
    kernel_mae_deriv = clCreateKernel(mae_deriv_program, "mae_deriv_inplace", &err);
    OCL_SAFE_CALL(err);

    // Производная Softmax
    kernel_softmax_deriv = clCreateKernel(softmax_deriv_program, "softmax_deriv_inplace", &err);
    OCL_SAFE_CALL(err);

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

    first_activation = false;
}

bool SamSystem::process_data() {
    QString fileName = QFileDialog::getOpenFileName(main_window, "Выберите файл", "", "CSV файлы (*.csv)");
    if (!fileName.isEmpty()) {
        QFile file(fileName);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QMessageBox::warning(main_window, "Ошибка", "Не удалось открыть файл");
            return false;
        }
    }
    else {
        return false;
    }

    auto processing_data = new DataFrame(this->main_window);

    if (!processing_data->load_data(fileName, false)) {
        return false;
    }
    auto temp_layers = model->get_layers();
    if (processing_data->get_cols() != temp_layers[0]->num_neuros) {
        QMessageBox::warning(main_window, "Ошибка", "Неверное количество столбцов");
        return false;
    }
    if (is_standartized) {
        processing_data->z_score(temp_layers[0]->num_neuros);
    }

    QThread* thread = new QThread();

    ForwardPass* worker = new ForwardPass(this);

    worker->moveToThread(thread);

    connect(thread, &QThread::started, worker, [worker, fileName, processing_data, this](){
        worker->doWork(fileName, processing_data, context);
    });
    connect(worker, &ForwardPass::finished, this, [this](bool success, QString log) {
        if (success) QMessageBox::information(this->main_window, "Выполнено", "Обработка выполнена успешно");
        else QMessageBox::warning(this->main_window, "Ошибка", log);
    });
    connect(worker, &ForwardPass::finished, thread, &QThread::quit);
    connect(thread, &QThread::finished, worker, &QObject::deleteLater);
    connect(thread, &QThread::finished, thread, &QObject::deleteLater);
    thread->start();

    return true;
}

int SamSystem::get_epochs() const {
    return this->curr_epochs;
}

bool SamSystem::test_data() {
    QString fileName = QFileDialog::getOpenFileName(main_window, "Выберите файл", "", "CSV файлы (*.csv)");
    if (!fileName.isEmpty()) {
        QFile file(fileName);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QMessageBox::warning(main_window, "Ошибка", "Не удалось открыть файл");
            return false;
        }
    }
    else {
        return false;
    }

    auto processing_data = new DataFrame(this->main_window);

    if (!processing_data->load_data(fileName, false)) {
        return false;
    }
    auto temp_layers = model->get_layers();
    if (processing_data->get_cols() != this->data->get_cols()) {
        QMessageBox::warning(main_window, "Ошибка", "Неверное количество столбцов");
        return false;
    }
    if (is_standartized) {
        processing_data->z_score(temp_layers[0]->num_neuros);
    }

    QThread* thread = new QThread();

    SamTest* worker = new SamTest(this);

    worker->moveToThread(thread);

    connect(thread, &QThread::started, worker, [worker, fileName, processing_data, this](){
        worker->doWork(processing_data, true, context);
    });
    connect(worker, &SamTest::finished, this, [this](bool success, QString log, float test) {
        if (success) QMessageBox::information(this->main_window, "Выполнено", "Результат теста: " + QString::number(test));
        else QMessageBox::warning(this->main_window, "Ошибка", log);
    });
    connect(worker, &SamTest::finished, thread, &QThread::quit);
    connect(thread, &QThread::finished, worker, &QObject::deleteLater);
    connect(thread, &QThread::finished, thread, &QObject::deleteLater);
    thread->start();

    return true;
}

void SamSystem::set_training_view(SamTraining* training) {
    this->training_view = training;
}

void SamSystem::ReLU_func(QVector<float>& vector) {
    for (int i = 0; i < vector.size(); i++) {
        vector[i] = (vector[i] >= 0 ? vector[i] : 0);
    }
}

void SamSystem::SoftMax_func(QVector<float>& vector) {
    float sum = 0;
    for (const auto& el : vector) {
        sum += std::exp(el);
    }
    for (int i = 0; i < vector.size(); i++) {
        vector[i] = std::exp(vector[i]) / sum;
    }
}

void SamSystem::Sigmoid_func(QVector<float>& vector) {
    for (int i = 0; i < vector.size(); i++) {
        vector[i] = 1.0f / (1.0f + std::exp(-vector[i]));
    }
}
void SamSystem::Tanh_func(QVector<float>& vector) {
    for (int i = 0; i < vector.size(); i++) {
        vector[i] = std::tanh(vector[i]);
    }
}

float SamSystem::MSE_loss(const QVector<QVector<float>>& predicted, const QVector<QVector<float>>& true_vals) {
    int cols = predicted.size();
    int rows = predicted[0].size();
    float total = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float diff = true_vals[j][i] - predicted[j][i];
            total += diff * diff;
        }
    }
    return total / (cols * rows);
}

float SamSystem::MAE_loss(const QVector<QVector<float>>& predicted, const QVector<QVector<float>>& true_vals) {
    int cols = predicted.size();
    int rows = predicted[0].size();
    float total = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            total += qAbs(true_vals[j][i] - predicted[j][i]);
        }
    }
    return total / (cols * rows);
}


float SamSystem::CrossEntropy_loss(const QVector<QVector<float>>& predicted, const QVector<QVector<float>>& true_vals) {
    QVector<float> loss(predicted[0].size(), 0);
    int cols = predicted.size();
    int rows = predicted[0].size();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            loss[i] -= true_vals[j][i] * log(predicted[j][i] + 1e-8);
        }
        loss[i] /= (float)cols;
    }
    return DataFrame::get_mean(loss);
}

QVector<QPair<cl_device_id, QString>> SamSystem::get_devices() const {
    return devices;
}

QPair<int, int> SamSystem::get_shape_data() const {
    return qMakePair(this->data->get_cols(), this->data->get_rows());
}

void SamSystem::init_model() {
    model->init_model();

    auto btns = this->training_view->get_btns();
    auto temp_funcs = this->model->get_funcs();
    bool soft_max_there = false;
    for (int i = 0; i < temp_funcs.size(); i++) {
        if (temp_funcs[i]->func == "SoftMax") {
            soft_max_there = true;
            break;
        }
    }
    if (!soft_max_there) {
        btns[2]->setEnabled(false);
        btns[2]->setStyleSheet(radio_button_style_disabled);
    }
    else {
        btns[2]->setChecked(true);
        for (int i = 0; i < 2; i++) {
            btns[i]->setEnabled(false);
            btns[i]->setStyleSheet(radio_button_style_disabled);
        }
    }

    auto temp_layers = model->get_layers();
    this->t = 0;
    this->best_loss = INFINITY;
    this->best_epoch = -1;
    this->m_w = QVector<QVector<float>>(model->get_weights_size());
    this->v_w = QVector<QVector<float>>(model->get_weights_size());
    this->m_b = QVector<QVector<float>>(model->get_bias_size());
    this->v_b = QVector<QVector<float>>(model->get_bias_size());
    for (int l = 1; l < temp_layers.size(); l++) {
        int N_l = temp_layers[l]->num_neuros;
        int N_prev = temp_layers[l - 1]->num_neuros;

        m_w[l - 1].resize(N_l * N_prev, 0.0f);
        v_w[l - 1].resize(N_l * N_prev, 0.0f);
        m_b[l - 1].resize(N_l, 0.0f);
        v_b[l - 1].resize(N_l, 0.0f);
    }

    for (int l = 1; l < temp_layers.size(); l++) {
        this->best_weights.push_back(new float[temp_layers[l]->num_neuros * temp_layers[l - 1]->num_neuros]);
        this->best_bias.push_back(new float[temp_layers[l]->num_neuros]);
    }
}

bool SamSystem::get_is_training() const {
    return training_now;
}

void SamSystem::set_is_training(bool val) {
    this->training_now = val;
}

void SamSystem::reset_model() {
    model->reset_model();
    this->curr_epochs = 0;
    this->training_view->set_epochs_view(0);
    this->first_activation = true;

    auto btns = this->training_view->get_btns();
    for (int i = 0; i < 3; i++) {
        btns[i]->setEnabled(true);
        btns[i]->setStyleSheet(radio_button_style);
    }

    this->t = 0;

    m_w.clear();
    m_b.clear();
    v_w.clear();
    v_b.clear();
    for (int i = 0; i < best_weights.size(); i++) {
        delete[] best_weights[i];
        delete[] best_bias[i];
    }
    best_weights.clear();
    best_bias.clear();
    best_epoch = -1;

    this->training_view->reset_series();
    this->training_view->reset_state();
}

bool SamSystem::z_score(int num_x) {
    if (is_standartized) {
        QMessageBox::warning(main_window, "Ошибка", "Значения уже были стандартизованы");
        return false;
    }
    is_standartized = true;
    return data->z_score(num_x);
}

void SamSystem::reset_data() {
    delete data;
    data = new DataFrame(this->main_window);
    this->is_standartized = false;
}

void SamSystem::reset_standartization() {
    this->is_standartized = false;
}

void SamSystem::modelize() {
    is_inited = !is_inited;
}

bool SamSystem::get_is_inited() const {
    return is_inited;
}

bool SamSystem::data_inited() const {
    return this->data->get_cols();
}

void SamSystem::set_neuros(int num, int index) {
    model->set_neuros(num, index);
}

bool SamSystem::add_layer(Layer* layer) {
    return model->add_layer(layer);
}

bool SamSystem::add_layer(Layer* layer, int index) {
    return model->add_layer(layer, index);
}

bool SamSystem::add_func(ActivationFunction* func) {
    return model->add_func(func);
}

void SamSystem::remove_layer(int index) {
    model->remove_layer(index);
}

void SamSystem::remove_func(int num_layer) {
    model->remove_func(num_layer);
}

QVector<Layer*> SamSystem::get_layers() const {
    return model->get_layers();
}

QVector<ActivationFunction*> SamSystem::get_funcs() const {
    return model->get_funcs();
}

int SamSystem::get_best_epoch() const {
    return best_epoch;
}
