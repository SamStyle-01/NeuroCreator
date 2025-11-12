#include "samsystem.h"
#include "forwardpass.h"
#include "backward.h"
#include "samtraining.h"
#include "samtest.h"
#include <math.h>

SamSystem::SamSystem(SamView* main_window) {
    this->data = new DataFrame(main_window);
    this->model = new SamModel(main_window, this);
    this->main_window = main_window;
    this->is_standartized = false;
    this->is_inited = false;
    this->curr_epochs = 0;

    this->ocl_inited = true;

    if(!ocl_init()) {
        this->ocl_inited = false;
        QMessageBox::warning(this->main_window, "Ошибка", "Не удалось инициализировать драйвер OpenCL");
    }

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

SamSystem::~SamSystem() {
    delete data;
    delete model;
}

bool SamSystem::backpropagation() {
    QThread* thread = new QThread();

    BackWard* worker = new BackWard(this);

    worker->moveToThread(thread);

    connect(thread, &QThread::started, worker, [worker](){
        worker->doWork();
    });
    connect(worker, &BackWard::finished, this, [this](bool success, QString log) {
        if (success) QMessageBox::information(this->main_window, "Выполнено", "Обучение выполнено успешно");
        else QMessageBox::warning(this->main_window, "Ошибка", log);
    });
    connect(worker, &BackWard::finished, thread, &QThread::quit);
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

    QMessageBox::critical(nullptr, "OpenCL ошибка", message);
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
    return data->load_data(fileName, true);
}

void SamSystem::set_device(cl_device_id index) {
    this->curr_device = index;
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

    connect(thread, &QThread::started, worker, [worker, fileName, processing_data](){
        worker->doWork(fileName, processing_data);
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

    connect(thread, &QThread::started, worker, [worker, fileName, processing_data](){
        worker->doWork(processing_data);
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
    QVector<float> loss(predicted[0].size(), 0);
    int cols = predicted.size();
    int rows = predicted[0].size();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            loss[i] += pow(true_vals[j][i] - predicted[j][i], 2);
        }
        loss[i] /= (float)cols;
    }
    return DataFrame::get_mean(loss);
}

float SamSystem::MAE_loss(const QVector<QVector<float>>& predicted, const QVector<QVector<float>>& true_vals) {
    QVector<float> loss(predicted[0].size(), 0);
    int cols = predicted.size();
    int rows = predicted[0].size();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            loss[i] += abs(true_vals[j][i] - predicted[j][i]);
        }
        loss[i] /= (float)cols;
    }
    return DataFrame::get_mean(loss);
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
}

void SamSystem::reset_model() {
    model->reset_model();
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
