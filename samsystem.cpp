#include "samsystem.h"
#include "forwardpass.h"
#include "backprop.h"
#include "samtraining.h"
#include "samtest.h"
#include "sambuttonsgroup.h"

extern QString radio_button_style_disabled;
extern QString radio_button_style;
extern QString button_style;
extern QString button_disabled;

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

        this->best_m_b = this->m_b;
        this->best_v_b = this->v_b;
        this->best_m_w = this->m_w;
        this->best_v_w = this->v_w;
        this->best_t = this->t;
    }
}

void SamSystem::set_curr_epochs(int epoch) {
    this->curr_epochs = epoch;
}

void SamSystem::set_best_model() {
    this->model->set_model(this->best_weights, this->best_bias);

    this->m_b = this->best_m_b;
    this->v_b = this->best_v_b;
    this->m_w = this->best_m_w;
    this->v_w = this->best_v_w;
    this->t = this->best_t;
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
    clReleaseKernel(kernel_bce_deriv);
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
        this->training_view->set_epochs_view(this->curr_epochs, best_loss);

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
    this->compilation = true;

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
        clReleaseKernel(kernel_bce_deriv);
        clReleaseContext(context);
    }

    cl_int err;
    context = clCreateContext(0, 1, &curr_device, nullptr, nullptr, &err);
    OCL_SAFE_CALL(err);

    QThread* thread = new QThread();

    SamThreadSystem* worker = new SamThreadSystem();

    worker->moveToThread(thread);

    auto buttons = this->scheme->get_devices();
    for (auto& el : buttons) {
        if (dynamic_cast<DeviceButton*>(el)->index != this->curr_device) {
            el->setStyleSheet(button_disabled);
        }
        el->setEnabled(false);
    }

    connect(thread, &QThread::started, worker, [worker, this](){
        worker->set_device(this->curr_device, context, kernels);
    });
    connect(worker, &SamThreadSystem::finished, this, [this](bool success, QString log) {
        if (success) {
            this->kernel_matrix_mult = kernels[0];
            this->kernel_backprop_linear = kernels[1];
            this->kernel_vectors_mult = kernels[2];
            this->kernel_weights_first_step = kernels[3];
            this->kernel_bias_first_step = kernels[4];
            this->kernel_weights_last_step = kernels[5];
            this->kernel_bias_last_step = kernels[6];
            this->kernel_relu = kernels[7];
            this->kernel_sigmoid = kernels[8];
            this->kernel_tanh = kernels[9];
            this->kernel_relu_deriv = kernels[10];
            this->kernel_sigmoid_deriv = kernels[11];
            this->kernel_tanh_deriv = kernels[12];
            this->kernel_mse_deriv = kernels[13];
            this->kernel_mae_deriv = kernels[14];
            this->kernel_softmax_deriv = kernels[15];
            this->kernel_bce_deriv = kernels[16];

            this->kernels.clear();
        }
        else QMessageBox::warning(this->main_window, "Ошибка", log);

        auto buttons = this->scheme->get_devices();
        for (auto& el : buttons) {
            if (dynamic_cast<DeviceButton*>(el)->index != this->curr_device) {
                el->setStyleSheet(button_style);
            }
            el->setEnabled(true);
        }

        this->compilation = false;
    });
    connect(worker, &SamThreadSystem::finished, thread, &QThread::quit);
    connect(thread, &QThread::finished, worker, &QObject::deleteLater);
    connect(thread, &QThread::finished, thread, &QObject::deleteLater);
    thread->start();

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

    if (!processing_data->load_data(fileName, false, this->data->get_cols() - this->model->get_layers().back()->num_neuros)) {
        return false;
    }
    auto temp_layers = model->get_layers();
    if (processing_data->get_cols() != temp_layers[0]->num_neuros) {
        QMessageBox::warning(main_window, "Ошибка", "Неверное количество столбцов");
        return false;
    }
    if (is_standartized) {
        auto pair_nums = this->data->get_mean_std();
        processing_data->z_score(temp_layers[0]->num_neuros, pair_nums.first, pair_nums.second);
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


bool SamSystem::process_data(QString data) {
    auto processing_data = new DataFrame(this->main_window);

    if (!processing_data->load_data(data, this->data->get_cols() - this->model->get_layers().back()->num_neuros)) {
        return false;
    }
    auto temp_layers = model->get_layers();
    if (processing_data->get_cols() != temp_layers[0]->num_neuros) {
        QMessageBox::warning(main_window, "Ошибка", "Неверное количество столбцов");
        return false;
    }
    if (is_standartized) {
        auto pair_nums = this->data->get_mean_std();
        processing_data->z_score(temp_layers[0]->num_neuros, pair_nums.first, pair_nums.second);
    }

    QThread* thread = new QThread();

    ForwardPass* worker = new ForwardPass(this);

    worker->moveToThread(thread);

    connect(thread, &QThread::started, worker, [worker, processing_data, this](){
        worker->doWork(processing_data, context);
    });
    connect(worker, &ForwardPass::finished, this, [this](bool success, QString result) {
        if (success) this->scheme->set_output_field(result);
        else QMessageBox::warning(this->main_window, "Ошибка", result);
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

    if (!processing_data->load_data(fileName, false, this->data->get_cols())) {
        return false;
    }
    auto temp_layers = model->get_layers();
    if (processing_data->get_cols() != this->data->get_cols()) {
        QMessageBox::warning(main_window, "Ошибка", "Неверное количество столбцов");
        return false;
    }
    if (is_standartized) {
        auto pair_nums = data->get_mean_std();
        processing_data->z_score(temp_layers[0]->num_neuros, pair_nums.first, pair_nums.second);
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

void SamSystem::set_view(SamTraining* training) {
    this->training_view = training;
}

void SamSystem::set_view(SamScheme* scheme) {
    this->scheme = scheme;
}

void SamSystem::ReLU_func(QVector<float>& vector) {
    for (int i = 0; i < vector.size(); i++) {
        vector[i] = (vector[i] >= 0 ? vector[i] : 0);
    }
}

void SamSystem::SoftMax_func(QVector<float>::Iterator begin, QVector<float>::Iterator end) {
    float sum = 0;
    for (auto it = begin; it != end; ++it) {
        sum += std::exp(*it);
    }
    for (auto it = begin; it != end; ++it) {
        *it = std::exp(*it) / sum;
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

float SamSystem::BCE_loss(
    const QVector<QVector<float>>& logits,
    const QVector<QVector<float>>& true_vals) {
    QVector<float> loss(logits[0].size(), 0);

    int cols = logits.size();
    int rows = logits[0].size();

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {

            float z = logits[j][i];
            float y = true_vals[j][i];

            float term = std::max(z, 0.0f)
                         - z * y
                         + std::log1p(std::exp(-std::fabs(z)));

            loss[i] += term;
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
    bool bce_there = false;
    int final_layer_neurons = this->get_layers().back()->num_neuros;
    for (int i = 0; i < temp_funcs.size(); i++) {
        if (temp_funcs[i]->func == "SoftMax") {
            soft_max_there = true;
            break;
        }
        else if ((temp_funcs[i]->func == "Sigmoid") && (temp_funcs[i]->num_layer == this->get_layers().size() - 1)
                 && (final_layer_neurons == 1)) {
            bce_there = true;
            break;
        }
    }
    if (!bce_there) {
        btns[3]->setEnabled(false);
        btns[3]->setStyleSheet(radio_button_style_disabled);
    }
    if (!soft_max_there) {
        btns[0]->setChecked(true);
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
    best_t = 0;
    best_m_b.clear();
    best_v_b.clear();
    best_m_w.clear();
    best_v_w.clear();
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

void SamSystem::save_state(QFile& file) const {
    QTextStream out(&file);
    this->model->save_state(out);
    out << this->curr_epochs << "\n";
    out << this->is_standartized << "\n";
    out << this->t << "\n";
    for (const auto& arr : this->m_w) {
        for (const auto& el : arr) {
            out << el << " ";
        }
        out << "\n";
    }
    for (const auto& arr : this->m_b) {
        for (const auto& el : arr) {
            out << el << " ";
        }
        out << "\n";
    }
    for (const auto& arr : this->v_w) {
        for (const auto& el : arr) {
            out << el << " ";
        }
        out << "\n";
    }
    for (const auto& arr : this->v_b) {
        for (const auto& el : arr) {
            out << el << " ";
        }
        out << "\n";
    }
    out << this->best_epoch << "\n";
    out << this->best_loss << "\n";
    out << this->best_t << "\n";
    for (const auto& arr : this->best_m_w) {
        for (const auto& el : arr) {
            out << el << " ";
        }
        out << "\n";
    }
    for (const auto& arr : this->best_m_b) {
        for (const auto& el : arr) {
            out << el << " ";
        }
        out << "\n";
    }
    for (const auto& arr : this->best_v_w) {
        for (const auto& el : arr) {
            out << el << " ";
        }
        out << "\n";
    }
    for (const auto& arr : this->best_v_b) {
        for (const auto& el : arr) {
            out << el << " ";
        }
        out << "\n";
    }
    auto temp_layers = model->get_layers();
    for (int l = 1; l < temp_layers.size(); l++) {
        int N_l = temp_layers[l]->num_neuros;
        int N_prev = temp_layers[l - 1]->num_neuros;
        for (int w = 0; w < N_l; w++) {
            for (int w2 = 0; w2 < N_prev; w2++) {
                int index = w * N_prev + w2;
                out << best_weights[l - 1][index] << " ";
            }
        }
        out << "\n";

        for (int b = 0; b < N_l; b++) {
            out << best_bias[l - 1][b] << " ";
        }
        out << "\n";
    }
    this->training_view->save_state(out);
}

bool SamSystem::load_state(QFile& file) {
    m_w.clear();
    m_b.clear();
    v_w.clear();
    v_b.clear();
    best_m_w.clear();
    best_m_b.clear();
    best_v_w.clear();
    best_v_b.clear();
    best_weights.clear();
    best_bias.clear();

    this->reset_model();
    QTextStream in(&file);

    bool ok = this->model->load_state(in);
    if (!ok) {
        QMessageBox::critical(this->main_window, "Ошибка", "Модель не соответствует данным!");
        return false;
    }

    QStringList curr_epochs_str = in.readLine().split(" ", Qt::SkipEmptyParts);
    this->curr_epochs = curr_epochs_str[0].toInt();

    QStringList is_standartized_str = in.readLine().split(" ", Qt::SkipEmptyParts);
    this->is_standartized = is_standartized_str[0].toInt();
    if (this->is_standartized) {
        this->data->z_score(this->model->get_layers()[0]->num_neuros);
    }

    this->is_inited = true;

    QStringList t_str = in.readLine().split(" ", Qt::SkipEmptyParts);
    this->t = t_str[0].toInt();

    for (int i = 0; i < this->model->get_weights_size(); i++) {
        QStringList m_w_str = in.readLine().split(" ", Qt::SkipEmptyParts);
        m_w.emplace_back();
        for (int j = 0; j < m_w_str.size(); j++) {
            m_w[i].push_back(m_w_str[j].toFloat());
        }
    }
    for (int i = 0; i < this->model->get_weights_size(); i++) {
        QStringList m_b_str = in.readLine().split(" ", Qt::SkipEmptyParts);
        m_b.emplace_back();
        for (int j = 0; j < m_b_str.size(); j++) {
            m_b[i].push_back(m_b_str[j].toFloat());
        }
    }
    for (int i = 0; i < this->model->get_weights_size(); i++) {
        QStringList v_w_str = in.readLine().split(" ", Qt::SkipEmptyParts);
        v_w.emplace_back();
        for (int j = 0; j < v_w_str.size(); j++) {
            v_w[i].push_back(v_w_str[j].toFloat());
        }
    }
    for (int i = 0; i < this->model->get_weights_size(); i++) {
        QStringList v_b_str = in.readLine().split(" ", Qt::SkipEmptyParts);
        v_b.emplace_back();
        for (int j = 0; j < v_b_str.size(); j++) {
            v_b[i].push_back(v_b_str[j].toFloat());
        }
    }

    QStringList best_epoch_str = in.readLine().split(" ", Qt::SkipEmptyParts);
    this->best_epoch = best_epoch_str[0].toInt();

    QStringList best_loss_str = in.readLine().split(" ", Qt::SkipEmptyParts);
    this->best_loss = best_loss_str[0].toFloat();

    QStringList best_t = in.readLine().split(" ", Qt::SkipEmptyParts);
    this->best_t = best_t[0].toInt();

    for (int i = 0; i < this->model->get_weights_size(); i++) {
        QStringList best_m_w_str = in.readLine().split(" ", Qt::SkipEmptyParts);
        best_m_w.emplace_back();
        for (int j = 0; j < best_m_w_str.size(); j++) {
            best_m_w[i].push_back(best_m_w_str[j].toFloat());
        }
    }
    for (int i = 0; i < this->model->get_weights_size(); i++) {
        QStringList best_m_b_str = in.readLine().split(" ", Qt::SkipEmptyParts);
        best_m_b.emplace_back();
        for (int j = 0; j < best_m_b_str.size(); j++) {
            best_m_b[i].push_back(best_m_b_str[j].toFloat());
        }
    }
    for (int i = 0; i < this->model->get_weights_size(); i++) {
        QStringList best_v_w_str = in.readLine().split(" ", Qt::SkipEmptyParts);
        best_v_w.emplace_back();
        for (int j = 0; j < best_v_w_str.size(); j++) {
            best_v_w[i].push_back(best_v_w_str[j].toFloat());
        }
    }
    for (int i = 0; i < this->model->get_weights_size(); i++) {
        QStringList best_v_b_str = in.readLine().split(" ", Qt::SkipEmptyParts);
        best_v_b.emplace_back();
        for (int j = 0; j < best_v_b_str.size(); j++) {
            best_v_b[i].push_back(best_v_b_str[j].toFloat());
        }
    }

    auto temp_layers = model->get_layers();
    for (int l = 1; l < temp_layers.size(); l++) {
        QStringList weights_str = in.readLine().split(" ", Qt::SkipEmptyParts);
        int N_l = temp_layers[l]->num_neuros;
        int N_prev = temp_layers[l - 1]->num_neuros;
        this->best_weights.push_back(new float[N_l * N_prev]);
        for (int w = 0, c = 0; w < N_l; w++) {
            for (int w2 = 0; w2 < N_prev; w2++) {
                int index = w * N_prev + w2;
                best_weights[l - 1][index] = weights_str[c++].toFloat();
            }
        }
        QStringList bias_str = in.readLine().split(" ", Qt::SkipEmptyParts);
        this->best_bias.push_back(new float[N_l]);
        for (int b = 0; b < N_l; b++) {
            best_bias[l - 1][b] = bias_str[b].toFloat();
        }
    }

    this->training_view->load_state(in);
    return true;
}

bool SamSystem::get_is_standartized() const {
    return this->is_standartized;
}

int SamSystem::get_cols() const {
    return this->data->get_cols();
}

bool SamSystem::compilation_now() const {
    return this->compilation;
}
