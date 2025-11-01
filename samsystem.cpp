#include "samsystem.h"
#include <fstream>

SamSystem::SamSystem(SamView* main_window) {
    data = new DataFrame(main_window);
    processing_data = new DataFrame(main_window);
    model = new SamModel(main_window, this);
    this->main_window = main_window;
    is_standartized = false;
    is_inited = false;

    this->ocl_inited = true;

    if(!ocl_init()) {
        this->ocl_inited = false;
        QMessageBox::warning(main_window, "Ошибка", "Не удалось инициализировать драйвер OpenCL");
    }

    // Поиск платформ
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));

    if(platformsCount == 0) {
        this->ocl_inited = false;
        QMessageBox::warning(main_window, "Ошибка", "Не удалось найти платформы OpenCL");
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
    delete processing_data;
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
    return data->load_data(fileName);
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

    delete processing_data;
    processing_data = new DataFrame(this->main_window);

    if (!processing_data->load_data(fileName)) {
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

    // Обработка данных
    cl_int err;
    // Создание контекста
    cl_context context = clCreateContext(0, 1, &this->curr_device, nullptr, nullptr, &err);
    OCL_SAFE_CALL(err);

    // Создание командной очереди
    cl_command_queue queue = clCreateCommandQueue(context, this->curr_device, 0, &err);
    OCL_SAFE_CALL(err);

    // Загрузка исходного кода ядра
    std::ifstream sourceFile("../../MatrixVectorMultiplicationKernel.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    if(sourceCode.empty()) {
        QMessageBox::warning(main_window, "Ошибка", "Не удалось считать файл ядра");
        return false;
    }
    const char *source_str = sourceCode.c_str();
    size_t source_len = sourceCode.length();

    // Создание программы и ее компиляция
    cl_program program = clCreateProgramWithSource(context, 1, &source_str, &source_len, &err);
    OCL_SAFE_CALL(err);

    err = clBuildProgram(program, 1, &this->curr_device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, this->curr_device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        QVector<char> build_log(log_size);
        clGetProgramBuildInfo(program, this->curr_device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
        OCL_SAFE_CALL(err);
        QMessageBox::warning(main_window, "Ошибка", "Ошибка компиляции ядра");
        return false;
    }

    // Создание и настройка ядра
    cl_kernel kernel = clCreateKernel(program, "matrixBatchMul", &err);
    OCL_SAFE_CALL(err);

    int final_layer_size = temp_layers.back()->num_neuros;

    QVector<QVector<float>> output(final_layer_size);
    for (auto &vec : output)
        vec.reserve(this->processing_data->get_rows());

    auto temp_funcs = this->model->get_funcs();
    QVector<Activation> activations_layers(temp_layers.size(), Activation::LINEAR);
    for (int i = 0; i < temp_funcs.size(); i++) {
        if (dynamic_cast<ReLU*>(temp_funcs[i]) != nullptr) {
            activations_layers[temp_funcs[i]->num_layer] = Activation::RELU;
        }
    }

    for (int i = 0; i < this->processing_data->get_rows(); i += 256) {
        const int size_batch = std::min(256, this->processing_data->get_rows() - i);
        QVector<float> input_vector(size_batch * temp_layers[0]->num_neuros);;

        auto& data = this->processing_data->get_data();
        for (int j = 0; j < size_batch; j++)
            for (int k = 0; k < processing_data->get_cols(); k++)
                input_vector[j * processing_data->get_cols() + k] = data[k][i + j];


        for (int c = 0; c < temp_layers.size() - 1; c++) {
            QVector<float> result_vector(size_batch * temp_layers[c + 1]->num_neuros, 0.0f);

            // 3. Создание буферов (память на устройстве)
            size_t size_A = temp_layers[c + 1]->num_neuros * temp_layers[c]->num_neuros * sizeof(float);
            size_t size_B = size_batch * temp_layers[c]->num_neuros * sizeof(float);
            size_t size_R = size_batch * temp_layers[c + 1]->num_neuros * sizeof(float);
            size_t size_bias = temp_layers[c + 1]->num_neuros * sizeof(float);

            cl_mem cl_matrix_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_A, this->model->get_weight(c), &err);
            OCL_SAFE_CALL(err);
            cl_mem cl_vector_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_B, input_vector.data(), &err);
            OCL_SAFE_CALL(err);
            cl_mem cl_result_vector = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_R, nullptr, &err);
            OCL_SAFE_CALL(err);
            cl_mem cl_bias = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size_bias, this->model->get_bias(c), &err);
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

            if (activations_layers[c] == Activation::RELU) {
                ReLU_func(result_vector);
            }
            input_vector = result_vector;

            // Очистка ресурсов
            clReleaseMemObject(cl_matrix_A);
            clReleaseMemObject(cl_vector_B);
            clReleaseMemObject(cl_result_vector);
        }
        for (int l = 0; l < size_batch; l++)
            for (int d = 0; d < final_layer_size; d++)
                output[d].push_back(input_vector[l * final_layer_size + d]);
    }
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    int index = fileName.lastIndexOf('/');
    QString path;
    if (index != -1) {
        path = fileName.left(index + 1);
    }
    QString fullFileName = path + "output.csv";

    QFile file(fullFileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }

    QTextStream out(&file);

    int rows = output[0].size();
    int cols = output.size();

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            out << QString::number(output[c][r], 'f', 6);
            if (c != cols - 1) out << ",";
        }
        out << "\n";
    }

    file.close();

    QMessageBox::information(main_window, "Выполнено", "Обработка выполнена успешно");

    return true;
}

void SamSystem::ReLU_func(QVector<float>& vector) {
    for (int i = 0; i < vector.size(); i++) {
        vector[i] = (vector[i] >= 0 ? vector[i] : 0);
    }
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
