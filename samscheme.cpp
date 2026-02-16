#include <QApplication>
#include "samscheme.h"
#include "sambuttonsgroup.h"
#include "samfield.h"
#include "samsystem.h"
#include "sammodel.h"
#include "samview.h"

extern double scale;

QString button_style;

QString button_style_n;

QString button_disabled;

SamScheme::SamScheme(SamView *parent, SamSystem *system) : QFrame{parent} {
    button_style = "QPushButton { background-color: #F5F5DC; border: none; border-top: 1px solid black; padding: 0px;"
                           "font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2))) + "pt; border-radius: 0px; }"
                                                                                    "QPushButton:hover { background-color: #DFE036; }"
                                                                                    "QPushButton:pressed { background-color: #AFAAFD; }";

    button_style_n = "QPushButton { background-color: #C5F5FC; border: none; border-top: 1px solid black; padding: 0px;"
                             "font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2))) + "pt; border-radius: 0px; }"
                                                                                      "QPushButton:hover { background-color: #DFE036; }"
                                                                                      "QPushButton:pressed { background-color: #AFAAFD; }";

    button_disabled = "QPushButton { background-color: #D5D5FC; border: none; border-top: 1px solid black; padding: 0px;"
                              "font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2))) + "pt; border-radius: 0px; }"
                                                                                       "QPushButton:hover { background-color: #DFE036; }"
                                                                                       "QPushButton:pressed { background-color: #AFAAFD; }";

    this->view = parent;
    int width = 1350 * scale;
    int height = 980 * scale;
    this->setMinimumSize(width, height);
    this->setStyleSheet("background-color: #F5EBE0;");
    this->layout = new QGridLayout(this);
    this->layout->setContentsMargins(0, 0, 0, 0);
    this->system = system;
    this->field = new SamField(this, this->system);

    // Архитектура модели
    auto *model_struct = new SamButtonsGroup(this);

    auto label_struct = new QLabel("Структура модели", model_struct);
    model_struct->setLabel(label_struct, "#FFD4A9");

    auto *linear_dense = new QPushButton("Полносвязный слой", model_struct);
    model_struct->addBtn(linear_dense, [this](){
        if (!this->system->get_is_inited()) {
            if (field->get_curr_layer().second != -1) {
                this->system->add_layer(new LinearLayer(), field->get_curr_layer().second + 1);
                if (this->system->data_inited() && this->system->get_layers().size() > 1) {
                    auto temp_layers = this->system->get_layers();
                    this->system->set_neuros(this->system->get_shape_data().first - temp_layers[0]->num_neuros, this->system->get_layers().size() - 1);
                    this->system->set_neuros(1, field->get_curr_layer().second + 1);
                }
            }
            else {
                this->system->add_layer(new LinearLayer());
                if (this->system->data_inited() && this->system->get_layers().size() > 1) {
                    auto temp_layers = this->system->get_layers();
                    this->system->set_neuros(this->system->get_shape_data().first - temp_layers[0]->num_neuros, this->system->get_layers().size() - 1);
                    this->system->set_neuros(1, this->system->get_layers().size() - 2);
                }
            }
            this->field->set_layers(this->system->get_layers());
        }
        else {
            QMessageBox::warning(this, "Ошибка", "Модель была инициализирована");
        }
        this->field->setFocus();
    });

    // // Регуляризация
    // auto *regularization = new SamButtonsGroup(this);

    // auto label_regularization = new QLabel("Регуляризация", regularization);
    // regularization->setLabel(label_regularization, "#FFD4A9");

    // auto *dropout_dense = new QPushButton("Слой отсева", regularization);
    // regularization->addBtn(dropout_dense, [](){});

    // auto *batchnorm_dense = new QPushButton("Слой пакетной нормализации", regularization);
    // regularization->addBtn(batchnorm_dense, [](){});

    // regularization->addStretch();

    // Функции активации
    auto *model_analysis = new SamButtonsGroup(this);

    auto label_analysis = new QLabel("Функции активации", model_analysis);
    model_analysis->setLabel(label_analysis, "#FFD4A9");

    auto *ReLU_btn = new QPushButton("ReLU", model_analysis);
    model_analysis->addBtn(ReLU_btn, [this](){
        if (!this->system->get_is_inited()) {
            if (this->field->get_curr_layer().second != -1) {
                auto funcs = this->system->get_funcs();
                for (int i = 0; i < funcs.size(); i++) {
                    if (funcs[i]->num_layer == this->field->get_curr_layer().second) {
                        QMessageBox::warning(this, "Ошибка", "К данному слою уже применена функция активации");
                        return;
                    }
                }
                this->system->add_func(new ActivationFunction("ReLU", this->field->get_curr_layer().second));
                this->field->set_funcs(this->system->get_funcs());
            }
            else QMessageBox::warning(this, "Ошибка", "Выберите слой");
        }
        else {
            QMessageBox::warning(this, "Ошибка", "Модель была инициализирована");
        }
        this->field->setFocus();
    });

    auto *Sigmoid = new QPushButton("Сигмоидальная", model_analysis);
    model_analysis->addBtn(Sigmoid, [this](){
        if (!this->system->get_is_inited()) {
            if (this->field->get_curr_layer().second != -1) {
                auto funcs = this->system->get_funcs();
                for (int i = 0; i < funcs.size(); i++) {
                    if (funcs[i]->num_layer == this->field->get_curr_layer().second) {
                        QMessageBox::warning(this, "Ошибка", "К данному слою уже применена функция активации");
                        return;
                    }
                }
                this->system->add_func(new ActivationFunction("Sigmoid", this->field->get_curr_layer().second));
                this->field->set_funcs(this->system->get_funcs());
            }
            else QMessageBox::warning(this, "Ошибка", "Выберите слой");
        }
        else {
            QMessageBox::warning(this, "Ошибка", "Модель была инициализирована");
        }
        this->field->setFocus();
    });

    auto *Tanh = new QPushButton("Гиперболический тангенс", model_analysis);
    model_analysis->addBtn(Tanh, [this](){
        if (!this->system->get_is_inited()) {
            if (this->field->get_curr_layer().second != -1) {
                auto funcs = this->system->get_funcs();
                for (int i = 0; i < funcs.size(); i++) {
                    if (funcs[i]->num_layer == this->field->get_curr_layer().second) {
                        QMessageBox::warning(this, "Ошибка", "К данному слою уже применена функция активации");
                        return;
                    }
                }
                this->system->add_func(new ActivationFunction("Tanh", this->field->get_curr_layer().second));
                this->field->set_funcs(this->system->get_funcs());
            }
            else QMessageBox::warning(this, "Ошибка", "Выберите слой");
        }
        else {
            QMessageBox::warning(this, "Ошибка", "Модель была инициализирована");
        }
        this->field->setFocus();
    });

    auto *SoftMax = new QPushButton("SoftMax", model_analysis);
    model_analysis->addBtn(SoftMax, [this](){
        if (!this->system->get_is_inited()) {
            if (this->field->get_curr_layer().second != -1) {
                auto funcs = this->system->get_funcs();
                for (int i = 0; i < funcs.size(); i++) {
                    if (funcs[i]->num_layer == this->field->get_curr_layer().second) {
                        QMessageBox::warning(this, "Ошибка", "К данному слою уже применена функция активации");
                        return;
                    }
                }
                this->system->add_func(new ActivationFunction("SoftMax", this->field->get_curr_layer().second));
                this->field->set_funcs(this->system->get_funcs());
            }
            else QMessageBox::warning(this, "Ошибка", "Выберите слой");
        }
        else {
            QMessageBox::warning(this, "Ошибка", "Модель была инициализирована");
        }
        this->field->setFocus();
    });

    model_analysis->addStretch();

    // Контейнер для 2 списков
    auto *containeer = new QFrame(this);
    containeer->setFixedWidth(390 * (scale + (1 - scale) / 2));
    containeer->setFixedHeight(400 * (scale + (1 - scale) / 2));
    containeer->setStyleSheet("background-color: #F4DCB0; border: 1px solid black; border-radius: 40px; padding: 10px;");
    auto *layout2 = new QVBoxLayout(containeer);
    layout2->setSpacing(10 * scale);
    containeer->setLayout(layout2);
    auto label_containeer = new QLabel("Архитектура", containeer);
    label_containeer->setStyleSheet("padding: 0px; font-family: 'Inter'; font-size: " + QString::number(int(16 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border: none;");
    label_containeer->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
    label_containeer->setFixedHeight(30);
    layout2->addWidget(label_containeer);
    layout2->addWidget(model_struct, 3, Qt::AlignHCenter);
    // layout2->addWidget(regularization, 3, Qt::AlignHCenter);
    layout2->addWidget(model_analysis, 3, Qt::AlignHCenter);
    layout2->setContentsMargins(5 * scale, 0, 5 * scale, 5 * scale);

    // Действия
    auto *actions = new SamButtonsGroup(this);

    auto label_actions = new QLabel("Действия", actions);
    actions->setLabel(label_actions, "#C3DF75");

    auto *load_data = new QPushButton("Загрузить данные", actions);
    auto *z_score = new QPushButton("Стандартизировать данные", actions);
    auto *init_model = new QPushButton("Инициализировать модель", actions);

    actions->addBtn(load_data, [this, load_data, z_score](){
        if (!this->system->get_is_inited()) {
            if (this->system->load_data()) {
                load_data->setStyleSheet(button_style_n);
                z_score->setStyleSheet(button_style);
                this->system->reset_standartization();

                auto temp_layers = this->system->get_layers();
                if (temp_layers.size() > 1) {
                    if (temp_layers[0]->num_neuros >= this->system->get_shape_data().first) {
                        temp_layers[0]->num_neuros = this->system->get_shape_data().first - 1;
                    }
                    this->system->set_neuros(this->system->get_shape_data().first - temp_layers[0]->num_neuros,
                                             this->system->get_layers().size() - 1);
                }
            }
            this->field->repaint();
        }
        else {
            QMessageBox::warning(this, "Ошибка", "Модель была инициализирована");
        }
    });

    actions->addBtn(init_model, [this, init_model, linear_dense,
                                 ReLU_btn, Sigmoid, Tanh, SoftMax, load_data, z_score](){
        if (this->system->data_inited()) {
            if (this->system->get_layers().size() >= 2) {
                if (this->system->get_ocl_inited()) {
                    auto temp_funcs = this->system->get_funcs();
                    for (int i = 0; i < temp_funcs.size(); i++) {
                        if (temp_funcs[i]->func == "SoftMax" && temp_funcs[i]->num_layer != this->system->get_layers().size() - 1) {
                            QMessageBox::warning(this, "Ошибка", "Функция активации SoftMax может быть лишь на последнем слое");
                            return;
                        }
                        else if (temp_funcs[i]->func == "SoftMax") {
                            auto temp_layers = this->system->get_layers();
                            if (temp_layers.back()->num_neuros == 1) {
                                QMessageBox::warning(this, "Ошибка", "Функция активации SoftMax не может быть применена к слою с 1 нейроном");
                                return;
                            }
                        }
                        else if (temp_funcs[i]->num_layer == 0) {
                            QMessageBox::warning(this, "Ошибка", "Функция активации не может быть применена ко входному слою");
                            return;
                        }
                    }
                    bool neurons_ok = true;
                    auto temp_layers = this->system->get_layers();
                    for (auto& el : temp_layers) {
                        if (el->num_neuros == 0) neurons_ok = false;
                    }
                    if (neurons_ok) {
                        this->system->modelize();
                        if (this->system->get_is_inited()) {
                            init_model->setStyleSheet(button_style_n);
                            init_model->setText("Сбросить модель");

                            linear_dense->setStyleSheet(button_disabled);

                            ReLU_btn->setStyleSheet(button_disabled);
                            SoftMax->setStyleSheet(button_disabled);
                            Sigmoid->setStyleSheet(button_disabled);
                            Tanh->setStyleSheet(button_disabled);

                            this->system->init_model();
                        }
                        else {
                            init_model->setStyleSheet(button_style);
                            init_model->setText("Инициализировать модель");

                            linear_dense->setStyleSheet(button_style);

                            ReLU_btn->setStyleSheet(button_style);
                            SoftMax->setStyleSheet(button_style);
                            Sigmoid->setStyleSheet(button_style);
                            Tanh->setStyleSheet(button_style);

                            load_data->setStyleSheet(button_style);
                            z_score->setStyleSheet(button_style);
                            this->system->reset_data();
                            this->field->repaint();

                            this->system->reset_model();
                        }
                    }
                    else {
                        QMessageBox::warning(this, "Ошибка", "OpenCl не был инициализирован");
                    }
                }
                else {
                    QMessageBox::warning(this, "Ошибка", "Слои не могут содержать 0 нейронов");
                }
            }
            else {
                QMessageBox::warning(this, "Ошибка", "Слишком мало слоёв");
            }
        }
        else {
            QMessageBox::warning(this, "Ошибка", "Данные не загружены");
        }
    });

    actions->addBtn(z_score, [this, z_score](){
        if (this->system->get_is_inited()) {
            auto temp_layers = this->system->get_layers();
            if (this->system->z_score(temp_layers[0]->num_neuros)) {
                z_score->setStyleSheet(button_style_n);
            }
        }
        else {
            QMessageBox::warning(this, "Ошибка", "Модель не инициализирована");
        }
    });

    auto *data_processing = new QPushButton("Обработать данные", actions);
    actions->addBtn(data_processing, [this](){
        if (this->system->get_is_inited()) {
            this->system->process_data();
        }
        else {
            QMessageBox::warning(this, "Ошибка", "Модель не была инициализирована");
        }
    });

    auto *save_model = new QPushButton("Сохранить модель", actions);
    actions->addBtn(save_model, [this](){
        if (this->system->get_epochs() > 0) {
            QString fileName = QFileDialog::getSaveFileName(
                this,
                "Сохранить файл",
                "",
                "Text Files (*.txt);;All Files (*)"
                );

            if (fileName.isEmpty())
                return;

            QFile file(fileName);
            if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
                QMessageBox::warning(this, "Ошибка", "Ошибка создания файла");
                return;
            }

            this->system->save_state(file);

            file.close();
            QMessageBox::information(this, "Успех", "Модель сохранена успешно");
        }
        else {
            QMessageBox::warning(this, "Ошибка", "Обучения модели не было");
        }
    });

    auto *load_model = new QPushButton("Загрузить модель", actions);
    actions->addBtn(load_model, [this, init_model, linear_dense, ReLU_btn, SoftMax, Sigmoid, Tanh, z_score](){
        if (!this->system->data_inited()) {
            QMessageBox::warning(this, "Ошибка", "Данные не загружены");
            return;
        }

        QString fileName = QFileDialog::getOpenFileName(
            this,
            "Открыть файл",
            "",
            "Text Files (*.txt);;All Files (*)"
            );

        if (fileName.isEmpty())
            return;

        QFile file(fileName);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QMessageBox::warning(this, "Ошибка", "Ошибка открытия файла");
            return;
        }

        this->system->load_state(file);

        file.close();

        init_model->setStyleSheet(button_style_n);
        init_model->setText("Сбросить модель");

        linear_dense->setStyleSheet(button_disabled);

        ReLU_btn->setStyleSheet(button_disabled);
        SoftMax->setStyleSheet(button_disabled);
        Sigmoid->setStyleSheet(button_disabled);
        Tanh->setStyleSheet(button_disabled);

        if (this->system->get_is_standartized()) {
            z_score->setStyleSheet(button_style_n);
        }

        this->field->set_layers(this->system->get_layers());
        this->field->set_funcs(this->system->get_funcs());
        QMessageBox::information(this, "Успех", "Модель загружена успешно");
    });

    auto *manual_input = new QPushButton("Ручной ввод", actions);
    actions->addBtn(manual_input, [](){

    });

    actions->addStretch();

    // Устройства
    auto *devices = new SamButtonsGroup(this);

    auto label_devices = new QLabel("Устройства", devices);
    devices->setLabel(label_devices, "#E7C4E2");

    QVector<QPair<cl_device_id, QString>> devices_list = system->get_devices();
    for (int i = 0; i < devices_list.size(); i++) {
        auto *btn = new DeviceButton(devices_list[i].second, devices, devices_list[i].first);
        devices->addBtn(btn, [this, btn, devices](){
            for (auto* el : devices->buttons) {
                el->setStyleSheet(button_style);
            }
            btn->setStyleSheet(button_style_n);
            this->system->set_device(dynamic_cast<DeviceButton*>(btn)->index);
        }, devices_list[i].second);
        if (!i) {
            btn->setStyleSheet(button_style_n);
            this->system->set_device(dynamic_cast<DeviceButton*>(btn)->index);
        }
    }

    devices->addStretch();

    // Вся панель управления
    auto area = new QScrollArea(this);
    QString styleSheet = R"(
    QScrollArea {
        border: 1px solid #EEE;
    }
    QScrollBar:vertical {
        border: none;
        background: transparent;
        width: 8px;
        margin: 0px;
    }
    QScrollBar::groove:vertical {
        border: none;
        background: rgba(0, 0, 0, 30);
        border-radius: 4px;
    }
    QScrollBar::handle:vertical {
        background: rgba(100, 100, 100, 150);
        min-height: 20px;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical:hover {
        background: rgba(100, 100, 100, 200);
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
        width: 0px;
    }
    )";
    area->setStyleSheet(styleSheet);
    auto form = new QWidget(area);
    area->setMinimumWidth(400 * (scale + (1 - scale) / 2) + 25);
    area->setWidget(form);
    area->setWidgetResizable(true);
    area->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    area->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    form->setStyleSheet("background-color: #F5EBE0;");
    auto layout3 = new QVBoxLayout(form);
    layout3->setSpacing(20 * scale);
    layout3->setContentsMargins(15 * scale, 15 * scale, 15 * scale, 15 * scale);
    form->setLayout(layout3);
    layout3->addWidget(containeer, 0, Qt::AlignHCenter);
    layout3->addWidget(actions, 0, Qt::AlignHCenter);
    layout3->addWidget(devices, 0, Qt::AlignHCenter);
    layout3->addStretch();

    // Контейнер для поля
    auto* btn_scheme = new QPushButton("Модель", this);
    btn_scheme->setStyleSheet("QPushButton { background-color: #BFEBC1; border-top-left-radius: 20px; padding: 0px; font-family: 'Inter'; font-size: " + QString::number(int(16 * (scale + (1 - scale) / 2) * 10) / 10) + "pt;"
                              "border-top-right-radius: 0px; border-bottom-left-radius: 0px; border-bottom-right-radius: 0px; border: 2px solid black;"
                              "border-bottom: none; } QPushButton:hover { background-color: #DFE036; }"
                              "QPushButton:pressed { background-color: #AFAAFD; }");
    btn_scheme->setFixedSize(175, 55 * scale);

    auto* btn_training = new QPushButton("Обучение", this);
    btn_training->setStyleSheet("QPushButton { background-color: #FFD4AA; border-top-left-radius: 0px; padding: 0px; font-family: 'Inter'; font-size: " + QString::number(int(16 * (scale + (1 - scale) / 2) * 10) / 10) + "pt;"
                                "border-top-right-radius: 20px; border-bottom-left-radius: 0px; border-bottom-right-radius: 0px; border: 2px solid black;"
                                "border-bottom: none; border-left: none; } QPushButton:hover { background-color: #DFE036; }"
                                "QPushButton:pressed { background-color: #AFAAFD; }");
    btn_training->setFixedSize(175, 55 * scale);
    connect(btn_training, &QPushButton::clicked, this, [this](){
        if (this->system->get_is_inited()) {
            this->view->setCurrentIndex(1);
        }
        else {
            QMessageBox::warning(this, "Ошибка", "Модель не была инициализирована");
        }
    });

    auto *field_container = new QWidget(this);
    auto *field_layout = new QGridLayout(field_container);
    field_layout->setContentsMargins(10 * scale, 10 * scale, 10 * scale, 10 * scale);
    field_layout->addWidget(btn_scheme, 0, 0, Qt::AlignRight);
    field_layout->addWidget(btn_training, 0, 1, Qt::AlignLeft);
    field_layout->addWidget(field, 1, 0, 1, 2);
    field_layout->setSpacing(0);


    // Добавление в макет
    this->layout->addWidget(area, 0, 0);
    this->layout->addWidget(field_container, 0, 1);
    this->layout->setColumnStretch(0, 1);
    this->layout->setColumnStretch(1, 4);
}

DeviceButton::DeviceButton(QString text, QWidget* parent, cl_device_id index) : QPushButton(text, parent) {
    this->index = index;
}
