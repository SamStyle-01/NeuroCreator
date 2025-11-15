#include <QApplication>
#include "samtraining.h"
#include "samsystem.h"
#include "samchart.h"
#include "sammodel.h"
#include "samview.h"

extern double scale;

QString radio_button_style = R"(
    QWidget { border: none; }
    QRadioButton::indicator {
        width: 20px;
        height: 20px;
    }
    QRadioButton::indicator::unchecked {
        border: 2px solid #555;
        background-color: #F5EBE0; /* Цвет кружка, когда не выбран */
        border-radius: 10px;
    }
    QRadioButton::indicator::checked {
        border: 2px solid #222;
        background-color: #7DD961; /* Цвет кружка, когда выбран */
        border-radius: 10px;
    }
)";

SamTraining::SamTraining(SamView *parent, SamSystem *system) : QFrame{parent} {
    this->view = parent;
    int width = 1350 * scale;
    int height = 930 * scale;
    this->setMinimumSize(width, height);
    this->setStyleSheet("background-color: #F5EBE0;");
    this->layout = new QGridLayout(this);
    this->layout->setContentsMargins(0, 0, 0, 0);
    this->system = system;
    this->field = new QFrame(parent);
    this->field->setStyleSheet("background-color: #F8F8FF; border: 2px solid black;");

    SamChart *chartView = new SamChart(this->field, system);

    QVBoxLayout *layout = new QVBoxLayout(this->field);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(chartView);

    // Панель эпох
    QWidget *epochs_containeer = new QWidget(this);
    epochs_containeer->setFixedSize(405 * (scale + (1 - scale) / 2), 245);
    epochs_containeer->setStyleSheet("background-color: #F4DCB0; border: 1px solid black; border-radius: 40px; padding: 15px;");
    QGridLayout *layout_epochs = new QGridLayout(epochs_containeer);

    QLabel* epochs_lbl = new QLabel("Настройки обучения", epochs_containeer);
    epochs_lbl->setFixedSize(240, 85);
    epochs_lbl->setStyleSheet("font-family: 'Inter'; font-size: 16pt; border: none;");

    QLabel* epochs_num = new QLabel("Количество эпох:", epochs_containeer);
    epochs_num->setStyleSheet("font-family: 'Inter'; font-size: 14pt; border: none;");
    epochs_num->setMaximumWidth(200);

    epochs_input = new QLineEdit("0", epochs_containeer);
    epochs_input->setMaximumSize(200 * (scale + (1 - scale) / 2), 50);
    epochs_input->setStyleSheet("font-family: 'Inter'; font-size: 14pt; background-color: #F5F5DC; border-radius: 5px;");
    epochs_input->setValidator(new QIntValidator(1, 1000000, epochs_input));

    QLabel* lr_num = new QLabel("Скорость обучения:", epochs_containeer);
    lr_num->setStyleSheet("font-family: 'Inter'; font-size: 14pt; border: none;");
    lr_num->setMaximumWidth(210);

    lr_input = new QLineEdit("0,01", epochs_containeer);
    lr_input->setMaximumSize(200 * (scale + (1 - scale) / 2), 50);
    lr_input->setStyleSheet("font-family: 'Inter'; font-size: 14pt; background-color: #F5F5DC; border-radius: 5px;");
    lr_input->setValidator(new QDoubleValidator(1, 1000000, 7, epochs_input));

    QLabel* batch_size_num = new QLabel("Размер батча:", epochs_containeer);
    batch_size_num->setStyleSheet("font-family: 'Inter'; font-size: 14pt; border: none;");
    batch_size_num->setMaximumWidth(200);

    batch_size_input = new QLineEdit("512", epochs_containeer);
    batch_size_input->setMaximumSize(200 * (scale + (1 - scale) / 2), 50);
    batch_size_input->setStyleSheet("font-family: 'Inter'; font-size: 14pt; background-color: #F5F5DC; border-radius: 5px;");
    batch_size_input->setValidator(new QIntValidator(1, 100000, epochs_input));

    layout_epochs->addWidget(epochs_lbl, 0, 0, 1, 2, Qt::AlignHCenter | Qt::AlignTop);
    layout_epochs->addWidget(epochs_num, 1, 0, Qt::AlignRight | Qt::AlignVCenter);
    layout_epochs->addWidget(epochs_input, 1, 1, Qt::AlignLeft | Qt::AlignVCenter);
    layout_epochs->addWidget(lr_num, 2, 0, Qt::AlignRight | Qt::AlignVCenter);
    layout_epochs->addWidget(lr_input, 2, 1, Qt::AlignLeft | Qt::AlignVCenter);
    layout_epochs->addWidget(batch_size_num, 3, 0, Qt::AlignRight | Qt::AlignVCenter);
    layout_epochs->addWidget(batch_size_input, 3, 1, Qt::AlignLeft | Qt::AlignVCenter);
    layout_epochs->setContentsMargins(15, 2, 15, 15);

    // Панель данных
    QWidget *data_containeer = new QWidget(this);
    data_containeer->setFixedSize(405 * (scale + (1 - scale) / 2), 185);
    data_containeer->setStyleSheet("background-color: #F4DCB0; border: 1px solid black; border-radius: 40px; padding: 15px;");
    QGridLayout *layout_data = new QGridLayout(data_containeer);

    QLabel* data_lbl = new QLabel("Данные", data_containeer);
    data_lbl->setFixedSize(130, 100);
    data_lbl->setStyleSheet("font-family: 'Inter'; font-size: 16pt; border: none;");
    data_lbl->setAlignment(Qt::AlignHCenter | Qt::AlignTop);

    QLabel* data_num_train = new QLabel("Доля обучения:", data_containeer);
    data_num_train->setStyleSheet("font-family: 'Inter'; font-size: 14pt; border: none;");
    data_num_train->setMaximumWidth(200);

    data_input_train = new QLineEdit("100", data_containeer);
    data_input_train->setMaximumSize(200 * (scale + (1 - scale) / 2), 50);
    data_input_train->setStyleSheet("font-family: 'Inter'; font-size: 14pt; background-color: #F5F5DC; border-radius: 5px;");
    data_input_train->setValidator(new QIntValidator(0, 100, data_input_train));

    QLabel* data_num_valid = new QLabel("Доля валидации:", data_containeer);
    data_num_valid->setStyleSheet("font-family: 'Inter'; font-size: 14pt; border: none;");
    data_num_valid->setMaximumWidth(200);

    QLineEdit* data_input_valid = new QLineEdit("0", data_containeer);
    data_input_valid->setMaximumSize(200 * (scale + (1 - scale) / 2), 50);
    data_input_valid->setStyleSheet("font-family: 'Inter'; font-size: 14pt; background-color: #F5F5DC; border-radius: 5px;");
    data_input_valid->setValidator(new QIntValidator(0, 100, data_input_valid));

    connect(data_input_train, &QLineEdit::textChanged, this, [this, data_input_valid](const QString &text) {
        if (text == "") {
            data_input_train->setText("");
            data_input_valid->setText("");
        }
        else if (data_input_train->text().toInt() <= 0) {
            data_input_train->setText("1");
            data_input_valid->setText("99");
        }
        else if (data_input_train->text().toInt() > 100) {
            data_input_train->setText("100");
            data_input_valid->setText("0");
        }
        else {
            data_input_train->setText(text);
            data_input_valid->setText(QString::number(100 - text.toInt()));
        }
    });

    connect(data_input_valid, &QLineEdit::textChanged, this, [this, data_input_valid](const QString &text) {
        if (text == "") {
            data_input_valid->setText("");
            data_input_train->setText("");
        }
        else if (data_input_valid->text().toInt() <= 0) {
            data_input_valid->setText("0");
            data_input_train->setText("100");
        }
        else if (data_input_valid->text().toInt() > 100) {
            data_input_valid->setText("99");
            data_input_train->setText("1");
        }
        else {
            data_input_valid->setText(text);
            data_input_train->setText(QString::number(100 - text.toInt()));
        }
    });

    layout_data->addWidget(data_lbl, 0, 0, 1, 2, Qt::AlignHCenter | Qt::AlignTop);
    layout_data->addWidget(data_num_train, 1, 0, Qt::AlignRight | Qt::AlignVCenter);
    layout_data->addWidget(data_input_train, 1, 1, Qt::AlignLeft | Qt::AlignVCenter);
    layout_data->addWidget(data_num_valid, 2, 0, Qt::AlignRight | Qt::AlignVCenter);
    layout_data->addWidget(data_input_valid, 2, 1, Qt::AlignLeft | Qt::AlignVCenter);
    layout_data->setContentsMargins(15, 2, 15, 15);

    // Панель функций потерь
    QWidget *loss_containeer = new QWidget(this);
    loss_containeer->setFixedSize(405 * (scale + (1 - scale) / 2), 212);
    loss_containeer->setStyleSheet("background-color: #F4DCB0; border: 1px solid black; border-radius: 40px; padding: 15px;");
    QGridLayout *layout_loss = new QGridLayout(loss_containeer);
    layout_loss->setSpacing(0);
    QButtonGroup* losses = new QButtonGroup(loss_containeer);
    connect(losses, &QButtonGroup::idClicked, [this](int id){
        switch (id) {
            case 1:
                this->curr_loss = LossFunc::MSE;
                break;
            case 2:
                this->curr_loss = LossFunc::MAE;
                break;
            case 3:
                this->curr_loss = LossFunc::CROSSENTROPY;
                break;
            default:
                QMessageBox::critical(this, "Ошибка", "Ошибка выбора функции потерь");
        }
    });

    QLabel* loss_lbl = new QLabel("Функция потерь", loss_containeer);
    loss_lbl->setFixedSize(230, 75);
    loss_lbl->setStyleSheet("font-family: 'Inter'; font-size: 16pt; border: none;");
    loss_lbl->setAlignment(Qt::AlignHCenter | Qt::AlignTop);

    QLabel* MSE_lbl = new QLabel("MSE:", loss_containeer);
    MSE_lbl->setStyleSheet("font-family: 'Inter'; font-size: 14pt; border: none;");

    QRadioButton* MSE_loss = new QRadioButton(loss_containeer);
    MSE_loss->setMinimumSize(80, 20);
    MSE_loss->setStyleSheet(radio_button_style);

    MSE_loss->setChecked(true);
    this->curr_loss = LossFunc::MSE;
    losses->addButton(MSE_loss);
    losses->setId(MSE_loss, 1);

    QLabel* MAE_lbl = new QLabel("MAE:", loss_containeer);
    MAE_lbl->setStyleSheet("font-family: 'Inter'; font-size: 14pt; border: none;");

    QRadioButton* MAE_loss = new QRadioButton(loss_containeer);
    MAE_loss->setMinimumSize(80, 20);
    MAE_loss->setStyleSheet(radio_button_style);
    losses->addButton(MAE_loss);
    losses->setId(MAE_loss, 2);

    QLabel* cross_entropy_lbl = new QLabel("CrossEntropy:", loss_containeer);
    cross_entropy_lbl->setStyleSheet("font-family: 'Inter'; font-size: 14pt; border: none;");

    QRadioButton* cross_entropy_loss = new QRadioButton(loss_containeer);
    cross_entropy_loss->setMinimumSize(80, 20);
    cross_entropy_loss->setStyleSheet(radio_button_style);
    losses->addButton(cross_entropy_loss);
    losses->setId(cross_entropy_loss, 3);

    layout_loss->addWidget(loss_lbl, 0, 0, 1, 2, Qt::AlignHCenter | Qt::AlignTop);
    layout_loss->addWidget(MSE_lbl, 1, 0, Qt::AlignRight | Qt::AlignVCenter);
    layout_loss->addWidget(MSE_loss, 1, 1, Qt::AlignLeft | Qt::AlignVCenter);
    layout_loss->addWidget(MAE_lbl, 2, 0, Qt::AlignRight | Qt::AlignVCenter);
    layout_loss->addWidget(MAE_loss, 2, 1, Qt::AlignLeft | Qt::AlignVCenter);
    layout_loss->addWidget(cross_entropy_lbl, 3, 0, Qt::AlignRight | Qt::AlignVCenter);
    layout_loss->addWidget(cross_entropy_loss, 3, 1, Qt::AlignLeft | Qt::AlignVCenter);
    layout_loss->setContentsMargins(15, 2, 15, 10);

    // Пройденные эпохи
    curr_epochs = new QLabel("Пройдено эпох: 0", this);
    curr_epochs->setStyleSheet("background-color: #E1E0F5; font-family: 'Inter'; font-size: 14pt;"
                               "border: 1px solid black; border-radius: 15px;");
    curr_epochs->setMinimumSize(405 * (scale + (1 - scale) / 2), 40);
    curr_epochs->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);

    // Кнопка Тестировать
    QPushButton* test_model = new QPushButton("Тестировать", this);
    test_model->setStyleSheet("QPushButton { background-color: #C5F3FF; border: 1px solid black;"
                          "font-family: 'Inter'; font-size: 14pt; border-radius: 20px; }"
                          "QPushButton:hover { background-color: #DFE036; }"
                          "QPushButton:pressed { background-color: #AFAAFD; }");
    test_model->setMinimumSize(405 * (scale + (1 - scale) / 2), 60);
    connect(test_model, &QPushButton::clicked, this, [this, test_model](){
        this->system->test_data();
    });

    // Кнопка Загрузить лучшую модель
    QPushButton* load_best_model = new QPushButton("Загрузить лучшую модель", this);
    load_best_model->setStyleSheet("QPushButton { background-color: #E1E0F5; border: 1px solid black;"
                              "font-family: 'Inter'; font-size: 14pt; border-radius: 20px; }"
                              "QPushButton:hover { background-color: #DFE036; }"
                              "QPushButton:pressed { background-color: #AFAAFD; }");
    load_best_model->setMinimumSize(405 * (scale + (1 - scale) / 2), 60);
    connect(load_best_model, &QPushButton::clicked, this, [this, load_best_model](){
        if (this->system->get_epochs()) {
            this->system->set_best_model();
            QMessageBox::information(this, "Успех", "Лучшая модель была загружена");
        }
        else {
            QMessageBox::warning(this, "Ошибка", "Обучения ещё не было");
            return;
        }
    });

    // Кнопка Начать/Остановить обучение
    QPushButton* fit_it = new QPushButton("Начать обучение", this);
    fit_it->setStyleSheet("QPushButton { background-color: #CDFFBA; border: 1px solid black;"
                          "font-family: 'Inter'; font-size: 14pt; border-radius: 20px; }"
                          "QPushButton:hover { background-color: #DFE036; }"
                          "QPushButton:pressed { background-color: #AFAAFD; }");
    fit_it->setMinimumSize(405 * (scale + (1 - scale) / 2), 60);
    connect(fit_it, &QPushButton::clicked, this, [this, fit_it, data_input_valid, load_best_model](){
        if (data_input_train->text() == "" || data_input_valid->text() == ""
            || lr_input->text() == "" || epochs_input->text() == ""
            || batch_size_input->text() == "") {
            QMessageBox::warning(this, "Ошибка", "Заполнены не все параметры обучения");
            return;
        }
        load_best_model->setStyleSheet("QPushButton { background-color: #CDFFBA; border: 1px solid black;"
                          "font-family: 'Inter'; font-size: 14pt; border-radius: 20px; }"
                          "QPushButton:hover { background-color: #DFE036; }"
                          "QPushButton:pressed { background-color: #AFAAFD; }");
        this->system->backpropagation();
        this->system->set_is_training(true);
    });

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
    area->setMinimumWidth(400 * (scale + (1 - scale) / 2) + 35);
    area->setWidget(form);
    area->setWidgetResizable(true);
    area->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    area->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    form->setStyleSheet("background-color: #F5EBE0;");
    auto layout3 = new QVBoxLayout(form);
    layout3->setSpacing(20 * scale);
    layout3->setContentsMargins(15 * scale, 15 * scale, 15 * scale, 15 * scale);
    form->setLayout(layout3);
    layout3->addWidget(epochs_containeer, 0, Qt::AlignHCenter);
    layout3->addWidget(data_containeer, 0, Qt::AlignHCenter);
    layout3->addWidget(loss_containeer, 0, Qt::AlignHCenter);
    layout3->addWidget(curr_epochs, 0, Qt::AlignHCenter);
    layout3->addWidget(load_best_model, 0, Qt::AlignHCenter);
    layout3->addWidget(test_model, 0, Qt::AlignHCenter);
    layout3->addWidget(fit_it, 0, Qt::AlignHCenter);
    layout3->addStretch();

    // Контейнер для поля
    auto* btn_scheme = new QPushButton("Модель", this);
    btn_scheme->setStyleSheet("QPushButton { background-color: #FFD4AA; border-top-left-radius: 20px; "
                              "padding: 0px; font-family: 'Inter'; font-size: 16pt;"
                              "border-top-right-radius: 0px; border-bottom-left-radius: 0px; "
                              "border-bottom-right-radius: 0px; border: 2px solid black;"
                              "border-bottom: none; } QPushButton:hover { background-color: #DFE036; }"
                              "QPushButton:pressed { background-color: #AFAAFD; }");
    btn_scheme->setFixedSize(175, 55 * scale);
    connect(btn_scheme, &QPushButton::clicked, this, [this](){
        this->view->setCurrentIndex(0);
    });

    auto* btn_training = new QPushButton("Обучение", this);
    btn_training->setStyleSheet("QPushButton { background-color: #BFEBC1; border-top-left-radius: 0px;"
                                "padding: 0px; font-family: 'Inter'; font-size: 16pt;"
                                "border-top-right-radius: 20px; border-bottom-left-radius: 0px;"
                                "border-bottom-right-radius: 0px; border: 2px solid black;"
                                "border-bottom: none; border-left: none; }"
                                "QPushButton:hover { background-color: #DFE036; }"
                                "QPushButton:pressed { background-color: #AFAAFD; }");
    btn_training->setFixedSize(175, 55 * scale);

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

int SamTraining::get_epochs() const {
    return this->epochs_input->text() != "" ? this->epochs_input->text().toInt() : 0;
}

void SamTraining::set_epochs(int epochs) {
    epochs_input->setText(QString::number(epochs));
}

LossFunc SamTraining::get_loss_func() const {
    return curr_loss;
}

void SamTraining::set_epochs_view(int epochs) {
    this->curr_epochs->setText("Пройдено эпох: " + QString::number(epochs));
}

int SamTraining::get_train_share() const {
    return this->data_input_train->text().toInt();
}

float SamTraining::get_learning_rate() const {
    return this->lr_input->text().replace(",", ".").toFloat();
}

int SamTraining::get_batch_size() const {
    return this->batch_size_input->text().toInt();
}
