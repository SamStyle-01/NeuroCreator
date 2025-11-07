#include <QApplication>
#include "samtraining.h"
#include "samsystem.h"
#include "samchart.h"
#include "sammodel.h"
#include "samview.h"

extern QString button_style;

extern QString button_style_n;

extern QString button_disabled;

extern double scale;

SamTraining::SamTraining(SamView *parent, SamSystem *system) : QFrame{parent} {
    this->view = parent;
    int width = 1300 * scale;
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
    epochs_containeer->setFixedSize(405, 135);
    epochs_containeer->setStyleSheet("background-color: #F4DCB0; border: 1px solid black; border-radius: 40px; padding: 15px;");
    QGridLayout *layout_epochs = new QGridLayout(epochs_containeer);

    QLabel* epochs_lbl = new QLabel("Эпохи", epochs_containeer);
    epochs_lbl->setFixedSize(130, 90);
    epochs_lbl->setStyleSheet("font-family: 'Inter'; font-size: 20pt; border: none;");

    QLabel* epochs_num = new QLabel("Количество эпох:", epochs_containeer);
    epochs_num->setStyleSheet("font-family: 'Inter'; font-size: 15pt; border: none;");
    epochs_num->setMaximumWidth(200);

    QLineEdit* epochs_input = new QLineEdit(epochs_containeer);
    epochs_input->setMaximumSize(200, 50);
    epochs_input->setStyleSheet("font-family: 'Inter'; font-size: 15pt; background-color: #F5F5DC; border-radius: 5px;");
    epochs_input->setValidator(new QIntValidator(1, 1000000, epochs_input));

    layout_epochs->addWidget(epochs_lbl, 0, 0, 1, 2, Qt::AlignHCenter | Qt::AlignTop);
    layout_epochs->addWidget(epochs_num, 1, 0, Qt::AlignRight | Qt::AlignVCenter);
    layout_epochs->addWidget(epochs_input, 1, 1, Qt::AlignLeft | Qt::AlignVCenter);
    layout_epochs->setContentsMargins(15, 2, 15, 15);

    // Панель данных
    QWidget *data_containeer = new QWidget(this);
    data_containeer->setFixedSize(405, 185);
    data_containeer->setStyleSheet("background-color: #F4DCB0; border: 1px solid black; border-radius: 40px; padding: 15px;");
    QGridLayout *layout_data = new QGridLayout(data_containeer);

    QLabel* data_lbl = new QLabel("Данные", data_containeer);
    data_lbl->setFixedSize(130, 100);
    data_lbl->setStyleSheet("font-family: 'Inter'; font-size: 20pt; border: none;");
    data_lbl->setAlignment(Qt::AlignHCenter | Qt::AlignTop);

    QLabel* data_num_train = new QLabel("Доля обучения:", data_containeer);
    data_num_train->setStyleSheet("font-family: 'Inter'; font-size: 15pt; border: none;");
    data_num_train->setMaximumWidth(200);

    QLineEdit* data_input_train = new QLineEdit(data_containeer);
    data_input_train->setMaximumSize(200, 50);
    data_input_train->setStyleSheet("font-family: 'Inter'; font-size: 15pt; background-color: #F5F5DC; border-radius: 5px;");
    data_input_train->setValidator(new QIntValidator(0, 100, data_input_train));

    QLabel* data_num_valid = new QLabel("Доля валидации:", data_containeer);
    data_num_valid->setStyleSheet("font-family: 'Inter'; font-size: 15pt; border: none;");
    data_num_valid->setMaximumWidth(200);

    QLineEdit* data_input_valid = new QLineEdit(data_containeer);
    data_input_valid->setMaximumSize(200, 50);
    data_input_valid->setStyleSheet("font-family: 'Inter'; font-size: 15pt; background-color: #F5F5DC; border-radius: 5px;");
    data_input_valid->setValidator(new QIntValidator(0, 100, data_input_valid));

    connect(data_input_train, &QLineEdit::textChanged, this, [data_input_train, data_input_valid](const QString &text) {
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

    connect(data_input_valid, &QLineEdit::textChanged, this, [data_input_train, data_input_valid](const QString &text) {
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
    loss_containeer->setFixedSize(405, 185);
    loss_containeer->setStyleSheet("background-color: #F4DCB0; border: 1px solid black; border-radius: 40px; padding: 15px;");
    QGridLayout *layout_loss = new QGridLayout(loss_containeer);

    QLabel* loss_lbl = new QLabel("Функция потерь", loss_containeer);
    loss_lbl->setFixedSize(130, 100);
    loss_lbl->setStyleSheet("font-family: 'Inter'; font-size: 20pt; border: none;");
    loss_lbl->setAlignment(Qt::AlignHCenter | Qt::AlignTop);

    QLabel* MSE_lbl = new QLabel("Среднеквадратическая:", loss_containeer);
    MSE_lbl->setStyleSheet("font-family: 'Inter'; font-size: 15pt; border: none;");
    MSE_lbl->setMaximumWidth(200);

    QLineEdit* MSE_loss = new QLineEdit(loss_containeer);
    MSE_loss->setMaximumSize(200, 50);
    MSE_loss->setStyleSheet("font-family: 'Inter'; font-size: 15pt; background-color: #F5F5DC; border-radius: 5px;");
    MSE_loss->setValidator(new QIntValidator(0, 100, MSE_loss));

    QLabel* MAE_lbl = new QLabel("Средняя абсолютная:", loss_containeer);
    MAE_lbl->setStyleSheet("font-family: 'Inter'; font-size: 15pt; border: none;");
    MAE_lbl->setMaximumWidth(200);

    QLineEdit* MAE_loss = new QLineEdit(loss_containeer);
    MAE_loss->setMaximumSize(200, 50);
    MAE_loss->setStyleSheet("font-family: 'Inter'; font-size: 15pt; background-color: #F5F5DC; border-radius: 5px;");
    MAE_loss->setValidator(new QIntValidator(0, 100, MAE_loss));

    QLabel* cross_entropy_lbl = new QLabel("Кросс-энтропия:", loss_containeer);
    cross_entropy_lbl->setStyleSheet("font-family: 'Inter'; font-size: 15pt; border: none;");
    cross_entropy_lbl->setMaximumWidth(200);

    QLineEdit* cross_entropy_loss = new QLineEdit(loss_containeer);
    cross_entropy_loss->setMaximumSize(200, 50);
    cross_entropy_loss->setStyleSheet("font-family: 'Inter'; font-size: 15pt; background-color: #F5F5DC; border-radius: 5px;");
    cross_entropy_loss->setValidator(new QIntValidator(0, 100, cross_entropy_loss));

    layout_loss->addWidget(loss_lbl, 0, 0, 1, 2, Qt::AlignHCenter | Qt::AlignTop);
    layout_loss->addWidget(MSE_lbl, 1, 0, Qt::AlignRight | Qt::AlignVCenter);
    layout_loss->addWidget(MSE_loss, 1, 1, Qt::AlignLeft | Qt::AlignVCenter);
    layout_loss->addWidget(MAE_lbl, 2, 0, Qt::AlignRight | Qt::AlignVCenter);
    layout_loss->addWidget(MAE_loss, 2, 1, Qt::AlignLeft | Qt::AlignVCenter);
    layout_loss->addWidget(cross_entropy_lbl, 2, 0, Qt::AlignRight | Qt::AlignVCenter);
    layout_loss->addWidget(cross_entropy_loss, 2, 1, Qt::AlignLeft | Qt::AlignVCenter);
    layout_loss->setContentsMargins(15, 2, 15, 15);

    // Пройденные эпохи
    QLabel* curr_epochs = new QLabel("Пройдено эпох: 0", this);
    curr_epochs->setStyleSheet("background-color: #E1E0F5; font-family: 'Inter'; font-size: 15pt; border: 1px solid black; border-radius: 15px;");
    curr_epochs->setMinimumSize(405, 40);
    curr_epochs->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);

    // Кнопка Тестировать
    QPushButton* test_model = new QPushButton("Тестировать", this);
    test_model->setStyleSheet("QPushButton { background-color: #C5F3FF; border: 1px solid black;"
                          "font-family: 'Inter'; font-size: 15pt; border-radius: 20px; }"
                          "QPushButton:hover { background-color: #DFE036; }"
                          "QPushButton:pressed { background-color: #AFAAFD; }");
    test_model->setMinimumSize(405, 60);

    // Кнопка Начать/Остановить обучение
    QPushButton* fit_it = new QPushButton("Начать обучение", this);
    fit_it->setStyleSheet("QPushButton { background-color: #CDFFBA; border: 1px solid black;"
                          "font-family: 'Inter'; font-size: 15pt; border-radius: 20px; }"
                          "QPushButton:hover { background-color: #DFE036; }"
                          "QPushButton:pressed { background-color: #AFAAFD; }");
    fit_it->setMinimumSize(405, 60);

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
    layout3->addWidget(epochs_containeer, 0, Qt::AlignHCenter);
    layout3->addWidget(data_containeer, 0, Qt::AlignHCenter);
    layout3->addWidget(loss_containeer, 0, Qt::AlignHCenter);
    layout3->addWidget(curr_epochs, 0, Qt::AlignHCenter);
    layout3->addWidget(test_model, 0, Qt::AlignHCenter);
    layout3->addWidget(fit_it, 0, Qt::AlignHCenter);
    layout3->addStretch();

    // Контейнер для поля
    auto* btn_scheme = new QPushButton("Модель", this);
    btn_scheme->setStyleSheet("QPushButton { background-color: #FFD4AA; border-top-left-radius: 20px; padding: 0px; font-family: 'Inter'; font-size: 16pt;"
                              "border-top-right-radius: 0px; border-bottom-left-radius: 0px; border-bottom-right-radius: 0px; border: 2px solid black;"
                              "border-bottom: none; } QPushButton:hover { background-color: #DFE036; }"
                              "QPushButton:pressed { background-color: #AFAAFD; }");
    btn_scheme->setFixedSize(175, 55 * scale);
    connect(btn_scheme, &QPushButton::clicked, this, [this](){
        this->view->setCurrentIndex(0);
    });

    auto* btn_training = new QPushButton("Обучение", this);
    btn_training->setStyleSheet("QPushButton { background-color: #BFEBC1; border-top-left-radius: 0px; padding: 0px; font-family: 'Inter'; font-size: 16pt;"
                                "border-top-right-radius: 20px; border-bottom-left-radius: 0px; border-bottom-right-radius: 0px; border: 2px solid black;"
                                "border-bottom: none; border-left: none; } QPushButton:hover { background-color: #DFE036; }"
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
