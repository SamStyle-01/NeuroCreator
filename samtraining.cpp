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
        background-color: #F5EBE0;
        border-radius: 10px;
    }
    QRadioButton::indicator::checked {
        border: 2px solid #222;
        background-color: #7DD961;
        border-radius: 10px;
    }
)";

QString radio_button_style_disabled = R"(
    QWidget { border: none; }
    QRadioButton::indicator {
        width: 20px;
        height: 20px;
    }
    QRadioButton::indicator::unchecked {
        border: 2px solid #555;
        background-color: #AAAAAA;
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

    chart_view = new SamChart(this->field, system);

    QVBoxLayout *layout = new QVBoxLayout(this->field);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(chart_view);

    // Панель эпох
    QWidget *epochs_containeer = new QWidget(this);
    epochs_containeer->setFixedSize(405 * (scale + (1 - scale) / 2), 245);
    epochs_containeer->setStyleSheet("background-color: #F4DCB0; border: 1px solid black; border-radius: 40px; padding: 15px;");
    QGridLayout *layout_epochs = new QGridLayout(epochs_containeer);

    QLabel* epochs_lbl = new QLabel("Настройки обучения", epochs_containeer);
    epochs_lbl->setFixedSize(240, 85);
    epochs_lbl->setStyleSheet("font-family: 'Inter'; font-size: " + QString::number(int(16 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border: none;");

    QLabel* epochs_num = new QLabel("Количество эпох:", epochs_containeer);
    epochs_num->setStyleSheet("font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border: none;");
    epochs_num->setMaximumWidth(200);
    epochs_num->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);

    epochs_input = new QLineEdit("0", epochs_containeer);
    epochs_input->setMaximumSize(200 * (scale + (1 - scale) / 2), 50);
    epochs_input->setStyleSheet("font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; background-color: #F5F5DC; border-radius: 5px;");
    epochs_input->setValidator(new QIntValidator(1, 1000000, epochs_input));

    QLabel* lr_num = new QLabel("Скорость обучения:", epochs_containeer);
    lr_num->setStyleSheet("font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border: none;");
    lr_num->setMaximumWidth(210);
    lr_num->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);

    lr_input = new QLineEdit("0,001", epochs_containeer);
    lr_input->setMaximumSize(200 * (scale + (1 - scale) / 2), 50);
    lr_input->setStyleSheet("font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; background-color: #F5F5DC; border-radius: 5px;");
    lr_input->setValidator(new QDoubleValidator(1, 1000000, 7, epochs_input));

    QLabel* batch_size_num = new QLabel("Размер батча:", epochs_containeer);
    batch_size_num->setStyleSheet("font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border: none;");
    batch_size_num->setMaximumWidth(200);
    batch_size_num->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Preferred);

    batch_size_input = new QLineEdit("512", epochs_containeer);
    batch_size_input->setMaximumSize(200 * (scale + (1 - scale) / 2), 50);
    batch_size_input->setStyleSheet("font-family: 'Inter'; font-size: "+ QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; background-color: #F5F5DC; border-radius: 5px;");
    batch_size_input->setValidator(new QIntValidator(1, 10000000, epochs_input));

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
    data_lbl->setStyleSheet("font-family: 'Inter'; font-size: " + QString::number(int(16 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border: none;");
    data_lbl->setAlignment(Qt::AlignHCenter | Qt::AlignTop);

    QLabel* data_num_train = new QLabel("Доля обучения:", data_containeer);
    data_num_train->setStyleSheet("font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border: none;");
    data_num_train->setMaximumWidth(200);

    data_input_train = new QLineEdit("100", data_containeer);
    data_input_train->setMaximumSize(200 * (scale + (1 - scale) / 2), 50);
    data_input_train->setStyleSheet("font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; background-color: #F5F5DC; border-radius: 5px;");
    data_input_train->setValidator(new QIntValidator(0, 100, data_input_train));

    QLabel* data_num_valid = new QLabel("Доля валидации:", data_containeer);
    data_num_valid->setStyleSheet("font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border: none;");
    data_num_valid->setMaximumWidth(200);

    data_input_valid = new QLineEdit("0", data_containeer);
    data_input_valid->setMaximumSize(200 * (scale + (1 - scale) / 2), 50);
    data_input_valid->setStyleSheet("font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; background-color: #F5F5DC; border-radius: 5px;");
    data_input_valid->setValidator(new QIntValidator(0, 100, data_input_valid));

    connect(data_input_train, &QLineEdit::textChanged, this, [this](const QString &text) {
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

    connect(data_input_valid, &QLineEdit::textChanged, this, [this](const QString &text) {
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
    loss_containeer->setFixedSize(405 * (scale + (1 - scale) / 2), 272);
    loss_containeer->setStyleSheet("background-color: #F4DCB0; border: 1px solid black; border-radius: 40px; padding: 15px;");
    QGridLayout *layout_loss = new QGridLayout(loss_containeer);
    layout_loss->setSpacing(0);
    QButtonGroup* losses = new QButtonGroup(loss_containeer);
    connect(losses, &QButtonGroup::idClicked, this, [this](int id){
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
            case 4:
                this->curr_loss = LossFunc::B_CROSSENTROPY;
                break;
            default:
                QMessageBox::critical(this, "Ошибка", "Ошибка выбора функции потерь");
        }
    });

    QLabel* loss_lbl = new QLabel("Функция потерь", loss_containeer);
    loss_lbl->setFixedSize(230, 75);
    loss_lbl->setStyleSheet("font-family: 'Inter'; font-size: " + QString::number(int(16 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border: none;");
    loss_lbl->setAlignment(Qt::AlignHCenter | Qt::AlignTop);

    QLabel* MSE_lbl = new QLabel("MSE:", loss_containeer);
    MSE_lbl->setStyleSheet("font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border: none;");

    MSE_loss = new QRadioButton(loss_containeer);
    MSE_loss->setMinimumSize(80, 20);
    MSE_loss->setStyleSheet(radio_button_style);

    MSE_loss->setChecked(true);
    this->curr_loss = LossFunc::MSE;
    losses->addButton(MSE_loss);
    losses->setId(MSE_loss, 1);

    QLabel* MAE_lbl = new QLabel("MAE:", loss_containeer);
    MAE_lbl->setStyleSheet("font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border: none;");

    MAE_loss = new QRadioButton(loss_containeer);
    MAE_loss->setMinimumSize(80, 20);
    MAE_loss->setStyleSheet(radio_button_style);
    losses->addButton(MAE_loss);
    losses->setId(MAE_loss, 2);

    QLabel* cross_entropy_lbl = new QLabel("CrossEntropy:", loss_containeer);
    cross_entropy_lbl->setStyleSheet("font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border: none;");

    cross_entropy_loss = new QRadioButton(loss_containeer);
    cross_entropy_loss->setMinimumSize(80, 20);
    cross_entropy_loss->setStyleSheet(radio_button_style);
    losses->addButton(cross_entropy_loss);
    losses->setId(cross_entropy_loss, 3);

    QLabel* bce_lbl = new QLabel("BCE:", loss_containeer);
    bce_lbl->setStyleSheet("font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border: none;");

    bce_loss = new QRadioButton(loss_containeer);
    bce_loss->setMinimumSize(80, 20);
    bce_loss->setStyleSheet(radio_button_style);
    losses->addButton(bce_loss);
    losses->setId(bce_loss, 4);

    layout_loss->addWidget(loss_lbl, 0, 0, 1, 2, Qt::AlignHCenter | Qt::AlignTop);
    layout_loss->addWidget(MSE_lbl, 1, 0, Qt::AlignRight | Qt::AlignVCenter);
    layout_loss->addWidget(MSE_loss, 1, 1, Qt::AlignLeft | Qt::AlignVCenter);
    layout_loss->addWidget(MAE_lbl, 2, 0, Qt::AlignRight | Qt::AlignVCenter);
    layout_loss->addWidget(MAE_loss, 2, 1, Qt::AlignLeft | Qt::AlignVCenter);
    layout_loss->addWidget(cross_entropy_lbl, 3, 0, Qt::AlignRight | Qt::AlignVCenter);
    layout_loss->addWidget(cross_entropy_loss, 3, 1, Qt::AlignLeft | Qt::AlignVCenter);
    layout_loss->addWidget(bce_lbl, 4, 0, Qt::AlignRight | Qt::AlignVCenter);
    layout_loss->addWidget(bce_loss, 4, 1, Qt::AlignLeft | Qt::AlignVCenter);
    layout_loss->setContentsMargins(15, 2, 15, 10);

    // Диапазон графика
    QWidget *chart_range_containeer = new QWidget(this);
    chart_range_containeer->setFixedSize(405 * (scale + (1 - scale) / 2), 130);
    chart_range_containeer->setStyleSheet("background-color: #F4DCB0; border: 1px solid black; border-radius: 40px; padding: 15px;");
    QGridLayout *chart_range_epochs = new QGridLayout(chart_range_containeer);

    this->left_bound = 0;
    this->right_bound = 0;

    QLabel* chart_range_lbl = new QLabel("Диапазон эпох", chart_range_containeer);
    chart_range_lbl->setFixedSize(220, 85);
    chart_range_lbl->setAlignment(Qt::AlignHCenter);
    chart_range_lbl->setStyleSheet("font-family: 'Inter'; font-size: " + QString::number(int(16 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border: none;");

    chart_left_bound = new QLineEdit(chart_range_containeer);
    chart_left_bound->setMaximumSize(150 * (scale + (1 - scale) / 2), 50);
    chart_left_bound->setStyleSheet("font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; background-color: #F5F5DC; border-radius: 5px;");
    chart_left_bound->setAlignment(Qt::AlignHCenter);
    chart_left_bound->setPlaceholderText("-");

    chart_right_bound = new QLineEdit(chart_range_containeer);
    chart_right_bound->setMaximumSize(150 * (scale + (1 - scale) / 2), 50);
    chart_right_bound->setStyleSheet("font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; background-color: #F5F5DC; border-radius: 5px;");
    chart_right_bound->setAlignment(Qt::AlignHCenter);
    chart_right_bound->setPlaceholderText("-");

    connect(chart_left_bound, &QLineEdit::editingFinished, this, [this]() {
        if (!this->system->get_epochs()) {
            this->chart_left_bound->setText("");
        }
        else {
            if (this->chart_left_bound->text() == "") left_bound = 1;
            else {
                QString filtered;
                for (QChar c : this->chart_left_bound->text()) {
                    if (c.isDigit()) {
                        filtered.append(c);
                    } else {
                        break;
                    }
                }
                left_bound = filtered.toInt();
                this->chart_left_bound->setText(QString::number(left_bound));
                if (left_bound < 1) {
                    left_bound = 1;
                    this->chart_left_bound->setText(QString::number(left_bound));
                }
                if (left_bound > right_bound) {
                    left_bound = right_bound;
                    this->chart_left_bound->setText(QString::number(left_bound));
                }
            }
            this->chart_view->reset_marker();
            this->update_chart(left_bound, right_bound);
        }
    });

    connect(chart_right_bound, &QLineEdit::editingFinished, this, [this]() {
        if (!this->system->get_epochs()) {
            this->chart_right_bound->setText("");
        }
        else {
            if (this->chart_right_bound->text() == "") right_bound = this->train_series.size();
            else {
                QString filtered;
                for (QChar c : this->chart_right_bound->text()) {
                    if (c.isDigit()) {
                        filtered.append(c);
                    } else {
                        break;
                    }
                }
                right_bound = filtered.toInt();
                this->chart_right_bound->setText(QString::number(right_bound));
                if (this->train_series.size() < right_bound) {
                    right_bound = this->train_series.size();
                    this->chart_right_bound->setText(QString::number(right_bound));
                }
                if (left_bound > right_bound) {
                    right_bound = left_bound;
                    this->chart_right_bound->setText(QString::number(right_bound));
                }
            }
            this->chart_view->reset_marker();
            this->update_chart(left_bound, right_bound);
        }
    });

    QLabel* space = new QLabel(":", this);
    space->setStyleSheet("font-family: 'Inter'; font-size: 16pt; border: none;");
    space->setFixedSize(40, 50);

    chart_range_epochs->addWidget(chart_range_lbl, 0, 0, 1, 3, Qt::AlignHCenter | Qt::AlignTop);
    chart_range_epochs->addWidget(chart_left_bound, 1, 0, Qt::AlignRight | Qt::AlignVCenter);
    chart_range_epochs->addWidget(space, 1, 1, Qt::AlignHCenter | Qt::AlignVCenter);
    chart_range_epochs->addWidget(chart_right_bound, 1, 2, Qt::AlignLeft| Qt::AlignVCenter);
    chart_range_epochs->setContentsMargins(25, 2, 25, 15);

    // Пройденные эпохи
    curr_epochs = new QLabel("Пройдено эпох: 0", this);
    curr_epochs->setStyleSheet("background-color: #E1E0F5; font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt;"
                               "border: 1px solid black; border-radius: 15px;");
    curr_epochs->setMinimumSize(405 * (scale + (1 - scale) / 2), 40);
    curr_epochs->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);

    // Кнопка Тестировать
    test_model = new QPushButton("Тестировать", this);
    test_model->setStyleSheet("QPushButton { background-color: #C5F3FF; border: 1px solid black;"
                          "font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border-radius: 20px; }"
                          "QPushButton:hover { background-color: #DFE036; }"
                          "QPushButton:pressed { background-color: #AFAAFD; }");
    test_model->setMinimumSize(405 * (scale + (1 - scale) / 2), 60);
    connect(test_model, &QPushButton::clicked, this, [this](){
        this->system->test_data();
    });

    // Кнопка Загрузить лучшую модель
    load_best_model = new QPushButton("Загрузить лучшую модель", this);
    load_best_model->setStyleSheet("QPushButton { background-color: #E1E0F5; border: 1px solid black;"
                              "font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border-radius: 20px; }"
                              "QPushButton:hover { background-color: #DFE036; }"
                              "QPushButton:pressed { background-color: #AFAAFD; }");
    load_best_model->setMinimumSize(405 * (scale + (1 - scale) / 2), 60);
    connect(load_best_model, &QPushButton::clicked, this, [this](){
        if (this->system->get_epochs()) {
            this->system->set_best_model();
            int best_epoch = this->system->get_best_epoch();
            this->system->set_curr_epochs(best_epoch + 1);
            this->set_epochs_view(best_epoch + 1);
            this->train_series.remove(best_epoch + 1, train_series.size() - best_epoch - 1);
            if (!this->valid_series.empty()) this->valid_series.remove(best_epoch + 1, valid_series.size() - best_epoch - 1);
            right_bound = train_series.size();
            left_bound = 1;
            this->update_chart(left_bound, right_bound);
            this->chart_left_bound->setText("");
            this->chart_right_bound->setText("");
            this->system->set_is_training(false);

            QMessageBox::information(this, "Успех", "Лучшая модель была загружена");
        }
        else {
            QMessageBox::warning(this, "Ошибка", "Обучения ещё не было");
            return;
        }
    });

    // Кнопка Начать/Остановить обучение
    fit_it = new QPushButton("Начать обучение", this);
    fit_it->setStyleSheet("QPushButton { background-color: #CDFFBA; border: 1px solid black;"
                          "font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border-radius: 20px; }"
                          "QPushButton:hover { background-color: #DFE036; }"
                          "QPushButton:pressed { background-color: #AFAAFD; }");
    fit_it->setMinimumSize(405 * (scale + (1 - scale) / 2), 60);
    connect(fit_it, &QPushButton::clicked, this, [this](){
        if (data_input_train->text() == "" || data_input_valid->text() == ""
            || lr_input->text() == "" || epochs_input->text() == ""
            || batch_size_input->text() == "") {
            QMessageBox::warning(this, "Ошибка", "Заполнены не все параметры обучения");
            return;
        }
        if (!this->system->get_is_training()) {
            load_best_model->setStyleSheet("QPushButton { background-color: #CDFFBA; border: 1px solid black;"
                              "font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border-radius: 20px; }"
                              "QPushButton:hover { background-color: #DFE036; }"
                              "QPushButton:pressed { background-color: #AFAAFD; }");
            this->system->backpropagation();
            this->system->set_is_training(true);
            fit_it->setText("Остановить обучение");
            fit_it->setStyleSheet("QPushButton { background-color: #C5F5FC; border: 1px solid black;"
                                  "font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border-radius: 20px; }"
                                  "QPushButton:hover { background-color: #DFE036; }"
                                  "QPushButton:pressed { background-color: #AFAAFD; }");

            if (chart_right_bound->text() != "") {
                chart_right_bound->setText("");
                right_bound = this->train_series.size();
            }
            this->MSE_loss->setEnabled(false);
            this->MAE_loss->setEnabled(false);
            this->cross_entropy_loss->setEnabled(false);
            this->bce_loss->setEnabled(false);
            this->chart_left_bound->setEnabled(false);
            this->chart_right_bound->setEnabled(false);
            this->data_input_train->setEnabled(false);
            this->data_input_valid->setEnabled(false);
            this->test_model->setEnabled(false);
            this->btn_scheme->setEnabled(false);
            this->load_best_model->setEnabled(false);
        }
        else {
            this->system->set_is_training(false);
            fit_it->setText("Начать обучение");
            fit_it->setStyleSheet("QPushButton { background-color: #CDFFBA; border: 1px solid black;"
                                  "font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border-radius: 20px; }"
                                  "QPushButton:hover { background-color: #DFE036; }"
                                  "QPushButton:pressed { background-color: #AFAAFD; }");

            this->chart_left_bound->setEnabled(true);
            this->chart_right_bound->setEnabled(true);
            this->test_model->setEnabled(true);
            this->btn_scheme->setEnabled(true);
            this->load_best_model->setEnabled(true);
        }
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
    layout3->addWidget(chart_range_containeer, 0, Qt::AlignHCenter);
    layout3->addWidget(curr_epochs, 0, Qt::AlignHCenter);
    layout3->addWidget(load_best_model, 0, Qt::AlignHCenter);
    layout3->addWidget(test_model, 0, Qt::AlignHCenter);
    layout3->addWidget(fit_it, 0, Qt::AlignHCenter);
    layout3->addStretch();

    // Контейнер для поля
    btn_scheme = new QPushButton("Модель", this);
    btn_scheme->setStyleSheet("QPushButton { background-color: #FFD4AA; border-top-left-radius: 20px; "
                              "padding: 0px; font-family: 'Inter'; font-size: " + QString::number(int(16 * (scale + (1 - scale) / 2) * 10) / 10) + "pt;"
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
                                "padding: 0px; font-family: 'Inter'; font-size: " + QString::number(int(16 * (scale + (1 - scale) / 2) * 10) / 10) + "pt;"
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

QVector<QRadioButton*> SamTraining::get_btns() {
    return QVector<QRadioButton*> {this->MSE_loss, this->MAE_loss, this->cross_entropy_loss, this->bce_loss};
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

void SamTraining::add_loss(float train_loss, float valid_loss) {
    this->train_series.push_back(train_loss);
    this->valid_series.push_back(valid_loss);
    if (this->chart_left_bound->text() != "")
        this->chart_view->add_loss(train_loss, valid_loss, train_series.size(), left_bound);
    else
        this->chart_view->add_loss(train_loss, valid_loss, train_series.size(), 1);
    right_bound++;
}

void SamTraining::add_loss(float train_loss) {
    this->train_series.push_back(train_loss);
    if (this->chart_left_bound->text() != "")
        this->chart_view->add_loss(train_loss, train_series.size(), left_bound);
    else
        this->chart_view->add_loss(train_loss, train_series.size(), 1);
    right_bound++;
}

void SamTraining::update_chart(int first_epoch, int last_epoch) {
    this->chart_view->clear_losses();
    if (!this->valid_series.empty()) {
        for (int i = first_epoch - 1; i < last_epoch; i++) {
            this->chart_view->add_loss(this->train_series[i], this->valid_series[i], i + 1, first_epoch - 1);
        }
        this->chart_view->set_range(first_epoch, last_epoch);
    }
    else {
        for (int i = first_epoch - 1; i < last_epoch; i++) {
            this->chart_view->add_loss(this->train_series[i], i + 1, first_epoch - 1);
        }
        this->chart_view->set_range(first_epoch, last_epoch);
    }
}

void SamTraining::reset_series() {
    this->train_series.clear();
    this->valid_series.clear();
    this->chart_view->reset_marker();
    this->chart_view->clear_losses();
}

void SamTraining::reset_state() {
    this->system->set_is_training(false);
    fit_it->setText("Начать обучение");
    fit_it->setStyleSheet("QPushButton { background-color: #CDFFBA; border: 1px solid black;"
                          "font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border-radius: 20px; }"
                                                                                             "QPushButton:hover { background-color: #DFE036; }"
                                                                                             "QPushButton:pressed { background-color: #AFAAFD; }");

    this->chart_left_bound->setEnabled(true);
    this->chart_right_bound->setEnabled(true);
    this->data_input_train->setEnabled(true);
    this->data_input_valid->setEnabled(true);

    this->MSE_loss->setEnabled(true);
    this->MSE_loss->setStyleSheet(radio_button_style);

    this->MAE_loss->setEnabled(true);
    this->MAE_loss->setStyleSheet(radio_button_style);

    this->bce_loss->setEnabled(true);
    this->bce_loss->setStyleSheet(radio_button_style);

    this->cross_entropy_loss->setEnabled(true);
    this->cross_entropy_loss->setStyleSheet(radio_button_style);

    this->chart_view->set_y_range(0, 25);
    this->chart_left_bound->setText("");
    this->chart_right_bound->setText("");
    this->left_bound = 0;
    this->right_bound = 0;
}

void SamTraining::training_done() {
    this->system->set_is_training(false);
    fit_it->setText("Начать обучение");
    fit_it->setStyleSheet("QPushButton { background-color: #CDFFBA; border: 1px solid black;"
                          "font-family: 'Inter'; font-size: " + QString::number(int(14 * (scale + (1 - scale) / 2) * 10) / 10) + "pt; border-radius: 20px; }"
                          "QPushButton:hover { background-color: #DFE036; }"
                          "QPushButton:pressed { background-color: #AFAAFD; }");

    this->chart_left_bound->setEnabled(true);
    this->chart_right_bound->setEnabled(true);
    this->test_model->setEnabled(true);
    this->btn_scheme->setEnabled(true);
}
