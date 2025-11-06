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
