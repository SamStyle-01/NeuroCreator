#include "sambuttonsgroup.h"

extern double scale;

SamButtonsGroup::SamButtonsGroup(QWidget *parent) : QFrame(parent) {
    // Общий стиль группы (рамка, закругление, фон)
    this->setStyleSheet(
        "QFrame { background-color: #FFFFF0; border: 2px solid black; border-top-left-radius: 20px;"
        "border-top-right-radius: 20px; border-bottom-left-radius: 0px; border-bottom-right-radius: 0px; padding: 0px; }"
        );
    this->setFixedWidth(350 * (scale + (1 - scale) / 2));
    layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    this->setLayout(layout);
}

void SamButtonsGroup::setLabel(QLabel *label, QString color) {
    this->label = label;
    this->label->setStyleSheet("background-color: " + color + "; border-top-left-radius: 20px; padding: 0px; font-family: 'Inter'; font-size: 16pt;"
                               "border-top-right-radius: 20px; border-bottom-left-radius: 0px; border-bottom-right-radius: 0px; border: none;");
    this->label->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
    this->layout->addWidget(this->label);
    this->label->setMinimumHeight(42 * scale);
}

void SamButtonsGroup::addBtn(QPushButton* btn, std::function<void()> fn) {
    btns.push_back(btn);
    btn->setParent(this);

    btn->setStyleSheet(
        "QPushButton {"
        "  background-color: #F5F5DC;"
        "  border: none;"
        "  border-top: 1px solid black;"
        "  padding: 0px;"
        "  font-family: 'Inter';"
        "  font-size: 14pt;"
        "  border-radius: 0px;"
        "}"
        "QPushButton:hover {"
        "  background-color: #DFE036;"
        "}"
        "QPushButton:pressed {"
        "  background-color: #AFAAFD;"
        "}"
        );
    btn->setMinimumHeight(35 * (scale + (1 - scale) / 2));

    btn->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    layout->addWidget(btn);
    this->setFixedHeight((this->btns.size() * 35 + 42) * (scale + (1 - scale) / 2));

    connect(btn, &QPushButton::clicked, this, fn);
}

void SamButtonsGroup::addStretch() {
    this->layout->addStretch();
}
