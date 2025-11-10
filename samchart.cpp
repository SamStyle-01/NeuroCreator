#include "samchart.h"

SamChart::SamChart(QWidget *parent, SamSystem *system) : QChartView(parent) {
    this->system = system;
    this->setDragMode(QGraphicsView::DragMode::NoDrag);
    this->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    this->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    this->setStyleSheet("border: none;");

    train.append(1, 12);
    train.append(2, 18);
    train.append(3, 21);
    train.append(4, 20);
    train.append(5, 20);
    train.append(6, 16);
    train.append(7, 18);
    train.append(8, 15);

    train.setName("Тренировочный набор данных");
    data.addSeries(&train);

    valid.append(1, 10);
    valid.append(2, 16);
    valid.append(3, 18);
    valid.append(4, 16);
    valid.append(5, 13);
    valid.append(6, 11);
    valid.append(7, 10);
    valid.append(8, 9);

    valid.setName("Валидационный набор данных");
    data.addSeries(&valid);

    auto *axisX = new QValueAxis();
    auto *axisY = new QValueAxis();

    qreal minX = 1;
    qreal maxX = 8;

    axisX->setRange(minX, maxX);
    axisY->setRange(0, 25);

    axisX->setTitleText("Эпохи");
    axisY->setTitleText("Функция потерь");

    axisX->setLabelFormat("%d");

    data.setAxisX(axisX, &train);
    data.setAxisY(axisY, &train);

    data.setAxisX(axisX, &valid);
    data.setAxisY(axisY, &valid);

    data.setMargins(QMargins(0, 0, 0, 0));
    data.layout()->setContentsMargins(0, 0, 0, 0);

    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    this->setContentsMargins(0, 0, 0, 0);

    this->setChart(&data);
    this->setRenderHint(QPainter::Antialiasing);
}

void SamChart::wheelEvent(QWheelEvent *event) {
    event->accept();
}

