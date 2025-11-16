#include "samchart.h"

SamChart::SamChart(QWidget *parent, SamSystem *system) : QChartView(parent) {
    this->system = system;
    this->setDragMode(QGraphicsView::DragMode::NoDrag);
    this->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    this->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    this->setStyleSheet("border: none;");

    train.setName("Тренировочный набор данных");
    data.addSeries(&train);

    valid.setName("Валидационный набор данных");
    data.addSeries(&valid);

    axisX = new QValueAxis(this);
    axisY = new QValueAxis(this);

    axisX->setRange(1, 1);
    axisY->setRange(0, 25);

    axisX->setTickCount(10);
    axisY->setTickCount(25);

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

void SamChart::add_loss(float train_loss, float val_loss, int curr_epoch) {
    this->train.append(curr_epoch, train_loss);
    this->valid.append(curr_epoch, val_loss);

    axisX->setRange(1, std::max(curr_epoch, 1));
    float max = -1;
    auto pts_t = train.points();
    auto pts_v = valid.points();

    for (int i = 0; i < train.count(); i++) {
        auto p_t = pts_t[i];
        if (p_t.y() > max) {
            max = p_t.y();
        }
        QPointF p_v = pts_v[i];
        if (p_v.y() > max) {
            max = p_v.y();
        }
    }
    axisY->setRange(0, max);
}

void SamChart::add_loss(float train_loss, int curr_epoch) {
    this->train.append(curr_epoch, train_loss);

    axisX->setRange(1, std::max(curr_epoch, 1));
    float max = -1;
    auto pts = train.points();

    for (int i = 0; i < train.count(); i++) {
        auto p_t = pts[i];
        if (p_t.y() > max) {
            max = p_t.y();
        }
    }
    axisY->setRange(0, max);
}

void SamChart::clear_losses() {
    this->train.clear();
    this->valid.clear();
}

void SamChart::set_range(int first, int last) {
    this->axisX->setRange(first, last);
}
