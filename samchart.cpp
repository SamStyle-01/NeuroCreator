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

    marker.setMarkerSize(10);
    marker.setColor(Qt::black);
    data.addSeries(&marker);
    auto markers = data.legend()->markers(&marker);
    for (QLegendMarker *lm : markers) {
        lm->setVisible(false);
    }

    connect(&train, &QLineSeries::clicked, this, [this](const QPointF &p){
        this->showPointInfo(p, train.points());
    });
    connect(&valid, &QLineSeries::clicked, this, [this](const QPointF &p){
        this->showPointInfo(p, valid.points());
    });

    axisX = new QValueAxis(this);
    axisY = new QValueAxis(this);

    axisX->setRange(1, 1);
    axisY->setRange(0, 25);

    axisX->setTickCount(10);
    axisY->setTickCount(25);

    this->max_y = 0;

    axisX->setTitleText("Эпохи");
    axisY->setTitleText("Функция потерь");

    axisX->setLabelFormat("%d");

    data.setAxisX(axisX, &marker);
    data.setAxisY(axisY, &marker);

    data.setAxisX(axisX, &train);
    data.setAxisY(axisY, &train);

    data.setAxisX(axisX, &valid);
    data.setAxisY(axisY, &valid);
    data.setMargins(QMargins(0, 0, 0, 0));
    data.layout()->setContentsMargins(0, 0, 0, 0);

    tooltip = new QGraphicsTextItem();
    tooltip->setZValue(9999);
    tooltip->setDefaultTextColor(Qt::white);
    tooltip->setVisible(false);

    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    this->setContentsMargins(0, 0, 0, 0);

    this->setChart(&data);
    this->setRenderHint(QPainter::Antialiasing);

    auto bg = new QGraphicsRectItem();
    bg->setBrush(QColor(0, 0, 0, 180));
    bg->setPen(Qt::NoPen);
    bg->setZValue(9998);
    bg->setVisible(false);

    data.scene()->addItem(bg);
    data.scene()->addItem(tooltip);

    tooltipBackground = bg;
}

void SamChart::wheelEvent(QWheelEvent *event) {
    event->accept();
}

void SamChart::hideTooltip() {
    tooltip->setVisible(false);
    tooltipBackground->setVisible(false);
}

void SamChart::mousePressEvent(QMouseEvent *event) {
    QPointF clickedValue = this->chart()->mapToValue(event->pos());

    bool hitTrain = false;
    bool hitValid = false;

    for (const QPointF &p : train.points()) {
        if (qAbs(p.x() - clickedValue.x()) < 0.3) {
            hitTrain = true;
            break;
        }
    }

    for (const QPointF &p : valid.points()) {
        if (qAbs(p.x() - clickedValue.x()) < 0.3) {
            hitValid = true;
            break;
        }
    }

    if (!hitTrain && !hitValid) {
        marker.clear();
        hideTooltip();
    }

    QChartView::mousePressEvent(event);
}

void SamChart::reset_marker() {
    marker.clear();
    hideTooltip();
}

void SamChart::showPointInfo(const QPointF &p, QList<QPointF> cont) {
    int x = std::round(p.x()) - cont[0].x() + 1;
    QPointF p_real = cont[x - 1];

    QString text = QString("Эпоха = %1\nПотеря = %2")
                       .arg(p_real.x())
                       .arg(p_real.y());

    tooltip->setPlainText(text);

    QPoint localPos = this->mapFromGlobal(QCursor::pos());

    QRectF tipRect = tooltip->boundingRect();
    int offsetX = 15;
    int offsetY = -tipRect.height() - 10;

    QPoint tooltipPos = localPos + QPoint(offsetX, offsetY);

    if (tooltipPos.x() + tipRect.width() > this->width()) {
        tooltipPos.setX(localPos.x() - tipRect.width() - 15);
    }

    if (tooltipPos.y() < 0) {
        tooltipPos.setY(localPos.y() + 15);
    }

    tooltip->setPos(this->chart()->mapToScene(tooltipPos));

    tooltipBackground->setRect(tipRect.adjusted(-5, -5, 5, 5));
    tooltipBackground->setPos(tooltip->pos());

    tooltip->setVisible(true);
    tooltipBackground->setVisible(true);

    marker.clear();
    marker.append(p_real);
}

void SamChart::add_loss(float train_loss, float val_loss, int curr_epoch, int begin) {
    this->train.append(curr_epoch, train_loss);
    this->valid.append(curr_epoch, val_loss);

    axisX->setRange(begin, std::max(curr_epoch, 1));
    auto pt_t = train.points().back();
    auto pt_v = valid.points().back();

    if (pt_t.y() > this->max_y) {
        max_y = pt_t.y();
    }
    if (pt_v.y() > this->max_y) {
        max_y = pt_v.y();
    }

    axisY->setRange(0, max_y);
}

void SamChart::add_loss(float train_loss, int curr_epoch, int begin) {
    this->train.append(curr_epoch, (double)train_loss);

    axisX->setRange(begin, std::max(curr_epoch, 1));

    auto pt_t = train.points().back();

    if (pt_t.y() > this->max_y) {
        max_y = pt_t.y();
    }

    axisY->setRange(0, max_y * 1.03);
}

void SamChart::add_loss_lite(float train_loss, float val_loss, int curr_epoch) {
    this->train.append(curr_epoch, train_loss);
    this->valid.append(curr_epoch, val_loss);
}

void SamChart::add_loss_lite(float train_loss, int curr_epoch) {
    this->train.append(curr_epoch, (double)train_loss);
}

void SamChart::update_range(int curr_epoch, int begin) {
    axisX->setRange(begin, std::max(curr_epoch, 1));

    float max = -1;
    auto pts_t = train.points();
    auto pts_v = valid.points();

    for (int i = 0; i < train.count(); i++) {
        auto p_t = pts_t[i];
        if (p_t.y() > max) {
            max = p_t.y();
        }
    }
    if (valid.count()) {
        for (int i = 0; i < valid.count(); i++){
            QPointF p_v = pts_v[i];
            if (p_v.y() > max) {
                max = p_v.y();
            }
        }
    }

    max_y = max;

    axisY->setRange(0, max * 1.03);
}

void SamChart::clear_losses() {
    this->train.clear();
    this->valid.clear();
    this->max_y = 0;
}

void SamChart::set_range(int first, int last) {
    this->axisX->setRange(first, last);
}

void SamChart::set_y_range(float lowest, float highest) {
    this->axisY->setRange(lowest, highest);
}
