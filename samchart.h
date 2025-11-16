#ifndef SAMCHART_H
#define SAMCHART_H

#include "pch.h"

class SamSystem;

class SamChart : public QChartView {
    Q_OBJECT
    SamSystem *system;
    QChart data;
    QLineSeries train;
    QLineSeries valid;
    QValueAxis* axisX;
    QValueAxis* axisY;
protected:
    void wheelEvent(QWheelEvent *event);
public:
    SamChart(QWidget *parent, SamSystem *system);
    void add_loss(float train_loss, float val_loss, int curr_epoch);
    void add_loss(float train_loss, int curr_epoch);
    void clear_losses();
    void set_range(int first, int last);
};

#endif // SAMCHART_H
