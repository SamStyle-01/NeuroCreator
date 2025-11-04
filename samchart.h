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
protected:
    void wheelEvent(QWheelEvent *event);
public:
    SamChart(QWidget *parent, SamSystem *system);
};

#endif // SAMCHART_H
