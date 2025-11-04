#ifndef TRAINING_H
#define TRAINING_H

#include "pch.h"

class SamField;
class SamSystem;
class SamView;
class SamChart;

class SamTraining : public QFrame {
    Q_OBJECT
    QGridLayout *layout;
    QFrame *field;
    SamSystem *system;
    SamView* view;
    SamChart* chart;
public:
    explicit SamTraining(SamView *parent, SamSystem *system);

signals:
};

#endif // SAMSCHEME_H
