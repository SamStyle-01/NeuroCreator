#ifndef TRAINING_H
#define TRAINING_H

#include "pch.h"

class SamField;
class SamSystem;
class SamView;
class SamChart;

enum class LossFunc {
    MSE,
    MAE,
    CROSSENTROPY
};

class SamTraining : public QFrame {
    Q_OBJECT
    QGridLayout *layout;
    QFrame *field;
    SamSystem *system;
    SamView* view;
    SamChart* chart;
    LossFunc curr_loss;
public:
    explicit SamTraining(SamView *parent, SamSystem *system);

signals:
};

#endif // SAMSCHEME_H
