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

    QLineEdit* epochs_input;
public:
    explicit SamTraining(SamView *parent, SamSystem *system);

    int get_epochs() const;
    void set_epochs(int epochs);
    LossFunc get_loss_func() const;
signals:
};

#endif // SAMSCHEME_H
