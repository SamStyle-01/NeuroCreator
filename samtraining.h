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
    QLabel* curr_epochs;
    QLineEdit* data_input_train;
    QLineEdit* lr_input;
    QLineEdit* batch_size_input;

    QLineEdit* epochs_input;
public:
    explicit SamTraining(SamView *parent, SamSystem *system);

    int get_epochs() const;
    void set_epochs(int epochs);
    void set_epochs_view(int epochs);
    LossFunc get_loss_func() const;
    int get_train_share() const;
    float get_learning_rate() const;
    int get_batch_size() const;
signals:
};

#endif // SAMSCHEME_H
