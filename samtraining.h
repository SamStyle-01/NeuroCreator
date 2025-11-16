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
    LossFunc curr_loss;
    QLabel* curr_epochs;
    QLineEdit* data_input_train;
    QLineEdit* lr_input;
    QLineEdit* batch_size_input;

    QVector<float> train_series;
    QVector<float> valid_series;

    QLineEdit* chart_left_bound;
    QLineEdit* chart_right_bound;
    int left_bound;
    int right_bound;

    SamChart *chartView;

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
    void add_loss(float train_loss, float valid_loss);
    void add_loss(float train_loss);

    void update_chart(int first_epoch, int last_epoch);
signals:
};

#endif // SAMSCHEME_H
