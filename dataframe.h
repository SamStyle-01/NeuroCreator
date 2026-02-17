#ifndef DATAFRAME_H
#define DATAFRAME_H

#include "pch.h"

class SamView;

class DataFrame {
    SamView* main_window;
    QVector<QVector<float>> data;
    int num_rows;
    int num_cols;

    QVector<float> mean;
    QVector<float> std;
public:
    DataFrame(SamView* main_window);

    int count_lines_in_file(const QString &filePath) const;
    bool load_data(QString path, bool is_main);
    bool load_data(QString data);
    bool z_score(int num_x);
    bool z_score(int num_x, QVector<float> mean, QVector<float> std);
    static float get_mean(const QVector<float>& data);
    static float get_std(const QVector<float>& data, float mean);
    int get_rows() const;
    int get_cols() const;

    void random_shuffle();
    QPair<DataFrame*, DataFrame*> train_test_split(int percent);

    QVector<QVector<float>>& get_data();
    void append_row(QVector<float> row);

    QPair<QVector<float>, QVector<float>> get_mean_std() const;
};


#endif // DATAFRAME_H
