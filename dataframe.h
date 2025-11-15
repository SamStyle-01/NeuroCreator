#ifndef DATAFRAME_H
#define DATAFRAME_H

#include "pch.h"

class SamView;

class DataFrame {
    SamView* main_window;
    QVector<QVector<float>> data;
    int num_rows;
    int num_cols;
public:
    DataFrame(SamView* main_window);

    int count_lines_in_file(const QString &filePath) const;
    bool load_data(QString path, bool is_main);
    bool z_score(int num_x);
    static float get_mean(const QVector<float>& data);
    static float get_std(const QVector<float>& data, float mean);
    int get_rows() const;
    int get_cols() const;

    void random_shuffle();
    QPair<DataFrame*, DataFrame*> train_test_split(int percent);

    QVector<QVector<float>>& get_data();
    void append_row(QVector<float> row);
};


#endif // DATAFRAME_H
