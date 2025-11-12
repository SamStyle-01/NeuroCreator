#include "dataframe.h"
#include "samview.h"
#include <cmath>

DataFrame::DataFrame(SamView* main_window) {
    this->main_window = main_window;
    num_rows = 0;
    num_cols = 0;
}

int DataFrame::count_lines_in_file(const QString &filePath) const {
    QFile file(filePath);

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::critical(this->main_window, "Ошибка", "Не удалось открыть файл для чтения: " + filePath);
        return -1;
    }

    QTextStream in(&file);
    int lineCount = 0;

    while (!in.atEnd()) {
        in.readLine();
        lineCount++;
    }

    file.close();

    return lineCount;
}

QVector<QVector<float>>& DataFrame::get_data() {
    return this->data;
}

bool DataFrame::load_data(QString path, bool is_main) {
    num_rows = count_lines_in_file(path);
    if (num_rows == -1) {
        QMessageBox::critical(this->main_window, "Ошибка", "Файл содержит некорректные данные: " + path);
        return false;
    }

    QFile file(path);

    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        QMessageBox::critical(this->main_window, "Ошибка", "Не удалось открыть файл для чтения: " + path);
        return false;
    }

    QTextStream in(&file);
    if (!in.atEnd()) {
        QStringList lst = in.readLine().split(",", Qt::SkipEmptyParts);
        num_cols = lst.size();
        if (is_main) {
            if (num_cols > 1) {
                for (int i = 0; i < num_cols; i++) {
                    data.emplace_back(QVector<float>());
                    data[i].reserve(num_rows);
                    data[i].emplaceBack(lst[i].toFloat());
                }
            }
            else {
                QMessageBox::critical(this->main_window, "Ошибка", "Файл содержит некорректные данные: " + path);
                return false;
            }
        }
        else {
            if (num_cols >= 1) {
                for (int i = 0; i < num_cols; i++) {
                    data.emplace_back(QVector<float>());
                    data[i].reserve(num_rows);
                    data[i].emplaceBack(lst[i].toFloat());
                }
            }
            else {
                QMessageBox::critical(this->main_window, "Ошибка", "Файл содержит некорректные данные: " + path);
                return false;
            }
        }
    }
    while (!in.atEnd()) {
        QString line = in.readLine();
        QStringList lst = line.split(",", Qt::SkipEmptyParts);
        if (num_cols) {
            for (int i = 0; i < num_cols; i++) {
                data[i].emplaceBack(lst[i].toFloat());
            }
        }
    }

    // for (int i = 0; i < 15; i++)
    //     qDebug() << data[0][num_rows - 1 - i] << " " << data[1][num_rows - 1 - i] << " " << data[2][i];
    // qDebug() << num_rows << " " << num_cols;

    file.close();
    return true;

    // QVector<float> vect {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    // qDebug() << get_std(vect, get_mean(vect));
}

bool DataFrame::z_score(int num_x) {
    if (!data.size()) {
        QMessageBox::critical(this->main_window, "Ошибка", "Данные отсутствуют");
        return false;
    }
    for (int i = 0; i < num_x; i++) {
        float mean = get_mean(data[i]);
        float std = get_std(data[i], mean);

        for (int j = 0; j < data[i].size(); j++) {
            data[i][j] = (data[i][j] - mean) / std;
        }
    }

    // for (int i = 0; i < 15; i++)
    //     qDebug() << data[0][num_rows - 1 - i] << " " << data[1][num_rows - 1 - i] << " " << data[2][i];
    // qDebug() << num_rows << " " << num_cols;

    return true;
}

float DataFrame::get_mean(const QVector<float>& data) {
    if (!data.size()) return 0;
    float result = data[0];
    for (int i = 1; i < data.size(); i++) {
        result += (data[i] - result) / (i + 1);
    }
    return result;
}

float DataFrame::get_std(const QVector<float>& data, float mean) {
    float squares = 0;
    for (int i = 0; i < data.size(); i++) {
        squares += pow(data[i] - mean, 2);
    }
    squares /= data.size();
    squares = sqrt(squares);
    return squares;
}

int DataFrame::get_rows() const {
    return this->num_rows;
}

int DataFrame::get_cols() const {
    return this->num_cols;
}
