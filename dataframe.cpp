#include "dataframe.h"
#include "samview.h"
#include <cmath>
#include <random>

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

bool DataFrame::load_data(QString path, bool is_main, int main_num_cols) {
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

    if (in.atEnd()) {
        QMessageBox::critical(this->main_window,
                              "Ошибка",
                              "Файл пустой: " + path);
        return false;
    }

    QString first_line = in.readLine().trimmed();

    if (first_line.isEmpty()) {
        QMessageBox::critical(this->main_window,
                              "Ошибка",
                              "Файл содержит пустую первую строку: " + path);
        return false;
    }

    QStringList lst = first_line.split(",", Qt::KeepEmptyParts);
    num_cols = lst.size();
    if (!is_main && num_cols != main_num_cols) {
        QMessageBox::critical(this->main_window,
                              "Ошибка",
                              "Файл содержит неверное количество столбцов: " + path);
        return false;
    }

    if ((is_main && num_cols <= 1) || (!is_main && num_cols < 1)) {
        QMessageBox::critical(this->main_window,
                              "Ошибка",
                              "Файл содержит некорректные данные: " + path);
        return false;
    }

    data.clear();
    data.resize(num_cols);

    for (int i = 0; i < num_cols; i++) {

        QString value = lst[i].trimmed();

        if (value.isEmpty()) {
            QMessageBox::critical(this->main_window,
                                  "Ошибка",
                                  "Обнаружен пропуск значения в первой строке.");
            return false;
        }

        bool ok = false;
        float number = value.toFloat(&ok);

        if (!ok) {
            QMessageBox::critical(this->main_window,
                                  "Ошибка",
                                  "Нечисловое значение: " + value +
                                      "\nФайл: " + path);
            return false;
        }

        data[i].reserve(num_rows);
        data[i].emplaceBack(number);
    }

    int row = 2;

    while (!in.atEnd()) {

        QString line = in.readLine().trimmed();

        if (line.isEmpty()) {
            QMessageBox::critical(this->main_window,
                                  "Ошибка",
                                  "Обнаружена пустая строка (строка "
                                      + QString::number(row) + ").");
            return false;
        }

        QStringList values = line.split(",", Qt::KeepEmptyParts);

        if (values.size() != num_cols) {
            QMessageBox::critical(this->main_window,
                                  "Ошибка",
                                  "Разное количество столбцов (строка "
                                      + QString::number(row) + ").");
            return false;
        }

        for (int col = 0; col < num_cols; col++) {

            QString value = values[col].trimmed();

            if (value.isEmpty()) {
                QMessageBox::critical(this->main_window,
                                      "Ошибка",
                                      "Пропуск значения (строка "
                                          + QString::number(row) + ").");
                return false;
            }

            bool ok = false;
            float number = value.toFloat(&ok);

            if (!ok) {
                QMessageBox::critical(this->main_window,
                                      "Ошибка",
                                      "Нечисловое значение: " + value +
                                          "\nСтрока: " + QString::number(row));
                return false;
            }

            data[col].emplaceBack(number);
        }

        row++;
    }

    // for (int i = 0; i < 15; i++)
    //     qDebug() << data[0][num_rows - 1 - i] << " " << data[1][num_rows - 1 - i] << " " << data[2][i];
    // qDebug() << num_rows << " " << num_cols;

    file.close();
    return true;

    // QVector<float> vect {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    // qDebug() << get_std(vect, get_mean(vect));
}

bool DataFrame::load_data(QString data_string, int main_num_cols) {
    if (data_string == "") {
        QMessageBox::critical(this->main_window, "Ошибка", "Поле ввода содержит некорректные данные.");
        return false;
    }

    QStringList lines = data_string.split('\n', Qt::SkipEmptyParts);

    if (lines.isEmpty()) {
        QMessageBox::critical(this->main_window,
                              "Ошибка",
                              "Поле ввода пустое.");
        return false;
    }

    QStringList first_row = lines[0].trimmed().split(",", Qt::KeepEmptyParts);
    num_cols = first_row.size();
    if (num_cols != main_num_cols) {
        QMessageBox::critical(this->main_window,
                              "Ошибка",
                              "Поле ввода содержит некорректное количество столбцов.");
        return false;
    }

    if (num_cols < 1) {
        QMessageBox::critical(this->main_window,
                              "Ошибка",
                              "Поле ввода содержит некорректные данные.");
        return false;
    }

    data.clear();
    data.resize(num_cols);

    for (int row = 0; row < lines.size(); row++) {

        QString line = lines[row].trimmed();

        if (line.isEmpty()) {
            QMessageBox::critical(this->main_window,
                                  "Ошибка",
                                  "Обнаружена пустая строка.");
            return false;
        }

        QStringList lst = line.split(",", Qt::KeepEmptyParts);

        if (lst.size() != num_cols) {
            QMessageBox::critical(this->main_window,
                                  "Ошибка",
                                  "Разное количество столбцов в строках.");
            return false;
        }

        for (int col = 0; col < num_cols; col++) {

            QString value = lst[col].trimmed();

            if (value.isEmpty()) {
                QMessageBox::critical(this->main_window,
                                      "Ошибка",
                                      "Обнаружен пропуск значения (строка "
                                          + QString::number(row + 1) + ").");
                return false;
            }

            bool ok = false;
            float number = value.toFloat(&ok);

            if (!ok) {
                QMessageBox::critical(this->main_window,
                                      "Ошибка",
                                      "Нечисловое значение: " + value +
                                          "\nСтрока: " + QString::number(row + 1));
                return false;
            }

            data[col].emplaceBack(number);
        }
    }

    this->num_rows = data[0].size();

    return true;
}

bool DataFrame::z_score(int num_x) {
    if (!data.size()) {
        QMessageBox::critical(this->main_window, "Ошибка", "Данные отсутствуют");
        return false;
    }
    this->mean = QVector<float>(num_x);
    this->std = QVector<float>(num_x);
    for (int i = 0; i < num_x; i++) {
        this->mean[i] = get_mean(data[i]);
        this->std[i] = get_std(data[i], this->mean[i]);

        for (int j = 0; j < data[i].size(); j++) {
            data[i][j] = (data[i][j] - this->mean[i]) / this->std[i];
        }
    }

    return true;
}

bool DataFrame::z_score(int num_x, QVector<float> mean, QVector<float> std) {
    if (!data.size()) {
        QMessageBox::critical(this->main_window, "Ошибка", "Данные отсутствуют");
        return false;
    }
    for (int i = 0; i < num_x; i++) {
        for (int j = 0; j < data[i].size(); j++) {
            data[i][j] = (data[i][j] - mean[i]) / std[i];
        }
    }

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

void DataFrame::random_shuffle() {
    int rows = this->get_rows();
    if (rows <= 1) return;

    static std::random_device rd;
    static std::mt19937 gen(rd());

    for(int i = rows - 1; i > 0; --i) {
        std::uniform_int_distribution<int> dist(0, i);
        int j = dist(gen);

        if (i != j) {
            for (int col = 0; col < this->get_cols(); ++col) {
                std::swap(this->data[col][i], this->data[col][j]);
            }
        }
    }
}

int DataFrame::get_rows() const {
    return this->num_rows;
}

int DataFrame::get_cols() const {
    return this->num_cols;
}

void DataFrame::append_row(QVector<float> row) {
    if (this->data.empty()) {
        this->num_cols = row.size();
        this->data = QVector<QVector<float>>(row.size());
    }
    for (int i = 0; i < row.size(); i++) {
        this->data[i].push_back(row[i]);
    }
    this->num_rows++;
}

QPair<DataFrame*, DataFrame*> DataFrame::train_test_split(int percent) {
    DataFrame* train = new DataFrame(main_window);
    DataFrame* test = new DataFrame(main_window);

    int bound = (int)((float)this->get_rows() * (float)percent / 100.0f);
    if (!bound) bound = 1;
    QVector<float> row;
    row.reserve(this->get_cols());
    for (int i = 0; i < bound; i++) {
        row.clear();
        for (int j = 0; j < this->get_cols(); j++) {
            row.push_back(this->data[j][i]);
        }
        train->append_row(row);
    }

    for (int i = bound; i < this->get_rows(); i++) {
        row.clear();
        for (int j = 0; j < this->get_cols(); j++) {
            row.push_back(this->data[j][i]);
        }
        test->append_row(row);
    }

    return qMakePair(train, test);
}

QPair<QVector<float>, QVector<float>> DataFrame::get_mean_std() const {
    return qMakePair(this->mean, this->std);
}
