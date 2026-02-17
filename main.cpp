#include <QApplication>
#include <QMainWindow>
#include <QWidget>

#include "samview.h"
#include "samscheme.h"
#include "samtraining.h"
#include "samsystem.h"

double scale;

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    auto const rec = QGuiApplication::primaryScreen()->size();
    scale = (double)rec.width() / 2560.0f;

    app.setStyleSheet("QWidget { color: black; } QMessageBox { background-color: #F0F0F0; }"
                      "QMessageBox QLabel { background-color: transparent; color: #333333;}"
                      "QMessageBox QPushButton { background-color: #E1E1E1; border: 1px solid #AAAAAA;"
                      "padding: 5px 10px; border-radius: 4px; color: #333333; }"
                      "QDialog { background-color: #F0F0F0; }");

    srand(static_cast<unsigned>(time(0)));

    app.setApplicationVersion("1.0");
    app.setOrganizationName("SamStyle-01");

    SamView* view = new SamView();
    SamSystem* system = new SamSystem(view);
    auto *scheme = new SamScheme(view, system);
    auto *training = new SamTraining(view, system);
    system->set_view(training, scheme);
    view->init(scheme, training, system);

    view->show();

    app.setApplicationName("Конструктор Нейросетей");

    return app.exec();
}
