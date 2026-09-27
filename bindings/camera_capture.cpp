#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

// OpenCV maneja la webcam y pybind11 permite llamar esta función desde Python
#include <opencv2/opencv.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

// Devuelve la foto como un arreglo de píxeles RGB (alto x ancho x 3)
py::array_t<std::uint8_t> capture(int index, bool preview, int warmup_frames) {
    if (index < 0 || warmup_frames < 0) {
        throw std::invalid_argument("El índice y los fotogramas de calentamiento deben ser >= 0.");
    }

    // En Linux probamos V4L2 primero. Si falla, OpenCV elige otro backend
    cv::VideoCapture camera(index, cv::CAP_V4L2);
    if (!camera.isOpened()) {
        camera.open(index);
    }
    if (!camera.isOpened()) {
        throw std::runtime_error("No se pudo abrir la cámara " + std::to_string(index)
                                 + ". Comprueba /dev/video" + std::to_string(index)
                                 + " y que ninguna otra aplicación la esté usando.");
    }

    const std::string window = "MobileNetV2 | ESPACIO: capturar | Q/ESC: salir";
    bool window_open = false;
    cv::Mat frame;

    try {
        // Descartamos los primeros cuadros para dar tiempo al ajuste de exposición
        cv::Mat discard;
        for (int i = 0; i < warmup_frames; ++i) {
            camera.read(discard);
        }

        if (preview) {
            // La ventana muestra lo que ve la cámara hasta que se tome la foto
            cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
            window_open = true;
            while (true) {
                if (!camera.read(frame) || frame.empty()) {
                    throw std::runtime_error("No se pudo leer un fotograma de la cámara.");
                }
                cv::imshow(window, frame);
                const int key = cv::waitKey(30) & 0xff;

                // Espacio o Enter conservan el cuadro que acabamos de mostrar
                if (key == 32 || key == 13) {
                    frame = frame.clone();
                    break;
                }

                // También se puede salir con Q, Esc o cerrando la ventana
                if (key == 'q' || key == 'Q' || key == 27
                    || cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1) {
                    throw std::runtime_error("Captura cancelada.");
                }
            }
        } else if (!camera.read(frame) || frame.empty()) {
            // En modo automático avisamos si falló la lectura del único cuadro
            throw std::runtime_error("No se pudo leer un fotograma de la cámara.");
        }
    } catch (...) {
        // Si ocurre un error o se cancela, liberamos la cámara antes de salir
        camera.release();
        if (window_open) {
            try { cv::destroyWindow(window); } catch (const cv::Exception&) {}
        }
        throw;
    }

    // Ya tenemos la foto; cerramos la cámara y la ventana de vista previa
    camera.release();
    if (window_open) {
        cv::destroyWindow(window);
    }

    // Los píxeles deben venir con tres canales de 8 bits para hacer la conversión
    if (frame.type() != CV_8UC3) {
        throw std::runtime_error("Se esperaba un fotograma BGR uint8 de tres canales.");
    }

    // OpenCV entrega BGR, PIL y el preprocesamiento de MobileNet usan RGB
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    py::array_t<std::uint8_t> pixels({rgb.rows, rgb.cols, 3});
    std::uint8_t* dest = pixels.mutable_data();

    // Copiamos cada fila al arreglo de NumPy que recibirá Python.
    for (int y = 0; y < rgb.rows; ++y) {
        std::memcpy(dest + static_cast<std::size_t>(y) * rgb.cols * 3,
                    rgb.ptr(y), static_cast<std::size_t>(rgb.cols) * 3);
    }
    return pixels;
}

}  // namespace

// Al importar camera_capture en Python, queda disponible camera_capture.capture()
PYBIND11_MODULE(camera_capture, module) {
    module.doc() = "Captura de webcam con OpenCV C++; devuelve RGB uint8 para Python.";
    module.def("capture", &capture, py::arg("index") = 0,
               py::arg("preview") = true, py::arg("warmup_frames") = 15);
}