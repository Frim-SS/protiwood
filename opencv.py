import cv2
import numpy as np


def extract_plywood(image_path, output_path="output.jpg"):
    # Загружаем изображение в оттенках серого для обработки
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Загружаем цветное изображение для финального результата
    color_image = cv2.imread(image_path)

    if gray is None or color_image is None:
        raise ValueError("Не удалось загрузить изображение.")

    # Шаг 1. Предварительная обработка: размытие для уменьшения шумов
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Шаг 2. Бинаризация с использованием порога Оцу
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Шаг 3. Морфологическая обработка для устранения разрывов в краях
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Шаг 4. Детектор краёв Canny для точного определения границ
    edges = cv2.Canny(closed, 50, 150)

    # Шаг 5. Поиск контуров
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Контуры не найдены.")

    # Выбираем самый крупный контур (предполагаем, что это лист фанеры)
    largest_contour = max(contours, key=cv2.contourArea)

    # Шаг 6. Вычисляем минимальный поворотный прямоугольник
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.int32)  # Используем np.int32 вместо np.int0

    # Получаем угол поворота (если угол меньше -45, корректируем его)
    angle = rect[-1]
    if angle < -45:
        angle += 90

    # Шаг 7. Выравнивание изображения по горизонту
    center = tuple(map(int, rect[0]))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(color_image, M, (color_image.shape[1], color_image.shape[0]))

    # Преобразуем координаты коробки согласно повороту
    pts = np.array(box, dtype=np.float32)
    pts = np.concatenate([pts, np.ones((4, 1), dtype=np.float32)], axis=1)  # однородные координаты
    rotated_pts = M.dot(pts.T).T
    rotated_pts = np.array(rotated_pts, dtype=np.int32)  # Используем np.int32 вместо np.int0

    # Шаг 8. Кадрирование: находим ограничивающий прямоугольник по преобразованным точкам
    x, y, w, h = cv2.boundingRect(rotated_pts)
    cropped = rotated[y:y + h, x:x + w]

    # Сохраняем результат
    cv2.imwrite(output_path, cropped)
    return output_path


# Пример использования:
if __name__ == '__main__':
    extract_plywood("gg.jpg", "extract2d_plywood.jpg")
