import cv2
import os


def split_image_into_16_parts(image_path, output_dir="output_parts"):
    # Загружаем изображение
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Не удалось загрузить изображение.")

    # Получаем размеры изображения
    height, width = img.shape[:2]

    # Определяем количество строк и столбцов (4х4)
    rows, cols = 4, 4
    cell_width = width // cols
    cell_height = height // rows

    # Создаем папку для сохранения, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    part_number = 0
    # Перебираем строки (сверху вниз)
    for row in range(rows):
        # Для каждой строки перебираем столбцы в обратном порядке (справа налево)
        for col in reversed(range(cols)):
            x_start = col * cell_width
            y_start = row * cell_height
            piece = img[y_start:y_start + cell_height, x_start:x_start + cell_width]
            part_number += 1
            output_file = os.path.join(output_dir, f"part_{part_number}.jpg")
            cv2.imwrite(output_file, piece)
            print(f"Сохранено: {output_file}")


if __name__ == '__main__':
    # Укажите путь к вашему изображению
    split_image_into_16_parts("extract2d_plywood.jpg")
