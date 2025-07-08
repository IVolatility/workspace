import argparse
import json
import numpy as np
import os
import pandas as pd
import sys
import time
import torch

from paddleocr import PaddleOCR
from pandas.io.formats.style import Styler
from PIL import Image
from surya.table_rec import TableRecPredictor
from typing import Any, Dict, List, Optional, Tuple

#from openpyxl import Workbook
#PaddleOCR(use_angle_cls=True, lang='en', show_log=False, det=False).__dict__
#TableRecPredictor().__dict__



def load_ocr_model() -> PaddleOCR:
    """
    Загружает и возвращает модель OCR.

    :return: Экземпляр модели PaddleOCR.
    """
    print("Загрузка модели OCR...")
    #return PaddleOCR(use_angle_cls=True, lang='en', show_log=False, rec_algorithm='CRNN') # show_log=False, max_batch_size = 20, total_process_num = os.cpu_count() * 2 - 1 , rec_batch_num=24) #, enable_mkldnn =True)
    return PaddleOCR(use_angle_cls=True, lang='en', show_log=False, det=False) #lang='ch', 'ru', '

def load_image(image_path: str) -> Image.Image:
    """
    Загружает изображение по указанному пути.

    :param image_path: Путь к изображению.
    :return: Объект изображения Image.Image.
    :raises ValueError: Если изображение не удалось загрузить.
    """
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Ошибка при загрузке изображения: {e}")
    
def extract_text_from_bbox(
    cell: Dict[str, Any], 
    image: Image.Image, 
    ocr: PaddleOCR
) -> Optional[Tuple[str, float]]:
    """
    Извлекает текст и confidence из области изображения, ограниченной bbox ячейки.

    :param cell: Ячейка таблицы с координатами.
    :param image: Изображение, содержащее таблицу.
    :param ocr: Загруженная OCR модель.
    :return: Текст и уверенность или None.
    """
    try:
        x1, y1, x2, y2 = map(int, cell['bbox'])
        cropped = image.crop((x1, y1, x2, y2))

        # Масштабирование изображения
        resize_factor = 2.0
        new_width = int(cropped.width * resize_factor)
        new_height = int(cropped.height * resize_factor)

        resized = cropped.resize((new_width, new_height), Image.LANCZOS)
        resized_np = np.array(resized)

        #result = ocr.ocr(resized_np, cls=True)
        result = ocr(resized_np)
        #print(result)
        if result and result[0]:
            return result[1][0]
            #return result[0][0][1] # (текст, confidence)
    except Exception as e:
        print(f"Ошибка при извлечении текста из ячейки: {e}")
    return None

def get_row_text(
    row_id: int, 
    cells: List[Dict[str, Any]],
    cell_map: Dict[Tuple[int, int], Tuple[str, float]], 
    image: Image.Image, 
    ocr: PaddleOCR
) -> Tuple[List[str], List[float]]:
    """
    Извлекает текст и уровень уверенности всех ячеек строки.

    :param row_id: Идентификатор строки.
    :param cells: Список всех ячеек таблицы.
    :param cell_map: Кэш ячеек с rowspan.
    :param image: Исходное изображение таблицы.
    :param ocr: OCR модель.
    :return: Списки текста и confidence по колонкам строки.
    """
    row_cells = [cell for cell in cells if cell['row_id'] == row_id]
    row_cells.sort(key=lambda c: c['col_id'])

    row_text, confidences = [], []

    col_pointer = 0

    for cell in row_cells:
        col_id = cell['col_id']
        colspan = cell.get('colspan')
        rowspan = cell.get('rowspan')

        if col_pointer < col_id:
            updated_cell = cell_map.get((row_id, col_pointer), ("", np.nan))  
            row_text.extend([updated_cell[0]] * colspan)
            confidences.extend([updated_cell[1]] * colspan)

            #row_text.append("")
            #confidences.append(np.nan)
            col_pointer += 1
        
        result  = extract_text_from_bbox(cell, image, ocr)
        text, confidence = result if result else ("", np.nan)

        row_text.extend([text] * colspan)
        confidences.extend([confidence] * colspan)

        # Запоминаем для будущих строк (rowspan > 1)
        if rowspan > 1:
            for r_offset in range(1, rowspan):
                for c_offset in range(colspan):
                    cell_map[(row_id + r_offset, col_id + c_offset)] = text, confidence

        col_pointer += colspan

    return row_text, confidences

def parse_table(
    table_predictions: Dict[str, Any],
    image: Image.Image,
    ocr: PaddleOCR
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Преобразует структуру таблицы в два DataFrame: текст и уверенность.

    :param table_predictions: Словарь с предсказаниями от TableRecPredictor.
    :param image: Исходное изображение таблицы.
    :param ocr: OCR модель для распознавания текста в ячейках.
    :return: Кортеж из DataFrame (текст, уверенность).
    :raises ValueError:  При отсутствии строк или ячеек.
    """
    cells = table_predictions.get('cells')
    rows = table_predictions.get('rows')

    if not cells or not rows:
        raise ValueError("В таблице отсутствуют строки или ячейки.")

    cell_map = {} #для rowspan > 1
    all_row_ids = sorted({row['row_id'] for row in rows})

    result = [get_row_text(rid, cells, cell_map, image, ocr) for rid in all_row_ids]
    data = [r[0] for r in result]
    data_confidence = [r[1] for r in result]

    #data = [get_row_text(rid, cells, cell_map, image, ocr)[0] for rid in all_row_ids]
    #data_confidence = [get_row_text(rid, cells, cell_map, image, ocr)[1] for rid in all_row_ids]
    
    return pd.DataFrame(data), pd.DataFrame(data_confidence)


def process_images(
    file_paths: List[str], 
    ocr: PaddleOCR, 
    table_rec_predictor: TableRecPredictor
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Обрабатывает изображения таблиц и возвращает их содержимое в виде DataFrame.

    :param file_paths: Пути к изображениям.
    :param ocr: OCR модель для распознавания текста.
    :param table_rec_predictor: Модель для предсказания структуры таблицы.
    :return: Кортеж из двух DataFrame или None при ошибке.
    """
    df_combined, df_conf_combined = pd.DataFrame(), pd.DataFrame()

    for i, file_path in enumerate(file_paths, start=1):
        try:
            print(f"[{i}] Загрузка изображения...")
            image = load_image(file_path)
            print(f"[{i}] Распознавание структуры таблицы...")
            table_predictions = table_rec_predictor([image])

            if not table_predictions:
                print(f"[{i}] Не удалось распознать структуру таблицы.")
                return None

            table_predictions = table_predictions[0].dict()
            df, df_conf  = parse_table(table_predictions, image, ocr)

            df_combined = pd.concat([df_combined, df], ignore_index=True)
            df_conf_combined = pd.concat([df_conf_combined, df_conf], ignore_index=True)

            print(f"[{i}] Успешно обработано!")
        except Exception as e:
            print(f"[{i}] Ошибка: {e}")

    return (df_combined, df_conf_combined) if not df_combined.empty else None

def style_confidence_cells(
    df: pd.DataFrame, 
    df_conf: pd.DataFrame, 
    threshold: float = 95.0
) -> Optional[Styler]:
    """
    Подсвечивает ячейки в DataFrame, если уровень уверенности ниже порога.

    :param df: Основной DataFrame с данными.
    :param df_conf: DataFrame с уровнями уверенности для каждой ячейки.
    :param threshold: Порог уверенности (по умолчанию 90).
    :return: Стилизованный DataFrame с подсветкой или None, если размеры не совпадают.
    """
    if df.shape != df_conf.shape:
        print("Ошибка: Размеры df и df_conf не совпадают.")
        return None

    def highlight(data, conf):
        styles = pd.DataFrame('', index=data.index, columns=data.columns)
        condition = ((conf * 100) < threshold)
        styles[condition] = 'background-color: #ffe6e6'
        return styles

    return df.style.apply(highlight, conf=df_conf, axis=None)

"""
def convert_to_structured_df(df: pd.DataFrame) -> pd.DataFrame:
    
    #Преобразует DataFrame с помощью OpenAI в структурированный 
    #DataFrame с заголовками и индексами.

    #:param df: Исходный DataFrame, полученный из таблицы.
    #:return: Структурированный DataFrame.


    json_df = openai_client.process_json_to_json(df.to_json(orient='table'))
    json_df = json_df.strip('```json\n').strip('```')
    json_df = json.loads(json_df)

    df_temp = df.copy()

    header_rows = len(json_df["row_index_levels"]) 
    index_cols = len(json_df["column_index_levels"])
    print(header_rows)
    print(index_cols)
    # Разделим нужные части один раз
    data = df_temp.iloc[header_rows:, index_cols:].values
    col_block = df_temp.iloc[:header_rows, index_cols:].values if header_rows else None
    idx_block = df_temp.iloc[header_rows:, :index_cols].values if index_cols else None
    corner_block = df_temp.iloc[:header_rows, :index_cols].values if header_rows and index_cols else None

    # Инициализация
    columns = None
    index = None
    data_corner = None

    # Обработка в зависимости от формы
    if header_rows > 1 and index_cols > 1:
        columns = pd.MultiIndex.from_tuples(list(zip(*col_block)))
        index = pd.MultiIndex.from_tuples(list(zip(*idx_block.T)))
        data_corner = corner_block[0, :]

    elif header_rows == 1 and index_cols > 1:
        columns = col_block.flatten()
        index = pd.MultiIndex.from_tuples(list(zip(*idx_block.T)))
        data_corner = corner_block[0, :]

    elif header_rows > 1 and index_cols == 1:
        columns = pd.MultiIndex.from_tuples(list(zip(*col_block)))
        index = idx_block.flatten()
        data_corner = corner_block[0, :]
        
    elif header_rows == 1 and index_cols == 1:
        columns = col_block
        index = idx_block
        data_corner = corner_block[0, :]
        
    elif header_rows > 1 and index_cols == 0:
        columns = pd.MultiIndex.from_tuples(list(zip(*col_block)))
        index = range(data.shape[0])  # Без индексных колонок — создаём RangeIndex

    elif header_rows == 1 and index_cols == 0:
        columns = col_block.flatten()
        index = range(data.shape[0])

    elif header_rows == 0 and index_cols > 1:
        columns = df_temp.columns[index_cols:]
        index = pd.MultiIndex.from_tuples(list(zip(*idx_block.T)))

    elif header_rows == 0 and index_cols == 1:
        columns = df_temp.columns[index_cols:]
        index = idx_block.flatten()

    elif header_rows == 0 and index_cols == 0:
        columns = df_temp.columns
        index = df_temp.index

    # Финальный датафрейм
    df_final = pd.DataFrame(data, index=index, columns=columns)

    if data_corner is not None:
        df_final.index.names = data_corner

    return df_final
"""

def list_of_strings(arg):
    return arg.split(',')

def handle_task(args, threshold: float = 95.0) -> None:
    #file_paths = [
    #    os.path.join(input_path, f)
    #    for f in os.listdir(input_path)
    #    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    #]
    file_paths = args.infiles
   
    #raise ValueError(f"Вход: {args}")
    #raise ValueError(f"Вход: {file_paths}")
    file_paths.sort(key=os.path.getctime)

    if not file_paths:
        print("Ошибка: не переданы изображения.")
        return None
    
    print(f"Задача получена. Обработка {len(file_paths)} изображений...")

    ocr = load_ocr_model()
    table_rec_predictor = TableRecPredictor()

    start_time = time.time()
    result = process_images(file_paths, ocr, table_rec_predictor)
    end_time = time.time()
    print(f"Время выполнения: {end_time - start_time:.4f} секунд")

    

    if result is not None:
        df, df_confidence = result
        res = style_confidence_cells(df, df_confidence)
        df.to_csv(f"reconciliation/out/{args.taskid}.csv", index=False, header=False)
        #res.to_excel(f"reconciliation/out/{args.taskid}.xlsx", engine="openpyxl", index=False, header=False)
        #res.to_excel("/home/chetverikovm/screenshots/output.xlsx", engine="openpyxl", index=False, header=False)
        #return result
    else:
        print("Не удалось обработать изображения.")
        return None

if __name__ == "__main__":
    #df, df_confidence = handle_task(input_path)
    #input_path = "/home/chetverikovm/screenshots"
    parser = argparse.ArgumentParser()
    #parser.add_argument('--arg', type=str)
    parser.add_argument('--infiles', type=list_of_strings, required=True)
    parser.add_argument('--taskid', type=str, required=True)
    args = parser.parse_args()

    #wb = Workbook()
    #ws = wb.active
    #ws["A1"] = f"Вход: {args}"
    #wb.save(f"reconciliation/out/{args.taskid}.xlsx")

    #with open(f"reconciliation/out/{args.taskid}.txt", "w", encoding="utf-8") as f:
    #    f.write(f"Вход: {args}")

    handle_task(args)
    #print(df)

#df.to_json(orient='table')



#res = style_confidence_cells(df, df_confidence)
#res.to_excel("D:\\IVolatility\\OCR\\styled_output.xlsx", engine="openpyxl")

#if __name__ == "__main__":
#    print("Task Worker запущен...")

#    input_path = "D:\\IVolatility\\OCR\\screenshots\\"
#    output_path = "D:\\IVolatility\\OCR\\"
#    confidence_threshold = 90  # В процентах
#    handle_task(input_path, threshold=confidence_threshold)
    #handle_task(input_path, output_path)