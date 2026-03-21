import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List  # noqa: F401

def heatmap(df: pd.DataFrame, username: str) -> None:
    """Создаёт тепловую карту корреляции только для числовых столбцов"""
    try:
        # Выбираем только числовые столбцы для корреляции
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        correlation_df = df[numeric_cols].corr()
        
        # Создаём тепловую карту
        plt.figure(figsize=(20, 15))
        sns.set(font_scale=1.5)
        k = sns.heatmap(correlation_df, annot=True, square=False)
        output_path = os.path.join('player_data', username, "corr_heatmap.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        k.get_figure().savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()  # Закрываем фигуру для освобождения памяти
        
    except Exception as e:
        print(f"Ошибка при создании тепловой карты: {str(e)}")

def driver_fn(username: str) -> None:
    """Обрабатывает данные и создаёт тепловую карту"""
    try:
        # Читаем датасет
        df = pd.read_csv(os.path.join('player_data', username, 'chess_dataset.csv'))
        
        # Добавляем разницу рейтингов
        df["rating_difference"] = df["player_rating"] - df["opponent_rating"]
        
        # Преобразуем результаты партий в числовые значения
        result_map = {
            "win": 1.0,
            "agreed": 0.5, 
            "timevsinsufficient": 0.5,
            "insufficient": 0.5,
            "stalemate": 0.5,
            "repetition": 0.5,
            "resigned": 0.0,
            "checkmated": 0.0,
            "timeout": 0.0,
            "abandoned": 0.0
        }
        
        df['result_val_for_player'] = df['result_for_player'].map(result_map)
        
        # Создаём тепловую карту
        heatmap(df, username)
        
    except Exception as e:
        print(f"Ошибка в основной функции: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        driver_fn(sys.argv[1])
    else:
        print("Укажите имя пользователя в качестве аргумента")