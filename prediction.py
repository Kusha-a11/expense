import requests
import get_data as gd  # noqa: F401
import json
import time
import os
import pandas as pd  # noqa: F401
import numpy as np  # noqa: F401
import statsmodels.api as sm  # noqa: F401
from sklearn import metrics  # noqa: F401
from sklearn.metrics import confusion_matrix  # noqa: F401
from sklearn.model_selection import train_test_split  # noqa: F401
from sklearn.linear_model import LogisticRegression  # noqa: F401
import mord as mrd  # noqa: F401


def predict(u1, u2):
    """Предсказать исход матча между двумя игроками"""
    di = {}
    headers = {
        'User-Agent': 'Chess Analysis App (Contact: github.com/yogen-ghodke-113)',
        'Accept': 'application/json'
    }

    try:
        # Получить статистику пользователя с логикой повторных попыток
        max_retries = 3
        retry_delay = 2  # секунды

        # Инициализировать рейтинги
        user_rating = None
        opp_rating = None

        # Получить рейтинг пользователя
        for attempt in range(max_retries):
            try:
                req = "https://api.chess.com/pub/player/" + u1 + "/stats"
                response = requests.get(req, headers=headers, timeout=10)
                
                if response.status_code == 429:  # Слишком много запросов
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise Exception("Превышен лимит запросов. Пожалуйста, попробуйте через несколько минут.")
                
                response.raise_for_status()
                user_stats = response.json()
                
                # Попробовать получить рейтинг блица, затем рапида, затем пули
                if "chess_blitz" in user_stats and "last" in user_stats["chess_blitz"]:
                    user_rating = user_stats["chess_blitz"]["last"]["rating"]
                elif "chess_rapid" in user_stats and "last" in user_stats["chess_rapid"]:
                    user_rating = user_stats["chess_rapid"]["last"]["rating"]
                elif "chess_bullet" in user_stats and "last" in user_stats["chess_bullet"]:
                    user_rating = user_stats["chess_bullet"]["last"]["rating"]
                else:
                    raise ValueError(f"Не удалось найти рейтинг для пользователя {u1}")
                break
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise Exception("Время запроса истекло. Пожалуйста, попробуйте снова.")

        # Получить рейтинг оппонента
        for attempt in range(max_retries):
            try:
                req = "https://api.chess.com/pub/player/" + u2 + "/stats"
                response = requests.get(req, headers=headers, timeout=10)
                
                if response.status_code == 429:  # Слишком много запросов
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        raise Exception("Превышен лимит запросов. Пожалуйста, попробуйте через несколько минут.")
                
                response.raise_for_status()
                opp_stats = response.json()
                
                # Попробовать получить рейтинг блица, затем рапида, затем пули
                if "chess_blitz" in opp_stats and "last" in opp_stats["chess_blitz"]:
                    opp_rating = opp_stats["chess_blitz"]["last"]["rating"]
                elif "chess_rapid" in opp_stats and "last" in opp_stats["chess_rapid"]:
                    opp_rating = opp_stats["chess_rapid"]["last"]["rating"]
                elif "chess_bullet" in opp_stats and "last" in opp_stats["chess_bullet"]:
                    opp_rating = opp_stats["chess_bullet"]["last"]["rating"]
                else:
                    raise ValueError(f"Не удалось найти рейтинг для пользователя {u2}")
                break
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise Exception("Время запроса истекло. Пожалуйста, попробуйте снова.")

        if user_rating is None or opp_rating is None:
            raise ValueError("Не удалось получить рейтинги для обоих игроков")

        di["user_rating"] = user_rating
        di["opp_rating"] = opp_rating
        diff = user_rating - opp_rating
        di["rating_diff"] = diff

    except requests.exceptions.RequestException as e:
        if "getaddrinfo failed" in str(e):
            raise Exception("Не удаётся подключиться к Chess.com. Пожалуйста, проверьте интернет-соединение.")
        elif "Name or service not known" in str(e):
            raise Exception("Не удаётся разрешить адрес Chess.com. Пожалуйста, проверьте настройки DNS.")
        else:
            raise Exception(f"Ошибка доступа к API Chess.com: {str(e)}")
    except (KeyError, ValueError) as e:
        raise Exception(f"Ошибка обработки статистики игрока: {str(e)}")
    except json.JSONDecodeError as e:
        raise Exception(f"Ошибка разбора ответа API: {str(e)}")

    try:
        # Прочитать расширенный набор данных из правильного пути
        adv_dataset_path = os.path.join('player_data', u1, 'chess_dataset_adv.csv')
        if not os.path.exists(adv_dataset_path):
            raise FileNotFoundError(f"Расширенный набор данных не найден по пути {adv_dataset_path}")
        df = pd.read_csv(adv_dataset_path)

        # Рассчитать ожидаемый результат по формуле ФИДЕ
        expected_score = 1 / (1 + 10 ** (-diff/400))
        
        # Настроить вероятности на основе разницы рейтингов и исторических данных
        df['rating_bucket'] = pd.cut(df['rating_difference'], 
                                   bins=[-float('inf'), -400, -200, -100, 0, 100, 200, 400, float('inf')],
                                   labels=['<-400', '-400to-200', '-200to-100', '-100to0', '0to100', '100to200', '200to400', '>400'])
        
        # Рассчитать исторические вероятности для каждого сегмента
        historical_probs = df.groupby('rating_bucket')['result_val_for_player'].agg(['mean', 'count']).reset_index()
        
        # Найти сегмент для текущей разницы рейтингов
        if diff <= -400:
            bucket = '<-400'
        elif -400 < diff <= -200:
            bucket = '-400to-200'
        elif -200 < diff <= -100:
            bucket = '-200to-100'
        elif -100 < diff <= 0:
            bucket = '-100to0'
        elif 0 < diff <= 100:
            bucket = '0to100'
        elif 100 < diff <= 200:
            bucket = '100to200'
        elif 200 < diff <= 400:
            bucket = '200to400'
        else:
            bucket = '>400'
            
        # Получить историческую вероятность для этого сегмента
        bucket_stats = historical_probs[historical_probs['rating_bucket'] == bucket].iloc[0]
        historical_win_rate = bucket_stats['mean']
        sample_size = bucket_stats['count']
        
        # Средневзвешенное между формулой ФИДЕ и историческими данными
        # Вес увеличивается с размером выборки до максимума 0.7
        historical_weight = min(0.7, sample_size / 100)
        fide_weight = 1 - historical_weight
        
        win_prob = (fide_weight * expected_score + historical_weight * historical_win_rate) * 100
        
        # Рассчитать вероятность ничьей на основе разницы рейтингов
        # Вероятность ничьей максимальна, когда рейтинги близки
        base_draw_prob = 30  # Базовая вероятность ничьей
        rating_factor = abs(diff) / 400  # Нормализовать разницу рейтингов
        draw_prob = base_draw_prob * (1 - min(1, rating_factor))  # Уменьшать вероятность ничьей с увеличением разницы рейтингов
        
        # Убедиться, что вероятность поражения составляет 100%
        loss_prob = 100 - win_prob - draw_prob
        
        # Определить результат на основе вероятностей
        if win_prob > max(draw_prob, loss_prob):
            result = 'Победа'
        elif draw_prob > max(win_prob, loss_prob):
            result = 'Ничья'
        else:
            result = 'Поражение'

        # Сохранить точность модели (используя историческую точность для соответствующего сегмента)
        di["ord_acc"] = f'Точность модели для игр с похожей разницей рейтингов: {bucket_stats["mean"]*100:.1f}% (на основе {int(bucket_stats["count"])} игр)'
        
        # Сохранить сводку модели
        summary = (f"Разница рейтингов: {diff}\n"
                  f"Исторический процент побед для похожей разницы: {historical_win_rate*100:.1f}%\n"
                  f"Ожидаемый результат по ФИДЕ: {expected_score*100:.1f}%\n"
                  f"Размер выборки: {int(sample_size)} игр\n"
                  f"Вес исторических данных: {historical_weight*100:.1f}%")
        di["summ1"] = summary

        di["result"] = (f'Результат: {result}\n\n'
                       f'Вероятность победы: {win_prob:.1f}%\n'
                       f'Вероятность ничьей: {draw_prob:.1f}%\n'
                       f'Вероятность поражения: {loss_prob:.1f}%')

    except FileNotFoundError as e:
        raise Exception(f"Ошибка: {str(e)}")
    except Exception as e:
        raise Exception(f"Ошибка при расчёте прогноза: {str(e)}")

    return di




#k1 = predict("tyrange","sudesh2911")
#print(k1) 