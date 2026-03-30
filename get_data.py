import pandas as pd
import requests
import io
import os
from typing import List, Dict, Any
import time
import concurrent.futures  # noqa: F401
import logging

try:
    import chess.pgn
except ImportError:
    raise ImportError(
        "Требуется python-chess. Установите его командой: pip install python-chess"
    )

logger = logging.getLogger(__name__)

def getGames(username: str) -> List[Dict[str, Any]]:
    """Получить все игры пользователя из API chess.com"""
    headers = {
        'User-Agent': 'Chess Analysis App (Contact: https://github.com/Kusha-a111/chess-analysis-app)',
        'Accept': 'application/json'
    }
    
    try:
        archives_url = f"https://api.chess.com/pub/player/{username}/games/archives"
        response = requests.get(archives_url, headers=headers, timeout=10)
        
        if response.status_code == 404:
            raise Exception("Пользователь не найден. Проверьте имя пользователя и попробуйте снова.")
        response.raise_for_status()
        
        archives = response.json()["archives"]
        
        all_games = []
        for archive_url in archives: 
            try:
                response = requests.get(archive_url, headers=headers, timeout=10)
                response.raise_for_status()
                all_games.extend(response.json()["games"])
                logger.info(f"Загружено {len(response.json()['games'])} игр из {archive_url}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"Не удалось загрузить архив {archive_url}: {str(e)}")
                continue
        
        if not all_games:
            raise Exception("Для этого пользователя не найдено игр")
        
        logger.info(f"Всего загружено игр: {len(all_games)}")
        return all_games
        
    except requests.exceptions.RequestException as e:
        if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
            raise Exception("Пользователь не найден. Проверьте имя пользователя и попробуйте снова.")
        elif "getaddrinfo failed" in str(e):
            raise Exception("Не удаётся подключиться к Chess.com. Проверьте интернет-соединение.")
        else:
            raise Exception(f"Ошибка доступа к API Chess.com: {str(e)}")

def filterList(games: List[Dict[str, Any]], username: str) -> None:
    """Отфильтровать список игр, оставив только стандартные шахматы"""
    # Удалить нестандартные шахматные игры (варианты и т.д.)
    for game in games[:]:  # Создаём копию для итерации при изменении
        if game.get("rules") != "chess":
            games.remove(game)
            continue
        
        # Удалить игры без корректного PGN
        if not game.get("pgn"):
            games.remove(game)
            continue
        
        # Удалить игры, в которых пользователь не участвовал
        if (game.get("white", {}).get("username") != username and 
            game.get("black", {}).get("username") != username):
            games.remove(game)

def createDataset(li: List[Dict[str, Any]], user: str) -> None:
    """Создать набор данных из шахматных игр и сохранить в CSV"""
    col = [
        'player_username', 'opponent_username', 'played_as', 'opponent_played_as',
        'result_for_player', 'result_for_opponent', 'player_rating', 'opponent_rating',
        'time_class', 'opening', 'moves', 'first_move', 'rated', 'PGN', 'FEN'
    ]

    df = pd.DataFrame(columns=col)

    for x in li:
        try:
            liz = [None] * 15

            if x["rules"] != "chess":
                continue

            liz[0] = user
            liz[8] = x["time_class"]
            liz[13] = x["pgn"]
            liz[14] = x["fen"]
            liz[12] = x["rated"]

            try:
                pgn = chess.pgn.read_game(io.StringIO(x["pgn"]))
                if pgn and "ECOUrl" in pgn.headers:
                    opening = pgn.headers["ECOUrl"][31:].replace("-", " ")
                    liz[9] = opening
            except Exception as e:
                print(f"Предупреждение: не удалось разобрать PGN для игры: {str(e)}")
                continue

            count = 0
            for moves in pgn.mainline_moves():
                if count == 0:
                    liz[11] = str(moves)
                count += 1

            liz[10] = str(int(count / 2))

            if x["white"]["username"] == user:
                liz[2] = "white"
                liz[3] = "black"
                liz[4] = x["white"]["result"]
                liz[5] = x["black"]["result"]
                liz[6] = x["white"]["rating"]
                liz[7] = x["black"]["rating"]
                liz[1] = x["black"]["username"]
            else:
                liz[2] = "black"
                liz[3] = "white"
                liz[4] = x["black"]["result"]
                liz[5] = x["white"]["result"]
                liz[6] = x["black"]["rating"]
                liz[7] = x["white"]["rating"]
                liz[1] = x["white"]["username"]

            if None not in liz:
                df.loc[len(df)] = liz
                
        except Exception as e:
            print(f"Предупреждение: не удалось обработать игру: {str(e)}")
            continue

    # Создать директорию в player_data, если она не существует
    user_dir = os.path.join('player_data', user)
    os.makedirs(user_dir, exist_ok=True)
    df.to_csv(os.path.join(user_dir, 'chess_dataset.csv'), index=False)

def createAdvancedDataset(username: str) -> None:
    """Создать расширенный набор данных с числовыми значениями результатов для прогнозирования"""
    try:
        # Прочитать исходный набор данных
        df = pd.read_csv(os.path.join('player_data', username, 'chess_dataset.csv'))
        
        # Создать столбец числового значения результата (0 — поражение, 0.5 — ничья, 1 — победа)
        df['result_val_for_player'] = df['result_for_player'].map({
            'win': 1.0,
            'agreed': 0.5,
            'timevsinsufficient': 0.5,
            'insufficient': 0.5,
            'stalemate': 0.5,
            'repetition': 0.5,
            'resigned': 0.0,
            'checkmated': 0.0,
            'timeout': 0.0,
            'abandoned': 0.0
        })
        
        # Вычислить разницу рейтингов
        df['rating_difference'] = df['player_rating'] - df['opponent_rating']
        
        # Сохранить расширенный набор данных
        df.to_csv(os.path.join('player_data', username, 'chess_dataset_adv.csv'), index=False)
        
    except Exception as e:
        raise Exception(f"Ошибка при создании расширенного набора данных: {str(e)}")

def check_cached_data(username: str) -> bool:
    """Проверить, существуют ли необходимые файлы данных и валидны ли они"""
    required_files = ["chess_dataset.csv", "chess_dataset_adv.csv"]
    player_dir = os.path.join('player_data', username)
    
    # Проверить, существуют ли все необходимые файлы
    if not all(os.path.exists(os.path.join(player_dir, file)) for file in required_files):
        return False
        
    # Проверить, что файлы не пусты и не устарели (менее 24 часов)
    try:
        for file in required_files:
            file_path = os.path.join(player_dir, file)
            # Проверить, не пуст ли файл
            if os.path.getsize(file_path) == 0:
                return False
            # Проверить, свежий ли файл
            if time.time() - os.path.getmtime(file_path) > 24 * 3600:  # 24 часа в секундах
                return False
        return True
    except Exception:
        return False

def driver_fn(username: str) -> None:
    """Основная функция для получения и обработки шахматных данных"""
    try:
        # Создать директорию пользователя
        user_dir = os.path.join('player_data', username)
        os.makedirs(user_dir, exist_ok=True)
        
        # Проверить, есть ли валидные кэшированные данные
        if check_cached_data(username):
            print(f"Используются кэшированные данные для {username}")
            return
            
        # Получить данные игр
        games = getGames(username)
        filterList(games, username)
        
        # Создать наборы данных
        createDataset(games, username)
        createAdvancedDataset(username)  # Создать расширенный набор данных для прогнозирования
        
    except Exception as e:
        raise Exception(f"Ошибка в основной функции: {str(e)}")