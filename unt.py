import os
import warnings
import logging
import sys
from typing import Dict, List, Optional  # noqa: F401
import gc
import threading
import _thread
import io
import chess
import chess.pgn
import cairosvg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go  # noqa: F401

# Настройка логирования с выводом в файл
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'visualization.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),  # Режим добавления
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info("="*50)  # Разделитель для новых запусков
logger.info("Запуск новой сессии визуализации")

# Подавление предупреждений
warnings.filterwarnings('ignore')

class TimeoutException(Exception):
    pass

def timeout_handler():
    _thread.interrupt_main()

def time_limit(timeout):
    """Контекстный менеджер таймаута, совместимый с Windows, использующий потоки"""
    timer = threading.Timer(timeout, timeout_handler)
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Время истекло!")
    finally:
        timer.cancel()

def check_dependencies() -> bool:
    """Проверить, установлены ли все необходимые зависимости"""
    try:
        import pandas as pd  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
        import seaborn as sns  # noqa: F401
        import plotly.graph_objects as go  # noqa: F401
        import plotly.io as pio  # noqa: F401
        import chess
        import chess.svg
        import chess.pgn  # noqa: F401
        
        # Настройка matplotlib для неинтерактивного бэкенда
        plt.switch_backend('Agg')
        
        return True
    except ImportError as e:
        logger.error(f"Отсутствует необходимая зависимость: {str(e)}")
        return False

def save_plotly_figure(fig, filepath: str, scale: int = 3) -> None:
    """Сохранить фигуру plotly с обработкой ошибок"""
    try:
        import plotly.io as pio
        logger.info(f"Сохранение фигуры в {filepath}")
        
        # Убедиться, что директория существует
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Настройка kaleido для лучшей производительности
        pio.kaleido.scope.chromium_args = (
                    '--no-sandbox',
                    '--disable-gpu',
                    '--disable-dev-shm-usage',
                    '--single-process'
        )
        
        # Сохранение с минимальной конфигурацией
        fig.write_image(
            filepath,
            format='png',
            engine='kaleido',
            scale=2,
            width=1200,
            height=800
        )
        logger.info("Фигура успешно сохранена")
        
    except Exception as e:
        logger.error(f"Не удалось сохранить фигуру: {str(e)}")
        # Запасной вариант — HTML
        try:
                html_path = filepath.replace('.png', '.html')
                fig.write_html(html_path)
                logger.info(f"Сохранено как HTML-запасной вариант: {html_path}")
        except Exception as e2:
            logger.error(f"HTML-запасной вариант также не удался: {str(e2)}")

def fight(df: pd.DataFrame, username: str) -> None:
    """Создать визуализацию анализа длины партий для проигранных игр"""
    try:
        logger.info("Запуск анализа длины партий...")
        
        # Получить последние 100 проигранных партий
        lost_games = df[df['result_for_opponent'] == "win"].tail(100)
        
        # Создать DataFrame для построения графика
        moves_df = pd.DataFrame({
            'game_no': range(len(lost_games)),
            'moves': lost_games['moves'].values
        })
        
        # Создать фигуру с matplotlib
        plt.figure(figsize=(12, 8))
        plt.style.use('seaborn')
        
        # Создать линейный график с маркерами
        plt.plot(moves_df['game_no'], moves_df['moves'], 
                marker='o', markersize=6, linewidth=2, 
                color='#2196F3', markerfacecolor='white',
                markeredgecolor='#2196F3', markeredgewidth=1.5)
        
        # Настроить график
        plt.title("Количество ходов в последних 100 проигранных партиях", size=16, pad=20)
        plt.xlabel("Номер партии", size=12, labelpad=10)
        plt.ylabel("Количество ходов", size=12, labelpad=10)
        
        # Добавить сетку
        plt.grid(True, alpha=0.3)
        
        # Настроить отступы
        plt.tight_layout()
        
        # Сохранить фигуру
        output_path = os.path.join('player_data', username, "fight.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Визуализация анализа длины партий завершена")
        
    except Exception as e:
        logger.error(f"Ошибка в анализе длины партий: {str(e)}")
        raise

def wh_countplot(df: pd.DataFrame, username: str) -> None:
    """Создать визуализацию анализа дебютов для белых"""
    try:
        # Анализ партий белыми
        white_df = df[df["played_as"] == "white"]
        white_op_freq = white_df['opening'].value_counts().head(20)
        
        # Установить размер фигуры и стиль
        plt.figure(figsize=(20, 15))
        sns.set(rc={'figure.figsize': (20, 15)})
        sns.set_style("darkgrid", {'axes.grid': False})
        
        # Создать график
        ax = sns.countplot(
            y='opening',
            data=white_df[white_df['opening'].isin(white_op_freq.index)],
            order=white_op_freq.index
        )
        
        # Добавить значения частоты в конце каждой полосы
        for i, v in enumerate(white_op_freq.values):
            ax.text(v + 0.1, i, str(v), va='center', fontsize=12)
        
        # Оформить график
        ax.set_ylabel("")
        ax.set_xlabel("Частота", size=20, labelpad=30)
        ax.tick_params(labelsize=17)
        
        # Расширить ось X, чтобы освободить место для подписей
        plt.xlim(0, max(white_op_freq.values) * 1.1)
        
        # Сохранить фигуру
        output_path = os.path.join('player_data', username, "top_op_wh.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
    except Exception as e:
        logger.error(f"Ошибка в анализе дебютов: {str(e)}")
        raise

def bl_countplot(df: pd.DataFrame, username: str) -> None:
    """Создать визуализацию анализа дебютов для чёрных"""
    try:
        # Анализ партий чёрными
        black_df = df[df["played_as"] == "black"]
        black_op_freq = black_df['opening'].value_counts().head(20)
        
        # Установить размер фигуры и стиль
        plt.figure(figsize=(20, 15))
        sns.set(rc={'figure.figsize': (20, 15)})
        sns.set_style("darkgrid", {'axes.grid': False})
        
        # Создать график
        ax = sns.countplot(
            y='opening',
            data=black_df[black_df['opening'].isin(black_op_freq.index)],
            order=black_op_freq.index
        )
        
        # Добавить значения частоты в конце каждой полосы
        for i, v in enumerate(black_op_freq.values):
            ax.text(v + 0.1, i, str(v), va='center', fontsize=12)
        
        # Оформить график
        ax.set_ylabel("")
        ax.set_xlabel("Частота", size=20, labelpad=30)
        ax.tick_params(labelsize=17)
        
        # Расширить ось X, чтобы освободить место для подписей
        plt.xlim(0, max(black_op_freq.values) * 1.1)
        
        # Сохранить фигуру
        output_path = os.path.join('player_data', username, "top_op_bl.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        
    except Exception as e:
        logger.error(f"Ошибка в анализе дебютов чёрными: {str(e)}")
        raise

def most_used_wh(df: pd.DataFrame, username: str) -> None:
    """Создать визуализации шахматной доски для топ-3 первых ходов белыми"""
    try:
        logger.info("Создание визуализации топ-3 первых ходов...")
        import chess
        import chess.svg
        
        # Получить топ-3 хода белыми
        white_df = df[df["played_as"] == "white"]
        if white_df.empty:
            logger.warning("Не найдено партий белыми фигурами")
            return
            
        # Извлечь и подсчитать первые ходы
        first_moves = white_df["first_move"].value_counts()
        logger.info(f"Найдены первые ходы: {first_moves.to_dict()}")
        
        if first_moves.empty:
            logger.warning("Не найдено первых ходов в партиях белыми")
            return

        # Обработать топ-3 хода
        for i, (move, count) in enumerate(first_moves.head(3).items(), 1):
            try:
                # Создать новую доску
                board = chess.Board()
                
                # Разобрать ход с помощью встроенного парсера python-chess
                try:
                    chess_move = board.parse_san(move)
                    board.push(chess_move)
                except ValueError:
                    logger.warning(f"Не удалось разобрать ход {move}")
                    continue
                    
                # Сгенерировать SVG в стиле Lichess
                svg_content = chess.svg.board(
                    board=board,
                    size=400,
                    coordinates=True,
                    colors={
                        'square light': '#f0d9b5',  # Светлые поля Lichess (коричневые)
                        'square dark': '#b58863',   # Тёмные поля Lichess (коричневые)
                        'square light lastmove': '#cdd26a',  # Подсветка последнего хода Lichess (светлые)
                        'square dark lastmove': '#aaa23a',   # Подсветка последнего хода Lichess (тёмные)
                        'margin': 'none',
                        'coord': '#666666'          # Цвет координат Lichess
                    }
                )
                
                # Сначала сохранить SVG
                svg_path = os.path.join('player_data', username, f'top_opening_move_as_white_{i}.svg')
                with open(svg_path, 'w') as f:
                    f.write(svg_content)
                
                # Конвертировать в PNG
                png_path = os.path.join('player_data', username, f'top_opening_move_as_white_{i}.png')
                cairosvg.svg2png(
                    url=svg_path,
                    write_to=png_path,
                    scale=2.0  # Увеличить качество
                )
                
                logger.info(f"Успешно создана визуализация для хода {i}: {move}")
                
            except Exception as e:
                logger.error(f"Ошибка при создании визуализации для хода {i}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Ошибка в визуализации топ-ходов: {str(e)}")
        raise

def most_used_bl(df: pd.DataFrame, username: str) -> None:
    """Создать визуализации шахматной доски для топ-3 первых ответов чёрными"""
    try:
        logger.info("Создание визуализации топ-3 ответов чёрными...")
        import chess
        import chess.svg
        import chess.pgn
        
        # Получить партии, где пользователь играл чёрными
        black_games = df[df['played_as'] == "black"]
        if black_games.empty:
            logger.warning("Не найдено партий чёрными фигурами")
            return
            
        # Словарь для хранения первых ходов чёрных
        black_replies = {}
        
        # Обработать каждую партию, чтобы извлечь первый ход чёрных
        for _, row in black_games.iterrows():
            pgn = chess.pgn.read_game(io.StringIO(row['PGN']))
            if pgn:
                moves = list(pgn.mainline_moves())
                if len(moves) >= 2:  # Убедиться, что есть как минимум 2 хода
                    board = chess.Board()
                    board.push(moves[0])  # Применить первый ход белых
                    black_move = moves[1]  # Получить ответ чёрных
                    move_san = board.san(black_move)  # Получить ход в нотации SAN
                    black_replies[move_san] = black_replies.get(move_san, 0) + 1

        if not black_replies:
            logger.warning("Не найдено ответов чёрных")
            return

        # Отсортировать ответы по частоте
        sorted_replies = sorted(black_replies.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"Найдены ответы чёрных: {dict(sorted_replies)}")

        # Обработать топ-3 ответа
        for i, (move, count) in enumerate(sorted_replies[:3], 1):
            try:
                # Создать новую доску
                board = chess.Board()
                
                # Сделать распространённый первый ход белыми (e4), чтобы показать контекст ответа чёрных
                board.push_san("e4")  # Мы покажем ответы чёрных на e4
                
                # Разобрать ход чёрных
                try:
                    chess_move = board.parse_san(move)
                    board.push(chess_move)
                except ValueError:
                    logger.warning(f"Не удалось разобрать ход {move}")
                    continue
                    
                # Сгенерировать SVG в стиле Lichess
                svg_content = chess.svg.board(
                    board=board,
                    size=400,
                    coordinates=True,
                    orientation=chess.BLACK,  # Показать доску с точки зрения чёрных
                    colors={
                        'square light': '#f0d9b5',  # Светлые поля Lichess (коричневые)
                        'square dark': '#b58863',   # Тёмные поля Lichess (коричневые)
                        'square light lastmove': '#cdd26a',  # Подсветка последнего хода Lichess (светлые)
                        'square dark lastmove': '#aaa23a',   # Подсветка последнего хода Lichess (тёмные)
                        'margin': 'none',
                        'coord': '#666666'          # Цвет координат Lichess
                    }
                )
                
                # Сначала сохранить SVG
                svg_path = os.path.join('player_data', username, f'top_reply_move_as_black_{i}.svg')
                with open(svg_path, 'w') as f:
                    f.write(svg_content)
                
                # Конвертировать в PNG
                png_path = os.path.join('player_data', username, f'top_reply_move_as_black_{i}.png')
                cairosvg.svg2png(
                    url=svg_path,
                    write_to=png_path,
                    scale=2.0  # Увеличить качество
                )
                
                logger.info(f"Успешно создана визуализация для ответа чёрных {i}: {move}")
                
            except Exception as e:
                logger.error(f"Ошибка при создании визуализации для хода {i}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Ошибка в визуализации топ-ответов чёрных: {str(e)}")
        raise

def create_rating_ladder(df: pd.DataFrame, username: str) -> None:
    """Создать визуализацию прогресса рейтинга, показывающую последние 150 партий для каждого контроля времени"""
    try:
        logger.info("Создание визуализации лестницы рейтинга...")
        
        # Определить контроли времени, которые мы хотим отслеживать (исключая ежедневные)
        time_controls = ["bullet", "blitz", "rapid"]
        
        # Создать фигуру со стилем seaborn
        plt.figure(figsize=(12, 6))
        sns.set_style("darkgrid", {'axes.grid': True, 'grid.color': '.8', 'grid.linestyle': '-'})
        
        # Отслеживать, есть ли данные для построения
        has_data = False
        
        # Определить маркеры и цвета для каждого контроля времени с лучшей видимостью
        style_map = {
            'bullet': ('X', '#FF3333'),  # Изменено 'x' на 'X' для большего маркера, ярко-красный
            'blitz': ('o', '#0066CC'),   # Тёмно-синий круг
            'rapid': ('s', '#00CC66'),   # Тёмно-зелёный квадрат
        }
        
        # Обработать каждый контроль времени
        for time_class in time_controls:
            # Получить последние 150 рейтинговых партий для этого контроля времени
            games = df[
                (df['rated']) & 
                (df['time_class'] == time_class)
            ].tail(150)
            
            # Пропустить, если нет партий для этого контроля времени
            if games.empty:
                logger.info(f"Не найдено рейтинговых партий в контроле {time_class}")
                continue
                
            # Удалить значения NaN из player_rating
            games = games.dropna(subset=['player_rating'])
            
            if not games.empty:
                has_data = True
                marker, color = style_map[time_class]
                # Создать линейный график
                sns.lineplot(
                    data=games, 
                    x=range(len(games)), 
                    y='player_rating',
                    label=time_class.capitalize(),
                    marker=marker,
                    color=color,
                    markersize=8,  # Увеличено с 6
                    markeredgewidth=2,  # Добавлено, чтобы сделать маркеры более заметными
                    linewidth=1.5
                )
        
        if not has_data:
            logger.warning("Не найдено рейтинговых партий ни в одном контроле времени")
            return
            
        # Настроить график
        plt.title("Прогресс рейтинга по контролю времени (последние 150 партий для каждого типа)", size=14, pad=20)
        plt.xlabel("Номер партии", size=12)
        plt.ylabel("Рейтинг", size=12)
        
        # Добавить легенду с пользовательским заголовком
        plt.legend(
            title="Контроль времени",
            title_fontsize=12,
            fontsize=10,
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
        
        # Настроить отступы, чтобы избежать обрезания подписей
        plt.tight_layout()
        
        # Сохранить график
        output_path = os.path.join('player_data', username, "rating_ladder_red.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Визуализация лестницы рейтинга успешно создана")
        
    except Exception as e:
        logger.error(f"Ошибка при создании лестницы рейтинга: {str(e)}")
        raise

def create_result_distribution(df: pd.DataFrame, username: str) -> None:
    """Создать круговую диаграмму результатов партий"""
    try:
        logger.info("Создание визуализации распределения результатов...")
        
        # Подсчитать результаты
        result_counts = df['result_for_player'].value_counts()
        
        # Создать фигуру
        plt.figure(figsize=(10, 8))
        plt.style.use('seaborn')
        
        # Создать круговую диаграмму с чистым стилем
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(result_counts)))
        plt.pie(result_counts.values, labels=result_counts.index, 
               autopct='%1.1f%%', startangle=90,
               textprops={'fontsize': 12},
               colors=colors)
        
        # Убрать ненужное оформление
        plt.axis('equal')
        
        # Сохранить фигуру
        output_path = os.path.join('player_data', username, "result_pi.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        logger.info("Визуализация распределения результатов завершена")
        
    except Exception as e:
        logger.error(f"Ошибка в распределении результатов: {str(e)}")
        raise

def create_time_control_dist(df: pd.DataFrame, username: str) -> None:
    """Создать визуализацию распределения контроля времени"""
    try:
        logger.info("Создание визуализации распределения контроля времени...")
        
        # Получить количество партий по контролю времени
        time_counts = df['time_class'].value_counts()
        
        # Создать фигуру
        plt.figure(figsize=(12, 8))
        
        # Определить цвета для каждого контроля времени
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99FFCC']
        
        # Создать круговую диаграмму с чистым стилем
        patches, texts, autotexts = plt.pie(
            time_counts.values, 
            labels=time_counts.index,
            colors=colors[:len(time_counts)],
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100.*sum(time_counts.values))})',
            startangle=90,
            textprops={'fontsize': 12}
        )
        
        # Добавить заголовок
        plt.title('Распределение по контролю времени', size=16, pad=20)
        
        # Добавить легенду
        plt.legend(
            patches,
            time_counts.index,
            title="Контроль времени",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
        
        # Равное соотношение сторон обеспечивает круглую форму
        plt.axis('equal')
        
        # Сохранить фигуру с дополнительным пространством для легенды
        output_path = os.path.join('player_data', username, "time_class.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info("Визуализация контроля времени завершена")
        
    except Exception as e:
        logger.error(f"Ошибка в распределении контроля времени: {str(e)}")
        raise

def create_color_results(df: pd.DataFrame, username: str) -> None:
    """Создать визуализации результатов по цвету фигур"""
    try:
        logger.info("Создание визуализаций результатов по цвету...")
        
        # Функция для создания пончиковой диаграммы
        def create_donut_chart(data, title, output_path):
            plt.figure(figsize=(10, 8))
            plt.style.use('seaborn')
            
            # Рассчитать проценты
            total = sum(data)
            percentages = [count/total * 100 for count in data]
            
            # Создать круговую диаграмму
            plt.pie(percentages, 
                   labels=['победа', 'ничья', 'поражение'],
                   colors=['#2ecc71', '#95a5a6', '#e74c3c'],  # Зелёный, серый, красный
                   autopct='%1.1f%%',
                   startangle=90,
                   textprops={'fontsize': 12})
            
            # Добавить круг в центре для создания эффекта пончика
            centre_circle = plt.Circle((0, 0), 0.70, color='gray', fc='white')
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)
            
            plt.title(title, size=14, pad=20)
            plt.axis('equal')
            
            # Сохранить фигуру
            plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
            plt.close()
        
        # Партии белыми
        white_games = df[df['played_as'] == 'white']
        white_wins = len(white_games[white_games['result_for_player'] == 'win'])
        white_draws = len(white_games[white_games['result_for_player'].isin(
            ['agreed', 'timevsinsufficient', 'insufficient', 'stalemate', 'repetition'])])
        white_losses = len(white_games[white_games['result_for_player'].isin(
            ['resigned', 'checkmated', 'timeout', 'abandoned'])])
        
        # Создать пончиковую диаграмму результатов белыми
        output_path = os.path.join('player_data', username, "result_as_wh.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        create_donut_chart([white_wins, white_draws, white_losses], 
                        'Результаты белыми', output_path)
        
        # Партии чёрными
        black_games = df[df['played_as'] == 'black']
        black_wins = len(black_games[black_games['result_for_player'] == 'win'])
        black_draws = len(black_games[black_games['result_for_player'].isin(
            ['agreed', 'timevsinsufficient', 'insufficient', 'stalemate', 'repetition'])])
        black_losses = len(black_games[black_games['result_for_player'].isin(
            ['resigned', 'checkmated', 'timeout', 'abandoned'])])
        
        # Создать пончиковую диаграмму результатов чёрными
        output_path = os.path.join('player_data', username, "result_as_bl.png")
        create_donut_chart([black_wins, black_draws, black_losses], 
                        'Результаты чёрными', output_path)
        
        logger.info("Визуализации результатов по цвету завершены")
        
    except Exception as e:
        logger.error(f"Ошибка в результатах по цвету: {str(e)}")
        raise

def create_top_5_openings(df: pd.DataFrame, username: str) -> None:
    """Создать анализ топ-5 наиболее успешных и наименее успешных дебютов для обоих цветов"""
    try:
        logger.info("Создание анализа топ-5 дебютов...")
        
        # Убедиться, что выходная директория существует
        output_dir = os.path.join('player_data', username)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Создана выходная директория: {output_dir}")
        
        def wrap_labels(text, width=30):
            """Обернуть текст на указанной ширине"""
            import textwrap
            return textwrap.fill(text, width=width)
        
        def calculate_opening_stats(color_df):
            """Рассчитать статистику по дебютам, используя систему очков"""
            opening_stats = []
            logger.info(f"Обработка {len(color_df)} партий")
            
            # Сначала получить количество партий по дебютам
            opening_counts = color_df['opening'].value_counts()
            logger.info(f"Найдено уникальных дебютов: {len(opening_counts)}")
            
            # Обработать дебюты
            for opening in opening_counts.index:
                games = color_df[color_df['opening'] == opening]
                total_games = len(games)
                
                wins = len(games[games['result_for_player'] == 'win'])
                draws = len(games[games['result_for_player'].isin(
                    ['agreed', 'timevsinsufficient', 'insufficient', 'stalemate', 'repetition'])])
                losses = len(games[games['result_for_player'].isin(
                    ['resigned', 'checkmated', 'timeout', 'abandoned'])])
                
                # Рассчитать очки по взвешенной системе
                points = (wins * 1.0) + (draws * 0.5) + (losses * 0)
                points_percentage = (points / total_games) * 100
                
                opening_stats.append({
                    'opening': opening,
                    'points_percentage': points_percentage,
                    'total_games': total_games,
                    'wins': wins,
                    'draws': draws,
                    'losses': losses,
                    'points': points
                })
            
            logger.info(f"Обработано {len(opening_stats)} дебютов")
            return opening_stats
        
        def create_opening_chart(stats, title, output_path, best=True):
            """Создать столбчатую диаграмму для дебютов"""
            try:
                logger.info(f"Создание диаграммы: {title}")
                logger.info(f"Доступно статистических данных: {len(stats)}")
                
                # Получить топ-5 или последние 5 по проценту очков
                selected_stats = stats[:5] if best else stats[-5:]
                if not best:
                    selected_stats = selected_stats[::-1]  # Обратный порядок для худших дебютов
                
                logger.info(f"Выбрано {len(selected_stats)} дебютов для визуализации")
                
                # Подготовить данные для построения
                openings = [wrap_labels(s['opening'], width=25) for s in selected_stats]
                wins = [s['wins'] for s in selected_stats]
                draws = [s['draws'] for s in selected_stats]
                losses = [s['losses'] for s in selected_stats]
                
                logger.info(f"Данные подготовлены - Дебютов: {len(openings)}, Побед: {len(wins)}, Ничьих: {len(draws)}, Поражений: {len(losses)}")
                
                # Создать DataFrame для построения
                most_used_openings = pd.DataFrame({
                    'wins': wins,
                    'draws': draws,
                    'losses': losses
                }, index=openings)
                
                # Создать фигуру с настроенным соотношением размеров
                plt.figure(figsize=(15, 8))
                
                # Создать ненасыщенную столбчатую диаграмму
                ax = most_used_openings[["wins", "draws", "losses"]].plot.barh(
                    rot=0, 
                    color=["green", "blue", "red"], 
                    stacked=False
                )
                
                # Настроить график
                ax.set_facecolor('xkcd:white')
                
                # Добавить легенду в рамке
                ax.legend(
                    prop={'size': 14},
                    frameon=True,
                    facecolor='white',
                    edgecolor='black',
                    bbox_to_anchor=(1.0, 1.0),
                    loc='upper left',
                    borderaxespad=0.
                )
                
                # Центрировать заголовок с меньшим размером
                plt.title(title, size=20, y=1.02, pad=15, ha='center')
                
                # Настроить подписи
                plt.xlabel("Количество партий", size=16, labelpad=20)
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
                
                # Добавить больше места для столбцов
                plt.subplots_adjust(left=0.3)
                
                # Сохранить фигуру
                logger.info(f"Сохранение фигуры в: {output_path}")
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Диаграмма успешно создана: {title}")
                
            except Exception as e:
                logger.error(f"Ошибка при создании диаграммы {title}: {str(e)}")
                logger.error(f"Тип ошибки: {type(e).__name__}")
                import traceback
                logger.error(f"Трассировка: {traceback.format_exc()}")
                plt.close()
                raise
        
        # Анализ дебютов белыми
        white_df = df[df['played_as'] == 'white']
        logger.info(f"Найдено {len(white_df)} партий белыми")
        white_stats = calculate_opening_stats(white_df)
        
        # Создать диаграммы для наиболее часто используемых дебютов
        create_opening_chart(
            sorted(white_stats, key=lambda x: x['total_games'], reverse=True),
            "Результаты по топ-5 дебютам белыми",
            os.path.join(output_dir, "result_top_5_wh.png"),
            best=True
        )
        
        # Анализ дебютов чёрными
        black_df = df[df['played_as'] == 'black']
        logger.info(f"Найдено {len(black_df)} партий чёрными")
        black_stats = calculate_opening_stats(black_df)
        
        # Создать диаграммы для наиболее часто используемых дебютов
        create_opening_chart(
            sorted(black_stats, key=lambda x: x['total_games'], reverse=True),
            "Результаты по топ-5 дебютам чёрными",
            os.path.join(output_dir, "result_top_5_bl.png"),
            best=True
        )
        
        logger.info("Анализ топ-5 дебютов завершён")
        
    except Exception as e:
        logger.error(f"Ошибка в анализе топ-5 дебютов: {str(e)}")
        logger.error(f"Тип ошибки: {type(e).__name__}")
        import traceback
        logger.error(f"Трассировка: {traceback.format_exc()}")
        raise  # Повторно поднять ошибку, чтобы она была передана дальше

def create_overall_results(df: pd.DataFrame, username: str) -> None:
    """Создать визуализацию общих результатов с детальной разбивкой по типам исходов"""
    try:
        logger.info("Создание визуализации общих результатов...")
        
        # Все возможные типы результатов и их количество
        all_results = ['win', 'resigned', 'checkmated', 'timeout', 'repetition', 
                      'abandoned', 'stalemate', 'time vs\ninsufficient', 'insufficient', 'agreed']
        
        # Отображение исходных значений на отображаемые
        result_map = {
            'timevsinsufficient': 'time vs\ninsufficient',
            'win': 'win',
            'resigned': 'resigned',
            'checkmated': 'checkmated',
            'timeout': 'timeout',
            'repetition': 'repetition',
            'abandoned': 'abandoned',
            'stalemate': 'stalemate',
            'insufficient': 'insufficient',
            'agreed': 'agreed'
        }
        
        # Создать копию столбца результатов с отображёнными значениями
        df['result_display'] = df['result_for_player'].map(result_map)
        result_counts = df['result_display'].value_counts()
        
        # Создать Series со всеми результатами, заполнив отсутствующие нулями
        most_frequently_opening = pd.Series(0, index=all_results)
        most_frequently_opening.update(result_counts)
        
        plt.figure(figsize=(20, 10))  # Увеличен размер фигуры
        
        # Создать пользовательскую палитру от тёмно-серо-голубого до светлых оттенков синего
        colors = sns.color_palette("Blues_d", n_colors=len(all_results))
        colors.reverse()  # Обратный порядок для соответствия исходному шаблону от тёмного к светлому
        
        # Создать столбчатую диаграмму с пользовательским стилем
        opening = sns.barplot(x=most_frequently_opening.index, 
                            y=most_frequently_opening.values,
                            palette=colors)
        
        # Оформить график
        plt.title("Распределение общих результатов партий", fontsize=24, pad=20)  # Добавлен заголовок
        plt.ylabel("Количество партий", fontsize=20, labelpad=15, weight='bold')
        plt.xlabel("Результат партии", fontsize=20, labelpad=15, weight='bold')
        
        # Добавить значения на вершинах столбцов с увеличенным размером
        for p in opening.patches:
            height = p.get_height()
            text = str(int(height))
            opening.text(p.get_x() + p.get_width() / 2, height + 1, 
                       text, ha="center", fontsize=14, fontweight='bold')
        
        # Установить цвет фона и убрать сетку
        opening.set_facecolor('xkcd:white')
        plt.grid(False)
        
        # Установить ось Y, начинающуюся с 0, с дополнительным отступом для подписей
        plt.ylim(0, max(most_frequently_opening.values) * 1.15)
        
        # Увеличить размер подписей делений и сделать их горизонтальными
        plt.xticks(fontsize=16, rotation=0, ha='center')
        plt.yticks(fontsize=16)
        
        # Настроить отступы, чтобы избежать обрезания подписей
        plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
        
        # Сохранить фигуру с более высоким DPI
        output_path = os.path.join('player_data', username, "overall_results.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info("Визуализация общих результатов завершена")
        
    except Exception as e:
        logger.error(f"Ошибка в общих результатах: {str(e)}")
        raise

def create_overall_results_pie(df: pd.DataFrame, username: str) -> None:
    """Создать круговую диаграмму общих результатов партий"""
    try:
        logger.info("Создание круговой диаграммы общих результатов...")
        
        # Получить количество партий по результатам
        result_counts = df['result_for_player'].value_counts()
        
        # Создать фигуру
        plt.figure(figsize=(12, 8))
        
        # Определить цвета для каждого типа результата
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(result_counts)))
        
        # Создать круговую диаграмму с чистым стилем
        patches, texts, autotexts = plt.pie(
            result_counts.values, 
            labels=result_counts.index,
            colors=colors,
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100.*sum(result_counts.values))})',
            startangle=90,
            textprops={'fontsize': 12}
        )
        
        # Добавить заголовок
        plt.title('Распределение общих результатов', size=16, pad=20)
        
        # Добавить легенду
        plt.legend(
            patches,
            result_counts.index,
            title="Результаты",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
        
        # Равное соотношение сторон обеспечивает круглую форму
        plt.axis('equal')
        
        # Сохранить фигуру с дополнительным пространством для легенды
        output_path = os.path.join('player_data', username, "overall_results_pie.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info("Круговая диаграмма общих результатов завершена")
        
    except Exception as e:
        logger.error(f"Ошибка в круговой диаграмме общих результатов: {str(e)}")
        raise

def wh_heatmap_beg(df: pd.DataFrame, username: str, vmax: Optional[float] = None) -> float:
    """Создать тепловую карту начальных полей для белых"""
    di = {
        "a1": [0, 0, 0, 0, 0, 0, 0, 0],
        "b1": [0, 0, 0, 0, 0, 0, 0, 0],
        "c1": [0, 0, 0, 0, 0, 0, 0, 0],
        "d1": [0, 0, 0, 0, 0, 0, 0, 0],
        "e1": [0, 0, 0, 0, 0, 0, 0, 0],
        "f1": [0, 0, 0, 0, 0, 0, 0, 0],
        "g1": [0, 0, 0, 0, 0, 0, 0, 0],
        "h1": [0, 0, 0, 0, 0, 0, 0, 0],
        "a2": [0, 0, 0, 0, 0, 0, 0, 0],
        "b2": [0, 0, 0, 0, 0, 0, 0, 0],
        "c2": [0, 0, 0, 0, 0, 0, 0, 0],
        "d2": [0, 0, 0, 0, 0, 0, 0, 0],
        "e2": [0, 0, 0, 0, 0, 0, 0, 0],
        "f2": [0, 0, 0, 0, 0, 0, 0, 0],
        "g2": [0, 0, 0, 0, 0, 0, 0, 0],
        "h2": [0, 0, 0, 0, 0, 0, 0, 0],
    }

    for index, row in df[df['played_as'] == "white"].iterrows():
        di[row["first_move"][0] + "1"][int(row["first_move"][1]) - 1] += 1
        di[row["first_move"][2] + "2"][int(row["first_move"][3]) - 1] += 1

    row = ['1', '2', '3', '4', '5', '6', '7', '8']
    a = di["a1"]
    b = di["b1"]
    c = di["c1"]
    d = di["d1"]
    e = di["e1"]
    f = di["f1"]
    g = di["g1"]
    h = di["h1"]

    mlist = [row, a, b, c, d, e, f, g, h]
    for lists in mlist:
        lists.reverse()

    board_open = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d,
                             'e': e, 'f': f, 'g': g, 'h': h}, index=row)

    plt.figure(figsize=(10, 10))
    board = sns.heatmap(board_open, cmap='Reds', square=True, linewidths=.1, linecolor='black', vmax=vmax)
    board.set_title('Тепловая карта начальных полей для белых', size=18, y=1.05)
    
    output_path = os.path.join('player_data', username, "heatmap_starting_white.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return board_open.values.max()

def wh_heatmap_end(df: pd.DataFrame, username: str, vmax: Optional[float] = None) -> float:
    """Создать тепловую карту конечных полей для белых"""
    di = {
        "a1": [0, 0, 0, 0, 0, 0, 0, 0],
        "b1": [0, 0, 0, 0, 0, 0, 0, 0],
        "c1": [0, 0, 0, 0, 0, 0, 0, 0],
        "d1": [0, 0, 0, 0, 0, 0, 0, 0],
        "e1": [0, 0, 0, 0, 0, 0, 0, 0],
        "f1": [0, 0, 0, 0, 0, 0, 0, 0],
        "g1": [0, 0, 0, 0, 0, 0, 0, 0],
        "h1": [0, 0, 0, 0, 0, 0, 0, 0],
        "a2": [0, 0, 0, 0, 0, 0, 0, 0],
        "b2": [0, 0, 0, 0, 0, 0, 0, 0],
        "c2": [0, 0, 0, 0, 0, 0, 0, 0],
        "d2": [0, 0, 0, 0, 0, 0, 0, 0],
        "e2": [0, 0, 0, 0, 0, 0, 0, 0],
        "f2": [0, 0, 0, 0, 0, 0, 0, 0],
        "g2": [0, 0, 0, 0, 0, 0, 0, 0],
        "h2": [0, 0, 0, 0, 0, 0, 0, 0],
    }

    for index, row in df[df['played_as'] == "white"].iterrows():
        di[row["first_move"][2] + "1"][int(row["first_move"][3]) - 1] += 1
        di[row["first_move"][2] + "2"][int(row["first_move"][3]) - 1] += 1

    row = ['1', '2', '3', '4', '5', '6', '7', '8']
    a = di["a2"]
    b = di["b2"]
    c = di["c2"]
    d = di["d2"]
    e = di["e2"]
    f = di["f2"]
    g = di["g2"]
    h = di["h2"]

    mlist = [row, a, b, c, d, e, f, g, h]
    for lists in mlist:
        lists.reverse()

    board_open = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d,
                             'e': e, 'f': f, 'g': g, 'h': h}, index=row)

    plt.figure(figsize=(10, 10))
    board = sns.heatmap(board_open, cmap='Reds', square=True, linewidths=.1, linecolor='black', vmax=vmax)
    board.set_title('Тепловая карта конечных полей для белых', size=18, y=1.05)
    
    output_path = os.path.join('player_data', username, "heatmap_landing_white.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return board_open.values.max()

def bl_heatmap_beg(df: pd.DataFrame, username: str, vmax: Optional[float] = None) -> float:
    """Создать тепловую карту начальных полей для чёрных"""
    # Инициализировать матрицу доски напрямую (8x8)
    board_matrix = np.zeros((8, 8))

    # Обрабатывать только партии, где пользователь играл чёрными
    black_games = df[df['played_as'] == "black"]
    
    for _, row in black_games.iterrows():
        pgn = chess.pgn.read_game(io.StringIO(row['PGN']))
        if pgn:
            moves = list(pgn.mainline_moves())
            if len(moves) >= 2:  # Убедиться, что есть как минимум 2 хода (первый ход белых и ответ чёрных)
                black_move = moves[1]  # Получить первый ход чёрных (второй ход в партии)
                from_square = chess.square_name(black_move.from_square)
                
                # Получить индексы вертикали и горизонтали (0-7)
                file_idx = ord(from_square[0]) - ord('a')  # Преобразовать a-h в 0-7
                rank_idx = 8 - int(from_square[1])  # Преобразовать 1-8 в 7-0 (перевёрнуто для отображения)
                
                # Увеличить счёт в матрице
                board_matrix[rank_idx][file_idx] += 1

    # Создать DataFrame для seaborn
    board_df = pd.DataFrame(
        board_matrix,
        index=['8', '7', '6', '5', '4', '3', '2', '1'],  # Горизонтали сверху вниз
        columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']  # Вертикали слева направо
    )

    plt.figure(figsize=(10, 10))
    board = sns.heatmap(board_df, cmap='Blues', square=True, linewidths=.1, linecolor='black', vmax=vmax)
    board.set_title('Тепловая карта начальных полей для чёрных', size=18, y=1.05)
    
    output_path = os.path.join('player_data', username, "heatmap_starting_black.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return board_matrix.max()

def bl_heatmap_end(df: pd.DataFrame, username: str, vmax: Optional[float] = None) -> float:
    """Создать тепловую карту конечных полей для чёрных"""
    # Инициализировать матрицу доски напрямую (8x8)
    board_matrix = np.zeros((8, 8))

    # Обрабатывать только партии, где пользователь играл чёрными
    black_games = df[df['played_as'] == "black"]
    
    for _, row in black_games.iterrows():
        pgn = chess.pgn.read_game(io.StringIO(row['PGN']))
        if pgn:
            moves = list(pgn.mainline_moves())
            if len(moves) >= 2:  # Убедиться, что есть как минимум 2 хода (первый ход белых и ответ чёрных)
                black_move = moves[1]  # Получить первый ход чёрных (второй ход в партии)
                to_square = chess.square_name(black_move.to_square)
                
                # Получить индексы вертикали и горизонтали (0-7)
                file_idx = ord(to_square[0]) - ord('a')  # Преобразовать a-h в 0-7
                rank_idx = 8 - int(to_square[1])  # Преобразовать 1-8 в 7-0 (перевёрнуто для отображения)
                
                # Увеличить счёт в матрице
                board_matrix[rank_idx][file_idx] += 1

    # Создать DataFrame для seaborn
    board_df = pd.DataFrame(
        board_matrix,
        index=['8', '7', '6', '5', '4', '3', '2', '1'],  # Горизонтали сверху вниз
        columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']  # Вертикали слева направо
    )

    plt.figure(figsize=(10, 10))
    board = sns.heatmap(board_df, cmap='Blues', square=True, linewidths=.1, linecolor='black', vmax=vmax)
    board.set_title('Тепловая карта конечных полей для чёрных', size=18, y=1.05)
    
    output_path = os.path.join('player_data', username, "heatmap_landing_black.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return board_matrix.max()

def create_combined_heatmaps(df: pd.DataFrame, username: str) -> None:
    """Создать и объединить все тепловые карты"""
    try:
        logger.info("Создание комбинированных тепловых карт...")
        
        # Первый проход для получения максимальных значений для согласованных шкал
        white_start_max = wh_heatmap_beg(df, username, None)
        white_end_max = wh_heatmap_end(df, username, None)
        white_max = max(white_start_max, white_end_max)
        
        black_start_max = bl_heatmap_beg(df, username, None)
        black_end_max = bl_heatmap_end(df, username, None)
        black_max = max(black_start_max, black_end_max)
        
        # Второй проход с согласованными шкалами
        wh_heatmap_beg(df, username, white_max)
        wh_heatmap_end(df, username, white_max)
        bl_heatmap_beg(df, username, black_max)
        bl_heatmap_end(df, username, black_max)
        
        # Создать комбинированную фигуру для тепловых карт белых
        plt.figure(figsize=(20, 10))
        
        # Начальные поля белых
        plt.subplot(1, 2, 1)
        img1 = plt.imread(os.path.join('player_data', username, "heatmap_starting_white.png"))
        plt.imshow(img1)
        plt.axis('off')
        
        # Конечные поля белых
        plt.subplot(1, 2, 2)
        img2 = plt.imread(os.path.join('player_data', username, "heatmap_landing_white.png"))
        plt.imshow(img2)
        plt.axis('off')
        
        plt.suptitle("Тепловые карты начальных и конечных полей для белых", size=24, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join('player_data', username, "heatmap_combined_white.png"), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # Создать комбинированную фигуру для тепловых карт чёрных
        plt.figure(figsize=(20, 10))
        
        # Начальные поля чёрных
        plt.subplot(1, 2, 1)
        img3 = plt.imread(os.path.join('player_data', username, "heatmap_starting_black.png"))
        plt.imshow(img3)
        plt.axis('off')
        
        # Конечные поля чёрных
        plt.subplot(1, 2, 2)
        img4 = plt.imread(os.path.join('player_data', username, "heatmap_landing_black.png"))
        plt.imshow(img4)
        plt.axis('off')
        
        plt.suptitle("Тепловые карты начальных и конечных полей для чёрных", size=24, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join('player_data', username, "heatmap_combined_black.png"), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info("Комбинированные тепловые карты успешно созданы")
        
    except Exception as e:
        logger.error(f"Ошибка в комбинированных тепловых картах: {str(e)}")
        raise

def driver_fn(username: str) -> None:
    """Основная функция для визуализаций"""
    try:
        logger.info(f"Запуск анализа игр для пользователя: {username}")
        
        # Настройка matplotlib для неинтерактивного бэкенда
        plt.switch_backend('Agg')
        
        if not check_dependencies():
            raise Exception("Отсутствуют необходимые зависимости")
            
        # Убедиться, что директория существует и является абсолютной
        user_dir = os.path.join('player_data', username)
        os.makedirs(user_dir, exist_ok=True)
        
        logger.info(f"Загрузка данных для {username}")
        df_path = os.path.join(user_dir, 'chess_dataset.csv')
        
        if not os.path.exists(df_path):
            raise FileNotFoundError(f"Набор данных не найден по пути {df_path}. Убедитесь, что данные сначала загружены.")
            
        df = pd.read_csv(df_path)
        if len(df) == 0:
            raise ValueError("Набор данных пуст")
            
        logger.info(f"Загружено {len(df)} партий")
        
        # Добавить производные столбцы
        logger.info("Обработка данных...")
        df["rating_difference"] = df["player_rating"] - df["opponent_rating"]
        df["moves"] = pd.to_numeric(df["moves"], errors='coerce')
        
        # Генерация всех визуализаций
        visualization_functions = [
            (fight, "анализ длины партий"),
            (wh_countplot, "анализ дебютов белыми"),
            (bl_countplot, "анализ дебютов чёрными"),
            (most_used_wh, "топ-3 первых ходов"),
            (most_used_bl, "топ-3 ответов чёрными"),
            (create_rating_ladder, "прогресс рейтинга"),
            (create_time_control_dist, "распределение контроля времени"),
            (create_color_results, "результаты по цвету"),
            (create_top_5_openings, "анализ топ-5 дебютов"),
            (create_overall_results, "общие результаты"),
            (create_overall_results_pie, "круговая диаграмма общих результатов"),
            (wh_heatmap_beg, "начальные поля белыми"),
            (wh_heatmap_end, "конечные поля белыми"),
            (bl_heatmap_beg, "начальные поля чёрными"),
            (bl_heatmap_end, "конечные поля чёрными"),
            (create_combined_heatmaps, "комбинированные тепловые карты")
        ]
        
        successful_visualizations = []
        failed_visualizations = []
        
        for viz_func, description in visualization_functions:
            try:
                logger.info(f"Запуск {description}...")
                
                # Очистить память перед каждой визуализацией
                plt.close('all')
                gc.collect()
                
                viz_func(df, username)
                logger.info(f"Завершён {description}")
                successful_visualizations.append(description)
                
            except Exception as e:
                logger.error(f"Ошибка в {description}: {str(e)}")
                logger.error(f"Тип ошибки: {type(e).__name__}")
                import traceback
                logger.error(f"Трассировка: {traceback.format_exc()}")
                failed_visualizations.append(f"{description} ({type(e).__name__})")
                continue
                
        # Вывести сводку
        logger.info("\nСводка визуализаций:")
        logger.info(f"Успешно выполнено: {len(successful_visualizations)}")
        for viz in successful_visualizations:
            logger.info(f"✓ {viz}")
            
        if failed_visualizations:
            logger.info(f"\nНеудачных визуализаций: {len(failed_visualizations)}")
            for viz in failed_visualizations:
                logger.info(f"✗ {viz}")
                
    except Exception as e:
        logger.error(f"Критическая ошибка в процессе визуализации: {str(e)}")
        logger.error(f"Тип ошибки: {type(e).__name__}")
        import traceback
        logger.error(f"Трассировка: {traceback.format_exc()}")
        raise  # Повторно поднять ошибку, чтобы она была передана дальше

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Укажите имя пользователя в качестве аргумента")
        sys.exit(1)
    try:
        driver_fn(sys.argv[1])
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}")
        sys.exit(1)
        