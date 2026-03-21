# Standard library imports
import os
import io
import logging
import time
from typing import Dict, Any

# Third-party imports
import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="Chess Analysis Dashboard",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Local imports
import get_data as gd  # noqa: E402
import visualize as viz  # noqa: E402
import prediction as pred  # noqa: E402

# Additional third-party imports with error handling
try:
    import pandas as pd  # noqa: F401 - используется в модулях
    import plotly.express as px  # noqa: F401 - используется в модулях
    import seaborn as sns  # noqa: F401 - используется в модулях
    import matplotlib.pyplot as plt  # noqa: F401 - используется в модулях
    from PIL import Image  # noqa: F401 - используется в модулях
    import chess.pgn  # noqa: F401 - используется в модулях
except ImportError as e:
    raise ImportError(f"""
    Missing required packages. Please install them using:
    pip install streamlit pandas plotly seaborn matplotlib pillow python-chess
    
    Error: {str(e)}
    """)

try:
    import cairosvg
except ImportError:
    raise ImportError(
        "CairoSVG is required. Install it with: pip install cairosvg"
    )

def check_dependencies() -> None:
    """Check if all visualization dependencies are installed"""
    required_packages = {
        'plotly': 'plotly',
        'seaborn': 'seaborn',
        'matplotlib': 'matplotlib',
        'chess': 'python-chess'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)
            
    if missing_packages:
        st.error(f"""
        Missing required packages: {', '.join(missing_packages)}
        Please install them using:
        pip install {' '.join(missing_packages)}
        """)
        st.stop()

def init_session_state() -> None:
    """Initialize session state variables"""
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

def render_home_tab() -> None:
    """Render the home tab content"""
    # Adjust column ratios for better alignment
    col1, col2, col3 = st.columns([0.5, 6, 0.5])
    
    # Add CSS for vertical alignment
    st.markdown("""
        <style>
        .logo-container {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            padding-top: 15px;  /* Adjust this value to align with title */
        }
        </style>
    """, unsafe_allow_html=True)
    
    with col1:
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        st.image("img_files/logo.ico", width=40)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <h1 style='text-align: center; color: white; margin: 0;'>
                Статистический анализ шахматиста с использованием Data Science Pipeline
            </h1> <hr>
            """, 
            unsafe_allow_html=True
        )
    with col3:
        st.markdown('<div class="logo-container">', unsafe_allow_html=True)
        st.image("img_files/logo.ico", width=40)
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.header("Введение")
    st.write(
        "«На рынке нет инструмента, который бы обеспечивал углубленный анализ "
        "всех партий игрока. Данное программное обеспечение предоставляет шахматистам инструмент для "
        "улучшения своей игры с помощью методов машинного обучения и "
        "науки о данных. Они работают путем изучения предыдущих партий игрока "
        "и получения полезных данных, которые помогут им учиться на своих прошлых ошибках»."
    )
    
    # Give more space to the images column
    col1, col2 = st.columns([1.5, 1])
    with col1:
        render_tutorial()
    with col2:
        # Add custom CSS to increase image size
        st.markdown("""
            <style>
            .stImage > img {
                max-width: 100%;
                height: auto;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Display Magnus
        st.image("img_files/magnus.png", use_container_width=True)
        
        # Create nested columns with more width
        st.markdown("<div style='padding: 10px 0px;'></div>", unsafe_allow_html=True)  # Add spacing
        subcol1, subcol2 = st.columns([1, 1])
        with subcol1:
            st.image("img_files/bobby.jpg", use_container_width=True)
        with subcol2:
            st.image("img_files/kasparov.png", use_container_width=True)

def render_tutorial() -> None:
    """Render the tutorial section"""
    st.header("Учебное пособие")
    st.write(
        "Нажмите на вкладки выше, чтобы переключаться между различными режимами работы.\n\n"
        "1. Ввод данных пользователя:\n"
        "   - Предоставляет полный статистический анализ данных, основанный на ваших предыдущих шахматных партиях.\n"
        "   - Просто введите свой логин на chess.com и подождите несколько секунд.\n"
        "   - Программа загрузит все ваши данные через API.\n\n"
        "2. Анализ игрока:\n"
        "   - Предоставляет подробный анализ ваших данных с помощью интерактивных диаграмм.\n"
        "   - Содержит подробные инструкции по интерпретации диаграмм.\n\n"
        "3. Прогноз на игру:\n"
        "   - Введите ваше имя пользователя и имя пользователя вашего оппонента.\n"
        "   - Посмотрите прогноз исхода вашей игры.\n"
        "   - Использует логистическую регрессию на основе данных предыдущих партий."
    )

def check_cached_analysis(username: str) -> bool:
    """Check if all required files exist in the player's cache directory"""
    required_files = [
        "chess_dataset.csv",
        "chess_dataset_adv.csv",
        "result_as_wh.png",
        "result_as_bl.png",
        "fight.png",
        "rating_ladder_red.png",
        "time_class.png",
        "overall_results.png",
        "result_top_5_wh.png",
        "result_top_5_bl.png",
        "corr_heatmap.png"
    ]
    
    player_dir = os.path.join('player_data', username)
    if not os.path.exists(player_dir):
        return False
        
    return all(os.path.exists(os.path.join(player_dir, file)) for file in required_files)

def render_user_input_tab() -> None:
    """Render the user input tab content"""
    st.title("Ввод данных пользователя")
    
    username = st.text_input(
        "Введите ваше имя пользователя Chess.com:",
        placeholder="ваше_имя_пользователя"
    )
    
    if st.button("Анализировать игры"):
        if username:
            # Create placeholder for progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()
            log_output = st.empty()
            
            try:
                # Update status
                status_text.text("🔄 Загрузка игр с Chess.com...")
                progress_bar.progress(10)
                
                # Create StringIO to capture logs
                log_capture = io.StringIO()
                log_handler = logging.StreamHandler(log_capture)
                log_handler.setFormatter(logging.Formatter('%(message)s'))
                logging.getLogger().addHandler(log_handler)
                
                # Download data
                gd.driver_fn(username)
                progress_bar.progress(40)
                status_text.text("🔄 Обработка данных игр...")
                
                # Update log display
                log_output.code(log_capture.getvalue())
                
                # Visualize data
                status_text.text("🔄 Создание визуализаций...")
                progress_bar.progress(70)
                viz.visualize_data(username)
                
                # Complete
                progress_bar.progress(100)
                status_text.text("✅ Анализ завершен!")
                
                # Update session state
                st.session_state.username = username
                st.session_state.analysis_complete = True
                
                # Show success message with instructions
                st.success("""
                ✅ Анализ завершен! 
                
                Перейдите на вкладку Анализ игрока, чтобы просмотреть результаты.
                """)
                
                # Keep success message visible
                time.sleep(2)
                
            except Exception as e:
                progress_bar.empty()
                error_msg = str(e)
                if "User not found" in error_msg:
                    st.error("❌ " + error_msg)
                elif "No games found" in error_msg:
                    st.error("❌ " + error_msg)
                elif "Unable to connect" in error_msg:
                    st.error("🌐 " + error_msg)
                else:
                    st.error("❌ Произошла ошибка во время анализа: " + error_msg)
            finally:
                # Remove log handler
                logging.getLogger().removeHandler(log_handler)
        else:
            st.warning("Пожалуйста, введите имя пользователя")

def render_analysis_tab() -> None:
    """Render the player analysis tab content"""
    st.title("Анализ игрока")
    
    if st.session_state.analysis_complete and st.session_state.username:
        username = st.session_state.username
        if os.path.exists(os.path.join('player_data', username, "corr_heatmap.png")):
            render_analysis_content(username)
    else:
        st.info("Пожалуйста, сначала выполните анализ игрока на вкладке Ввод данных пользователя.")

def render_analysis_content(username: str) -> None:
    """Render the analysis content for a given username"""
    try:
        # Top Openings as White
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Топ-20 самых популярных дебютов белыми")
        st.write("Это частотная столбчатая диаграмма топ-20 самых популярных дебютов пользователя белыми.")
        
        top_op_wh_path = os.path.join('player_data', username, "top_op_wh.png")
        if os.path.exists(top_op_wh_path):
            st.image(top_op_wh_path)
        else:
            st.error("Визуализация анализа дебютов белыми недоступна")
            
        # Top Openings as Black
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Топ-20 самых популярных дебютов черными")
        st.write("Это частотная столбчатая диаграмма топ-20 самых популярных дебютов пользователя черными.")
        
        top_op_bl_path = os.path.join('player_data', username, "top_op_bl.png")
        if os.path.exists(top_op_bl_path):
            st.image(top_op_bl_path)
        else:
            st.error("Визуализация анализа дебютов черными недоступна")
        
        # Top First Moves
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Топ-3 первых хода белыми")
        cols = st.columns([1.2, 1.2, 1.2, 0.1])
        
        # Check for move visualizations
        for i, col in enumerate(cols[:-1], 1):
            svg_path = os.path.join('player_data', username, f'top_opening_move_as_white_{i}.svg')
            png_path = os.path.join('player_data', username, f'top_opening_move_as_white_{i}.png')
            
            with col:
                if os.path.exists(svg_path):
                    try:
                        # Convert SVG to PNG if not already done
                        if not os.path.exists(png_path):
                            cairosvg.svg2png(
                                url=svg_path,
                                write_to=png_path,
                                scale=2.0  # Increase quality
                            )
                        # Display PNG image
                        st.image(png_path)
                    except Exception as e:
                        st.error(f"Ошибка конвертации шахматной доски {i}: {str(e)}")
                else:
                    st.warning(f"Визуализация хода {i} недоступна")
        
        # Top Black Replies
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Топ-3 ответных хода черными")
        cols = st.columns([1.2, 1.2, 1.2, 0.1])
        
        # Check for black reply visualizations
        for i, col in enumerate(cols[:-1], 1):
            svg_path = os.path.join('player_data', username, f'top_reply_move_as_black_{i}.svg')
            png_path = os.path.join('player_data', username, f'top_reply_move_as_black_{i}.png')
            
            with col:
                if os.path.exists(svg_path):
                    try:
                        # Convert SVG to PNG if not already done
                        if not os.path.exists(png_path):
                            cairosvg.svg2png(
                                url=svg_path,
                                write_to=png_path,
                                scale=2.0  # Increase quality
                            )
                        # Display PNG image
                        st.image(png_path)
                    except Exception as e:
                        st.error(f"Ошибка конвертации шахматной доски {i}: {str(e)}")
                else:
                    st.warning(f"Визуализация хода {i} недоступна")
        
        # Add heatmap visualizations
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Тепловые карты начальных ходов белыми и черными")
        st.write("Это тепловые карты первых ходов, сделанных белыми и черными. Более темные клетки означают более высокую частоту.")
        
        # White heatmaps
        white_heatmap_path = os.path.join('player_data', username, "heatmap_combined_white.png")
        if os.path.exists(white_heatmap_path):
            st.image(white_heatmap_path)
        else:
            st.warning("Тепловые карты для белых недоступны")
            
        # Black heatmaps
        black_heatmap_path = os.path.join('player_data', username, "heatmap_combined_black.png")
        if os.path.exists(black_heatmap_path):
            st.image(black_heatmap_path)
        else:
            st.warning("Тепловые карты для черных недоступны")
        
        # Results by Color (side by side)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.header("Результаты по цвету фигур")
        st.write("Это результаты партий, сыгранных пользователем.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Результаты белыми")
            st.write("Эта кольцевая диаграмма показывает соотношение побед/ничьих/поражений при игре белыми.")
            white_results_path = os.path.join('player_data', username, "result_as_wh.png")
            if os.path.exists(white_results_path):
                st.image(white_results_path)
            else:
                st.warning("Визуализация результатов белыми недоступна")
                
        with col2:
            st.subheader("Результаты черными")
            st.write("Эта кольцевая диаграмма показывает соотношение побед/ничьих/поражений при игре черными.")
            black_results_path = os.path.join('player_data', username, "result_as_bl.png")
            if os.path.exists(black_results_path):
                st.image(black_results_path)
            else:
                st.warning("Визуализация результатов черными недоступна")
        
        # Add other visualizations with existence checks
        visualizations = [
            ("fight.png", "Насколько упорно игрок сопротивляется при проигрыше", 
             "Это все партии, которые пользователь проиграл. Большее количество ходов в партии означает, что пользователь оказал достойное сопротивление перед сдачей. Меньшее количество ходов указывает на то, что игрок допустил грубую ошибку в начале партии."),
            ("time_class.png", "Распределение по контролю времени", 
             "Эта круговая диаграмма показывает распределение различных контролей времени в ваших партиях."),
            ("rating_ladder_red.png", "Прогресс рейтинга", 
             "Этот график показывает прогресс вашего рейтинга за последние 150 рейтинговых партий в различных контролях времени."),
            ("overall_results.png", "Общие результаты", 
             "Частотный график результатов всех партий, сыгранных пользователем на сайте."),
            ("overall_results_pie.png", "Распределение общих результатов (круговая диаграмма)", 
             "Круговая диаграмма, показывающая распределение всех результатов партий."),
            ("result_top_5_wh.png", "Анализ силы и слабости дебютов белыми", 
             "Эти графики очень важны для анализа сильных и слабых сторон. Более длинная красная полоса указывает на дебют, который пользователь играл чаще всего, но также и проигрывал чаще всего. Самая длинная зеленая полоса указывает на самый сильный и часто играемый дебют."),
            ("result_top_5_bl.png", "Анализ силы и слабости дебютов черными", 
             "Эти графики очень важны для анализа сильных и слабых сторон. Более длинная красная полоса указывает на дебют, который пользователь играл чаще всего, но также и проигрывал чаще всего. Самая длинная зеленая полоса указывает на самый сильный и часто играемый дебют."),
            ("corr_heatmap.png", "Корреляционная тепловая карта", 
             "Эта тепловая карта показывает корреляции между различными числовыми аспектами ваших партий.")
        ]
        
        for viz_file, title, description in visualizations:
            viz_path = os.path.join('player_data', username, viz_file)
            if os.path.exists(viz_path):
                st.markdown("<hr>", unsafe_allow_html=True)  # Add horizontal rule before heading
                st.header(title)
                st.write(description)
                st.image(viz_path)
            else:
                st.warning(f"Визуализация {title} недоступна")
                
    except Exception as e:
        st.error(f"Ошибка при отображении содержимого анализа: {str(e)}")
        st.info("Некоторые визуализации могут быть недоступны")

def render_prediction_tab() -> None:
    """Render the game prediction tab content"""
    st.title("Прогноз на игру")
    
    col1, col2 = st.columns(2)
    with col1:
        user1 = st.text_input(
            "Введите ваше имя пользователя:",
            placeholder="ваше_имя"
        )
    with col2:
        user2 = st.text_input(
            "Введите имя пользователя оппонента:",
            placeholder="имя_оппонента"
        )
    
    if st.button("Предсказать исход игры"):
        if user1 and user2:
            # First check if the advanced dataset exists
            adv_dataset_path = os.path.join('player_data', user1, 'chess_dataset_adv.csv')
            if not os.path.exists(adv_dataset_path):
                st.error("📊 Ошибка обработки шахматных данных. Пожалуйста, сначала выполните анализ на вкладке Ввод данных пользователя.")
                return
                
            with st.spinner("Построение модели логистической регрессии..."):
                try:
                    results = pred.predict(user1, user2)
                    display_prediction_results(results)
                except Exception as e:
                    error_msg = str(e)
                    if "Could not find blitz rating" in error_msg:
                        st.error("⚠️ " + error_msg + "\nОба игрока должны играть в блиц-партии на Chess.com, чтобы использовать эту функцию.")
                    elif "Error accessing Chess.com API" in error_msg:
                        st.error("🌐 Не удалось получить доступ к API Chess.com. Пожалуйста, проверьте подключение к интернету и повторите попытку.")
                    else:
                        st.error("❌ " + error_msg)
        else:
            st.warning("Пожалуйста, введите оба имени пользователя")

def display_prediction_results(results: Dict[str, Any]) -> None:
    """Display the prediction results"""
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Ваш рейтинг", results["user_rating"])
        st.metric("Разница в рейтинге", results["rating_diff"])
    with col2:
        st.metric("Рейтинг оппонента", results["opp_rating"])
    
    with st.expander("Просмотреть детали модели", expanded=False):
        st.code(results["summ1"], language="text")
        st.write(results["ord_acc"])
    
    st.success(results["result"])

def render_about_tab() -> None:
    """Render the about tab content"""
    st.title("О программе")
    
    # Create two columns with different widths
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
     
        """)
        
        # Add resume display with a button
        resume_path = ""
        # Keep PDFbyte variable used, to avoid unused variable warnings and make button active
        PDFbyte = None
        if os.path.exists(resume_path):
            with open(resume_path, "rb") as pdf_file:
                PDFbyte = pdf_file.read()
        if PDFbyte is not None:
            st.download_button("Скачать резюме", data=PDFbyte, file_name="resume.pdf", mime="application/pdf")
        else:
            st.info("Файл резюме не найден.")
        
        st.markdown("""
          
        """)
    
    with col2:
        # Add Gukesh's image and congratulatory message
        st.image("img_files/gukesh.jpg", width=450)
        st.markdown("""
            <p style='text-align: center; font-style: italic; font-size: 1.2em; margin-top: 10px;'>
               Поздравляем Гукеша с тем, что он стал самым молодым чемпионом мира по шахматам!
            </p>
        """, unsafe_allow_html=True)

def main() -> None:
    """Main application entry point"""
    # Check dependencies before starting the app
    check_dependencies()
    
    # Initialize session state
    init_session_state()

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        [data-testid="stToolbar"] { display: none; }

        .main {
            padding: 0rem 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            background-color: #262730;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0F1116;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create tabs
    tabs = st.tabs([
        "Дом",
        "Ввод данных пользователем",
        "Анализ игрока",
        "Прогноз на игру",
        "О программе"
    ])

    # Render each tab
    with tabs[0]:
        render_home_tab()
    with tabs[1]:
        render_user_input_tab()
    with tabs[2]:
        render_analysis_tab()
    with tabs[3]:
        render_prediction_tab()
    with tabs[4]:
        render_about_tab()

if __name__ == "__main__":
    main()
    