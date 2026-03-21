import concurrent.futures
import os
import subprocess
from typing import List, Tuple
import sys
import time

def run_visualization_module(args: Tuple[str, str, str, str]) -> Tuple[str, float, str, str]:
    """Запускает отдельный модуль визуализации и возвращает результат"""
    python_exe, module_file, username, description = args
    module_start = time.time()
    
    try:
        result = subprocess.run(
            [python_exe, module_file, username],
            check=True,
            capture_output=True,
            text=True,
            timeout=60  # таймаут 60 секунд
        )
        duration = time.time() - module_start
        return (description, duration, result.stdout, "")
    except subprocess.TimeoutExpired:
        return (description, time.time() - module_start, "", "Превышен таймаут процесса (60 секунд)")
    except Exception as e:
        return (description, time.time() - module_start, "", str(e))

def visualize_data(username: str) -> None:
    """Визуализирует шахматные данные с использованием параллельной обработки"""
    try:
        print("\nЗапуск процесса визуализации...")
        start_time = time.time()
        
        modules: List[Tuple[str, str]] = [
            ("unt.py", "анализ игр"),
            ("heatmap1.py", "тепловая карта 1"),
            ("heatmap2.py", "тепловая карта 2"),
            ("heatmap3.py", "корреляционный анализ")
        ]
        
        python_exe = sys.executable
        print(f"Используется Python: {python_exe}")
        
        # Подготовка аргументов для параллельной обработки
        args = [(python_exe, module[0], username, module[1]) for module in modules]
        
        # Запуск визуализаций параллельно
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_visualization_module, arg) for arg in args]
            
            for future in concurrent.futures.as_completed(futures):
                description, duration, stdout, error = future.result()
                if error:
                    print(f"✗ Ошибка в {description} после {duration:.2f} секунд:")
                    print(f"Ошибка: {error}")
                else:
                    print(f"✓ Завершён {description} за {duration:.2f} секунд")
                    if stdout:
                        print(f"Вывод:\n{stdout}")
        
        total_duration = time.time() - start_time
        print(f"\nПроцесс визуализации завершён за {total_duration:.2f} секунд")
        
    except Exception as e:
        print(f"\n✗ Критическая ошибка в процессе визуализации: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) > 1:
        visualize_data(sys.argv[1])
    else:
        print("Пожалуйста, укажите имя пользователя в качестве аргумента")
        