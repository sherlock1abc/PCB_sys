import sqlite3
import pandas as pd
from datetime import datetime

# 数据库文件名
DB_NAME = "detection_history.db"

def init_db():
    """初始化数据库，创建数据表（如果不存在）"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            detect_time TEXT NOT NULL,
            filename TEXT NOT NULL,
            defect_count INTEGER NOT NULL,
            duration REAL NOT NULL,
            fps REAL NOT NULL,
            details TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_record(filename, defect_count, duration, fps, details=""):
    """保存一条检测记录到数据库"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO history (detect_time, filename, defect_count, duration, fps, details)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (detect_time, filename, defect_count, duration, fps, details))
    conn.commit()
    conn.close()

def get_all_records():
    """从数据库读取所有记录，返回一个 DataFrame"""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM history ORDER BY detect_time DESC", conn)
    conn.close()
    return df