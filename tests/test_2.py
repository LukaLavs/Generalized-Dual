import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from pathlib import Path

file_path = Path("setup.py")
try:
    file_path.read_text(encoding="utf-8")
    print("File is UTF-8")
except UnicodeDecodeError:
    print("File is NOT UTF-8")