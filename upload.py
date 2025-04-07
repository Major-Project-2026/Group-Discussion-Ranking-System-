import tkinter as tk
from tkinter import filedialog

def select_audio_file():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title="Select an Audio File",
        filetypes=[("Audio Files", "*.wav;*.mp3;*.flac;*.m4a")]
    )
    return file_path

if __name__ == "__main__":
    file_path = select_audio_file()
    if file_path:
        print("Selected file:", file_path)
        with open("selected_file.txt", "w") as f:
            f.write(file_path)
    else:
        print("No file selected.")
