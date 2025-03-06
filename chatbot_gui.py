import tkinter as tk  # Import tkinter library
from chatbot import get_response_from_bot  # Import get_response_from_bot function

root = tk.Tk()  # Create main window
root.title("Chatbot")  # Set window title

# Function to send message
def send_message():
    user_message = entry_box.get("1.0", tk.END).strip()  # Get user message
    chat_log.config(state=tk.NORMAL)  # Enable chat log
    chat_log.insert(tk.END, "You: " + user_message + '\n\n')  # Insert user message into chat log
    chat_log.config(foreground="#442265", font=("Verdana", 12))  # Configure chat log
    response