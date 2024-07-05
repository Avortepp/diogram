import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tkinter as tk
from tkinter import messagebox

# Function to create a database and tables
def create_database():
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS data
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       value REAL,
                       note TEXT)''')  # Добавлен столбец 'note'
    cursor.execute('''CREATE TABLE IF NOT EXISTS metrics
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       r2 REAL,
                       mae REAL,
                       mse REAL)''')
    conn.commit()
    conn.close()
# Function to insert data into the 'data' table
def insert_data(data):
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()
    cursor.executemany('INSERT INTO data (value, note) VALUES (?, ?)', data)
    conn.commit()
    conn.close()

# Function to train the model and evaluate its accuracy
def train_and_evaluate():
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()

    # Fetching data from the 'data' table
    cursor.execute('SELECT value FROM data')
    results = cursor.fetchall()
    values = np.array(results).flatten()

    # Simple machine learning with scikit-learn
    X = np.arange(len(values)).reshape(-1, 1)
    y = values
    model = LinearRegression()
    model.fit(X, y)
    predicted_values = model.predict(X)

    # Evaluating prediction accuracy
    r2 = r2_score(y, predicted_values)
    mae = mean_absolute_error(y, predicted_values)
    mse = mean_squared_error(y, predicted_values)

    # Inserting metrics into the 'metrics' table
    cursor.execute('INSERT INTO metrics (r2, mae, mse) VALUES (?, ?, ?)', (r2, mae, mse))
    conn.commit()

    conn.close()

    return r2, mae, mse, values, predicted_values

# Function to display metrics from the 'metrics' table
def print_metrics():
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()

    # Fetching metrics data from the 'metrics' table
    cursor.execute('SELECT * FROM metrics ORDER BY id DESC LIMIT 1')
    result = cursor.fetchone()

    if result:
        messagebox.showinfo("Metrics", f"R² Score: {result[1]:.2f}\nMean Absolute Error (MAE): {result[2]:.2f}\nMean Squared Error (MSE): {result[3]:.2f}")
    else:
        messagebox.showinfo("Metrics", "No metrics data available")

    conn.close()

# Function to plot data and visualize predictions
def plot_data(values, predicted_values):
    plt.figure(figsize=(8, 6))
    plt.plot(values, label='Original Data')
    plt.plot(predicted_values, label='Prediction')
    plt.title('Data Prediction')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to check the current state of the database
def check_database_state():
    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()

    # Checking tables in the database
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    table_names = ""
    if tables:
        for table in tables:
            table_names += table[0] + "\n"
    else:
        table_names = "No tables in the database"

    messagebox.showinfo("Tables in Database", table_names)

    conn.close()

# Main block for GUI interaction
if __name__ == "__main__":
    # Creating the database and tables (if they don't exist)
    create_database()

    # Creating the main application window
    root = tk.Tk()
    root.title("Data Prediction")

    # Frame for data input
    frame_data = tk.Frame(root)
    frame_data.pack(pady=10)

    # Label and entry for data input
    tk.Label(frame_data, text="Data to insert (comma-separated):").pack(side=tk.LEFT)
    entry_data = tk.Entry(frame_data, width=30)
    entry_data.pack(side=tk.LEFT)

    # Label and entry for note input
    tk.Label(frame_data, text="Note:").pack(side=tk.LEFT)
    entry_note = tk.Entry(frame_data, width=30)
    entry_note.pack(side=tk.LEFT)

    # Function for the "Add Data" button
    def add_data():
        data_str = entry_data.get().strip()
        note_str = entry_note.get().strip()  # Fetching note input
        if data_str:
            try:
                data = [(float(x.strip()), note_str) for x in data_str.split(',')]  # Adding note to data
                insert_data(data)
                messagebox.showinfo("Success", "Data successfully added to the database")
            except ValueError:
                messagebox.showerror("Error", "Error inserting data. Make sure the data is entered correctly.")
            entry_data.delete(0, tk.END)
            entry_note.delete(0, tk.END)  # Clearing note entry
        else:
            messagebox.showwarning("Warning", "Enter data to add")

    # Button to add data
    btn_add_data = tk.Button(frame_data, text="Add Data", command=add_data)
    btn_add_data.pack(side=tk.LEFT, padx=10)

    # Frame for "Train Model" and "Show Metrics" buttons
    frame_buttons = tk.Frame(root)
    frame_buttons.pack(pady=20)

    # Function for the "Train Model" button
    def train_model():
        r2, mae, mse, values, predicted_values = train_and_evaluate()
        print_metrics()
        plot_data(values, predicted_values)

    # Button to train the model
    btn_train_model = tk.Button(frame_buttons, text="Train Model", command=train_model)
    btn_train_model.pack(side=tk.LEFT, padx=10)

    # Function for the "Show Metrics" button
    def show_metrics():
        print_metrics()

    # Button to show metrics
    btn_show_metrics = tk.Button(frame_buttons, text="Show Metrics", command=show_metrics)
    btn_show_metrics.pack(side=tk.LEFT, padx=10)

    # Frame for the "Check Database State" button
    frame_check_db = tk.Frame(root)
    frame_check_db.pack(pady=10)

    # Button to check the database state
    btn_check_db = tk.Button(frame_check_db, text="Check Database State", command=check_database_state)
    btn_check_db.pack()

    # Running the main event loop
    root.mainloop()