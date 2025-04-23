import sys
import os
import time
import threading
import csv
import uuid
import serial
import serial.tools.list_ports
import numpy as np
import git
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Configuration Constants
DATA_DIR = "recordings"
GITHUB_REPO_PATH = "git@github.com:zubayertahmid/air-writing-data.git"  # Update this with your GitHub repo path
BENGALI_ALPHABETS = ["অ", "আ", "ই", "ঈ", "উ", "ঊ", "ঋ", "এ", "ঐ", "ও", "ঔ", 
                    "ক", "খ", "গ", "ঘ", "ঙ", "চ", "ছ", "জ", "ঝ", "ঞ", 
                    "ট", "ঠ", "ড", "ঢ", "ণ", "ত", "থ", "দ", "ধ", "ন", 
                    "প", "ফ", "ব", "ভ", "ম", "য", "র", "ল", "শ", "ষ", 
                    "স", "হ", "ড়", "ঢ়", "য়", "ৎ", "ং", "ঃ", "ঁ"]
BUFFER_SIZE = 100  # Number of data points to display in the graphg
SERIAL_BAUD_RATE = 115200

class SerialThread(QThread):
    data_received = pyqtSignal(list)
    connection_status = pyqtSignal(bool, str)
    
    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.running = False
        self.port_name = None

    def connect_to_device(self, port_name):
        try:
            self.port_name = port_name
            self.serial_port = serial.Serial(port_name, SERIAL_BAUD_RATE, timeout=1)
            self.running = True
            self.connection_status.emit(True, f"Connected to {port_name}")
            return True
        except Exception as e:
            self.connection_status.emit(False, f"Error: {str(e)}")
            return False

    def disconnect_device(self):
        if self.serial_port and self.serial_port.is_open:
            self.running = False
            self.serial_port.close()
            self.serial_port = None
            self.connection_status.emit(False, "Disconnected")

    def run(self):
        while self.running:
            try:
                if self.serial_port and self.serial_port.in_waiting:
                    line = self.serial_port.readline().decode('utf-8').strip()
                    data = [float(x) for x in line.split(',') if x]
                    if len(data) == 7:  # Ensure we have all 7 expected values
                        self.data_received.emit(data)
                time.sleep(0.01)  # Small delay to prevent CPU hogging
            except Exception as e:
                print(f"Serial read error: {str(e)}")
                self.connection_status.emit(False, f"Error: {str(e)}")
                self.running = False
                break

class DataGraph(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Data storage for plotting
        self.times = np.linspace(0, BUFFER_SIZE-1, BUFFER_SIZE)
        self.acc_x = np.zeros(BUFFER_SIZE)
        self.acc_y = np.zeros(BUFFER_SIZE)
        self.acc_z = np.zeros(BUFFER_SIZE)
        self.gyro_x = np.zeros(BUFFER_SIZE)
        self.gyro_y = np.zeros(BUFFER_SIZE)
        self.gyro_z = np.zeros(BUFFER_SIZE)
        
        self.acc_lines = None
        self.gyro_lines = None
        
        self.setup_plot()
        
    def setup_plot(self):
        self.axes.clear()
        self.acc_lines, = self.axes.plot(self.times, self.acc_x, 'r-', label='Acc X')
        self.acc_liney, = self.axes.plot(self.times, self.acc_y, 'g-', label='Acc Y')
        self.acc_linez, = self.axes.plot(self.times, self.acc_z, 'b-', label='Acc Z')
        self.gyro_linex, = self.axes.plot(self.times, self.gyro_x, 'r--', label='Gyro X')
        self.gyro_liney, = self.axes.plot(self.times, self.gyro_y, 'g--', label='Gyro Y')
        self.gyro_linez, = self.axes.plot(self.times, self.gyro_z, 'b--', label='Gyro Z')
        
        self.axes.set_ylim(-20, 20)  # Adjust based on your sensor range
        self.axes.set_xlabel('Time')
        self.axes.set_ylabel('Value')
        self.axes.legend(loc='upper right')
        self.axes.grid(True)
        self.fig.tight_layout()
        
    def update_data(self, new_data):
        # Shift data arrays and add new values
        self.acc_x = np.roll(self.acc_x, -1)
        self.acc_y = np.roll(self.acc_y, -1)
        self.acc_z = np.roll(self.acc_z, -1)
        self.gyro_x = np.roll(self.gyro_x, -1)
        self.gyro_y = np.roll(self.gyro_y, -1)
        self.gyro_z = np.roll(self.gyro_z, -1)
        
        self.acc_x[-1] = new_data[0]
        self.acc_y[-1] = new_data[1]
        self.acc_z[-1] = new_data[2]
        self.gyro_x[-1] = new_data[3]
        self.gyro_y[-1] = new_data[4]
        self.gyro_z[-1] = new_data[5]
        
        # Update the plot
        self.acc_lines.set_ydata(self.acc_x)
        self.acc_liney.set_ydata(self.acc_y)
        self.acc_linez.set_ydata(self.acc_z)
        self.gyro_linex.set_ydata(self.gyro_x)
        self.gyro_liney.set_ydata(self.gyro_y)
        self.gyro_linez.set_ydata(self.gyro_z)
        
        self.draw()

class DataRecorder:
    def __init__(self):
        self.recording = False
        self.data_buffer = []
        self.start_time = None
        self.user_id = None
        self.alphabet = None
        self.hand_preference = None
        
    def start_recording(self, user_id, alphabet, hand_preference):
        self.recording = True
        self.data_buffer = []
        self.start_time = time.time()
        self.user_id = user_id
        self.alphabet = alphabet
        self.hand_preference = hand_preference
        
    def add_data(self, data):
        if self.recording:
            timestamp = time.time() - self.start_time
            self.data_buffer.append([timestamp] + data)
    
    def stop_recording(self):
        self.recording = False
        return len(self.data_buffer) > 0
    
    def save_data(self):
        if not self.data_buffer:
            return False
        
        # Create directory structure if it doesn't exist
        alphabet_dir = os.path.join(DATA_DIR, self.alphabet)
        os.makedirs(alphabet_dir, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{self.alphabet}_{self.user_id}_{self.hand_preference}_{timestamp}.csv"
        filepath = os.path.join(alphabet_dir, filename)
        
        # Write data to CSV
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['timestamp', 'user_id', 'hand_preference', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
            
            # Write data rows
            for row in self.data_buffer:
                timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, _ = row  # Ignore x7
                writer.writerow([timestamp, self.user_id, self.hand_preference, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z])
        
        # Sync with GitHub
        try:
            self.sync_with_github(filepath, f"Added new recording: User ID {self.user_id}, Alphabet {self.alphabet}, {self.hand_preference} Hand")
            return True
        except Exception as e:
            print(f"GitHub sync error: {str(e)}")
            return False
    
    def sync_with_github(self, filepath, commit_message):
        try:
            repo = git.Repo(GITHUB_REPO_PATH)
            
            # Check if the file is already being tracked
            relative_path = os.path.relpath(filepath, GITHUB_REPO_PATH)
            repo.git.add(relative_path)
            
            # Commit and push
            repo.git.commit('-m', commit_message)
            origin = repo.remote(name='origin')
            origin.push()
            
            return True
        except Exception as e:
            print(f"GitHub sync error: {str(e)}")
            raise e

class UserManager:
    def __init__(self):
        self.users_file = os.path.join(DATA_DIR, "users.csv")
        self.users = {}
        self.load_users()
        
    def load_users(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        user_id, username = row[0], row[1]
                        self.users[username] = user_id
    
    def register_user(self, username):
        if username in self.users:
            return self.users[username]
        
        # Generate a new user ID
        user_id = str(uuid.uuid4())[:8]
        self.users[username] = user_id
        
        # Save to file
        with open(self.users_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if os.path.getsize(self.users_file) == 0:
                writer.writerow(['user_id', 'username'])
            writer.writerow([user_id, username])
        
        return user_id
    
    def get_user_id(self, username):
        if username in self.users:
            return self.users[username]
        return None
    
    def get_all_users(self):
        return [(username, user_id) for username, user_id in self.users.items()]

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Air Writing System")
        self.setGeometry(100, 100, 1000, 600)
        
        self.serial_thread = SerialThread()
        self.serial_thread.data_received.connect(self.update_data)
        self.serial_thread.connection_status.connect(self.update_connection_status)
        
        self.data_recorder = DataRecorder()
        self.user_manager = UserManager()
        
        self.setup_ui()
        
        # Start with device detection
        self.refresh_ports()
        
        # Setup timer for UI updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(100)  # Update every 100ms
    
    def setup_ui(self):
        # Main widget and layout
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        
        # Top section: Device connection
        connection_group = QtWidgets.QGroupBox("Device Connection")
        connection_layout = QtWidgets.QHBoxLayout()
        
        self.port_combo = QtWidgets.QComboBox()
        self.refresh_button = QtWidgets.QPushButton("Refresh")
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.status_label = QtWidgets.QLabel("Status: Disconnected")
        
        connection_layout.addWidget(QtWidgets.QLabel("Port:"))
        connection_layout.addWidget(self.port_combo)
        connection_layout.addWidget(self.refresh_button)
        connection_layout.addWidget(self.connect_button)
        connection_layout.addWidget(self.status_label)
        connection_layout.addStretch()
        
        connection_group.setLayout(connection_layout)
        main_layout.addWidget(connection_group)
        
        # Middle section: Real-time data visualization
        graph_group = QtWidgets.QGroupBox("Real-time Data")
        graph_layout = QtWidgets.QVBoxLayout()
        
        self.graph = DataGraph(width=9, height=4)
        graph_layout.addWidget(self.graph)
        
        graph_group.setLayout(graph_layout)
        main_layout.addWidget(graph_group)
        
        # Bottom section: User registration and recording controls
        control_group = QtWidgets.QGroupBox("Recording Controls")
        control_layout = QtWidgets.QGridLayout()
        
        # User management
        control_layout.addWidget(QtWidgets.QLabel("Username:"), 0, 0)
        self.username_input = QtWidgets.QLineEdit()
        control_layout.addWidget(self.username_input, 0, 1)
        
        self.register_button = QtWidgets.QPushButton("Register")
        control_layout.addWidget(self.register_button, 0, 2)
        
        control_layout.addWidget(QtWidgets.QLabel("User ID:"), 0, 3)
        self.user_id_label = QtWidgets.QLabel("Not registered")
        control_layout.addWidget(self.user_id_label, 0, 4)
        
        # Recording settings
        control_layout.addWidget(QtWidgets.QLabel("Alphabet:"), 1, 0)
        self.alphabet_combo = QtWidgets.QComboBox()
        self.alphabet_combo.addItems(BENGALI_ALPHABETS)
        control_layout.addWidget(self.alphabet_combo, 1, 1)
        
        control_layout.addWidget(QtWidgets.QLabel("Hand:"), 1, 2)
        self.hand_combo = QtWidgets.QComboBox()
        self.hand_combo.addItems(["Right Hand", "Left Hand"])
        control_layout.addWidget(self.hand_combo, 1, 3)
        
        # Recording controls
        self.record_button = QtWidgets.QPushButton("Start Recording")
        self.record_button.setEnabled(False)
        control_layout.addWidget(self.record_button, 2, 0, 1, 2)
        
        self.recording_status = QtWidgets.QLabel("Not recording")
        control_layout.addWidget(self.recording_status, 2, 2, 1, 3)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # Connect signals
        self.refresh_button.clicked.connect(self.refresh_ports)
        self.connect_button.clicked.connect(self.toggle_connection)
        self.register_button.clicked.connect(self.register_user)
        self.record_button.clicked.connect(self.toggle_recording)
    
    def refresh_ports(self):
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_combo.addItem(port.device)
    
    def toggle_connection(self):
        if self.serial_thread.serial_port is None:
            port = self.port_combo.currentText()
            if port:
                if self.serial_thread.connect_to_device(port):
                    self.serial_thread.start()
                    self.connect_button.setText("Disconnect")
                    self.record_button.setEnabled(True)
        else:
            self.serial_thread.disconnect_device()
            self.connect_button.setText("Connect")
            self.record_button.setEnabled(False)
    
    def update_connection_status(self, connected, message):
        self.status_label.setText(f"Status: {message}")
        if not connected and self.connect_button.text() == "Disconnect":
            self.connect_button.setText("Connect")
            self.record_button.setEnabled(False)
    
    def register_user(self):
        username = self.username_input.text().strip()
        if username:
            user_id = self.user_manager.register_user(username)
            self.user_id_label.setText(user_id)
            self.record_button.setEnabled(self.serial_thread.serial_port is not None)
        else:
            QtWidgets.QMessageBox.warning(self, "Registration Error", "Please enter a username")
    
    def toggle_recording(self):
        if not self.data_recorder.recording:
            # Start recording
            user_id = self.user_id_label.text()
            if user_id == "Not registered":
                QtWidgets.QMessageBox.warning(self, "Recording Error", "Please register a user first")
                return
            
            alphabet = self.alphabet_combo.currentText()
            hand = "Right" if self.hand_combo.currentText() == "Right Hand" else "Left"
            
            self.data_recorder.start_recording(user_id, alphabet, hand)
            self.record_button.setText("Stop Recording")
            self.recording_status.setText(f"Recording: {alphabet} - {hand} Hand")
        else:
            # Stop recording
            if self.data_recorder.stop_recording():
                reply = QtWidgets.QMessageBox.question(
                    self, "Save Recording", 
                    "Do you want to save this recording?",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                )
                
                if reply == QtWidgets.QMessageBox.Yes:
                    if self.data_recorder.save_data():
                        QtWidgets.QMessageBox.information(
                            self, "Success", 
                            "Data saved and synchronized with GitHub"
                        )
                    else:
                        QtWidgets.QMessageBox.warning(
                            self, "Save Error", 
                            "Failed to save data"
                        )
            
            self.record_button.setText("Start Recording")
            self.recording_status.setText("Not recording")
    
    def update_data(self, data):
        self.graph.update_data(data)
        
        if self.data_recorder.recording:
            self.data_recorder.add_data(data)
    
    def update_ui(self):
        # Periodic UI updates if needed
        pass
    
    def closeEvent(self, event):
        # Clean up before closing
        if self.serial_thread.isRunning():
            self.serial_thread.disconnect_device()
            self.serial_thread.wait()
        event.accept()

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style
    
    # Set application font
    font = QtGui.QFont("Arial", 10)
    app.setFont(font)
    
    # Set application stylesheet for modern look
    app.setStyleSheet("""
        QGroupBox {
            border: 1px solid #CCCCCC;
            border-radius: 5px;
            margin-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #0078D7;
            color: white;
            border: none;
            padding: 5px 15px;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #005A9E;
        }
        QPushButton:disabled {
            background-color: #CCCCCC;
            color: #888888;
        }
        QComboBox {
            border: 1px solid #CCCCCC;
            border-radius: 3px;
            padding: 2px 5px;
        }
        QLineEdit {
            border: 1px solid #CCCCCC;
            border-radius: 3px;
            padding: 2px 5px;
        }
    """)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())