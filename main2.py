import sys
import json
import os
import numpy as np
import serial
import serial.tools.list_ports
import time
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QComboBox, QStatusBar, QMessageBox, 
                            QSplitter, QGroupBox, QGridLayout, QFrame, QLineEdit)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QColor, QPalette
import csv
import uuid
import git

# Bengali Alphabet List
BENGALI_ALPHABETS = ["অ", "আ", "ই", "ঈ", "উ", "ঊ", "ঋ", "এ", "ঐ", "ও", "ঔ", "ক", "খ", "গ", "ঘ", "ঙ", 
                    "চ", "ছ", "জ", "ঝ", "ঞ", "ট", "ঠ", "ড", "ঢ", "ণ", "ত", "থ", "দ", "ধ", "ন", "প", 
                    "ফ", "ব", "ভ", "ম", "য", "র", "ল", "শ", "ষ", "স", "হ", "ড়", "ঢ়", "য়"]

# Constants
SAMPLE_WINDOW = 100  # Number of data points to display in real-time
DATA_LABELS = ["Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z"]
COLORS = ["#2C3E50", "#E74C3C", "#3498DB", "#9B59B6", "#16A085", "#F39C12"]
DATA_DIR = "recordings"
GITHUB_REPO_PATH = "git@github.com:zubayertahmid/air-writing-data.git"  # Update with your repo
HAND_PREFERENCES = ["Right Hand", "Left Hand"]

class SerialReaderThread(QThread):
    """Thread for reading data from serial port without blocking UI"""
    data_received = pyqtSignal(list)
    connection_error = pyqtSignal(str)
    connection_status = pyqtSignal(bool)
    
    def __init__(self, port=None, baud_rate=115200):
        super().__init__()
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn = None
        self.running = False
        
    def connect_to_device(self, port):
        """Connect to the specified serial port"""
        try:
            # Close previous connection if exists
            if self.serial_conn is not None:
                if self.serial_conn.is_open:
                    self.serial_conn.close()
                self.serial_conn = None
                
            # Create new connection
            self.port = port
            self.serial_conn = serial.Serial(
                port=port,
                baudrate=self.baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
            
            # Flush any leftover data
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            
            # Wait for the serial connection to stabilize
            time.sleep(2.0)
            
            # Signal successful connection
            self.connection_status.emit(True)
            return True
        except Exception as e:
            self.connection_error.emit(f"Connection error: {str(e)}")
            self.serial_conn = None
            self.connection_status.emit(False)
            return False
            
    def run(self):
        """Main thread execution loop"""
        self.running = True
        debug_counter = 0  # For debugging
        
        while self.running:
            # Check if connection exists
            if self.serial_conn is None:
                time.sleep(0.1)
                continue
                
            try:
                # Check if connection is open
                if not self.serial_conn.is_open:
                    time.sleep(0.1)
                    continue
                    
                # Read data if available
                if self.serial_conn.in_waiting > 0:
                    try:
                        # Read one complete line
                        line = self.serial_conn.readline().decode('utf-8').strip()
                        
                        # Debug output every 50 lines
                        debug_counter += 1
                        if debug_counter % 50 == 0:
                            print(f"Raw data received: {line}")
                        
                        # Skip empty lines
                        if not line:
                            continue
                            
                        # Parse data
                        parts = [part.strip() for part in line.split(',')]
                        if len(parts) == 7:
                            try:
                                data = [float(val) for val in parts]
                                self.data_received.emit(data)
                            except ValueError as ve:
                                print(f"Value error parsing data: {ve} - Raw data: {line}")
                                continue
                        else:
                            print(f"Expected 7 values but got {len(parts)} - Raw data: {line}")
                    except UnicodeDecodeError:
                        # Skip invalid UTF-8 data
                        continue
                else:
                    # No data waiting, yield to other threads
                    self.msleep(10)
            except Exception as e:
                # Handle connection errors
                error_message = str(e)
                print(f"Serial error: {error_message}")
                self.connection_error.emit(f"Read error: {error_message}")
                
                # Reset connection on error
                try:
                    if self.serial_conn is not None:
                        self.serial_conn.close()
                except:
                    pass
                self.serial_conn = None
                self.connection_status.emit(False)
                
                # Wait before retry
                time.sleep(1)
    
    def stop(self):
        """Stop the thread and close connection"""
        self.running = False
        try:
            if self.serial_conn is not None:
                if self.serial_conn.is_open:
                    self.serial_conn.close()
                self.serial_conn = None
        except:
            pass
        self.wait()


class AccelGyroPlot(FigureCanvas):
    """Canvas for plotting accelerometer and gyroscope data"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Create figure and subplots
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor('#F5F5F5')
        
        # Create 2 subplots - one for accelerometer, one for gyroscope
        self.accel_ax = self.fig.add_subplot(211)
        self.gyro_ax = self.fig.add_subplot(212)
        
        # Set titles and labels
        self.accel_ax.set_title('Accelerometer Data', fontsize=10)
        self.gyro_ax.set_title('Gyroscope Data', fontsize=10)
        self.accel_ax.set_ylabel('Value (G)', fontsize=8)
        self.gyro_ax.set_ylabel('Value (°/s)', fontsize=8)
        self.gyro_ax.set_xlabel('Time (samples)', fontsize=8)
        
        # Adjust spacing
        self.fig.tight_layout(pad=2.0)
        
        # Initialize canvas
        super(AccelGyroPlot, self).__init__(self.fig)
        self.setParent(parent)
        
        # Data buffers
        self.times = deque(maxlen=SAMPLE_WINDOW)
        self.accel_data = [deque(maxlen=SAMPLE_WINDOW) for _ in range(3)]
        self.gyro_data = [deque(maxlen=SAMPLE_WINDOW) for _ in range(3)]
        
        # Initialize line objects
        self.accel_lines = []
        self.gyro_lines = []
        
        # Create the lines for accelerometer
        for i in range(3):
            line, = self.accel_ax.plot([], [], lw=1.5, color=COLORS[i], label=DATA_LABELS[i])
            self.accel_lines.append(line)
            
        # Create the lines for gyroscope
        for i in range(3):
            line, = self.gyro_ax.plot([], [], lw=1.5, color=COLORS[i+3], label=DATA_LABELS[i+3])
            self.gyro_lines.append(line)
            
        # Add legends
        self.accel_ax.legend(loc='upper right', fontsize='small')
        self.gyro_ax.legend(loc='upper right', fontsize='small')
        
        # Start with some initial data to avoid issues
        for i in range(SAMPLE_WINDOW):
            self.times.append(i)
            for j in range(3):
                self.accel_data[j].append(0)
                self.gyro_data[j].append(0)
                
        # Set y-axis limits
        self.accel_ax.set_ylim(-3, 3)
        self.gyro_ax.set_ylim(-3, 3)
        
        # Initial draw
        self.update_plot()
    
    def update_data(self, data):
        """Add new data to the plot buffers"""
        # Add new time point
        if len(self.times) > 0:
            self.times.append(self.times[-1] + 1)
        else:
            self.times.append(0)
            
        # Add accelerometer data (first 3 values)
        for i in range(3):
            self.accel_data[i].append(data[i])
            
        # Add gyroscope data (next 3 values)
        for i in range(3):
            self.gyro_data[i].append(data[i+3])
            
    def update_plot(self):
        """Update the plot with current data"""
        time_array = list(self.times)
        
        # Update x-axis limits to show the most recent data
        self.accel_ax.set_xlim(max(0, time_array[-1] - SAMPLE_WINDOW), max(SAMPLE_WINDOW, time_array[-1]))
        self.gyro_ax.set_xlim(max(0, time_array[-1] - SAMPLE_WINDOW), max(SAMPLE_WINDOW, time_array[-1]))
        
        # Update accelerometer lines
        for i, line in enumerate(self.accel_lines):
            line.set_data(time_array, list(self.accel_data[i]))
            
        # Update gyroscope lines
        for i, line in enumerate(self.gyro_lines):
            line.set_data(time_array, list(self.gyro_data[i]))
            
        # Redraw
        self.fig.canvas.draw_idle()


class DataRecorder:
    """Handles recording and saving sensor data with GitHub sync"""
    def __init__(self):
        self.recording = False
        self.current_alphabet = None
        self.recorded_data = []
        self.user_id = None
        self.hand_preference = None
        self.start_time = None
        
    def start_recording(self, alphabet, user_id, hand_preference):
        """Start recording data for the specified alphabet"""
        self.recording = True
        self.current_alphabet = alphabet
        self.user_id = user_id
        self.hand_preference = hand_preference
        self.recorded_data = []
        self.start_time = time.time()
        return True
        
    def add_data_point(self, data):
        """Add a new data point to the current recording session"""
        if not self.recording:
            return False
            
        # Store timestamp and sensor values
        self.recorded_data.append({
            "timestamp": time.time() - self.start_time,
            "user_id": self.user_id,
            "hand_preference": self.hand_preference,
            "accel_x": data[0],
            "accel_y": data[1],
            "accel_z": data[2],
            "gyro_x": data[3], 
            "gyro_y": data[4],
            "gyro_z": data[5],
            # Note: ignoring data[6] (x7) as per requirements
        })
        return True
        
    def stop_recording(self):
        """Stop the current recording session"""
        self.recording = False
        return len(self.recorded_data) > 0
        
    def save_recording(self):
        """Save the recorded data to a file and sync with GitHub"""
        if not self.current_alphabet or not self.recorded_data:
            return False, "No data to save"
            
        # Create directory for the alphabet if it doesn't exist
        alphabet_dir = os.path.join(DATA_DIR, self.current_alphabet)
        os.makedirs(alphabet_dir, exist_ok=True)
        
        # Create a unique filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(alphabet_dir, f"{self.current_alphabet}_{self.user_id}_{self.hand_preference}_{timestamp}.csv")
        
        # Save the data to CSV file
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['timestamp', 'user_id', 'hand_preference', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
                
                # Write data rows
                for row in self.recorded_data:
                    writer.writerow([
                        row['timestamp'],
                        row['user_id'],
                        row['hand_preference'],
                        row['accel_x'],
                        row['accel_y'],
                        row['accel_z'],
                        row['gyro_x'],
                        row['gyro_y'],
                        row['gyro_z']
                    ])
            
            # Sync with GitHub
            try:
                repo = git.Repo(GITHUB_REPO_PATH)
                relative_path = os.path.relpath(filename, GITHUB_REPO_PATH)
                repo.git.add(relative_path)
                commit_message = f"Added recording: {self.current_alphabet} by user {self.user_id} ({self.hand_preference})"
                repo.git.commit('-m', commit_message)
                origin = repo.remote(name='origin')
                origin.push()
                
                return True, filename
            except Exception as e:
                return False, f"GitHub sync error: {str(e)}"
                
        except Exception as e:
            return False, str(e)


class UserManager:
    """Handles user registration and management"""
    def __init__(self):
        self.users_file = os.path.join(DATA_DIR, "users.csv")
        self.users = {}  # {username: user_id}
        self.load_users()
        
    def load_users(self):
        """Load existing users from file"""
        os.makedirs(DATA_DIR, exist_ok=True)
        
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader, None)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        user_id, username = row[0], row[1]
                        self.users[username] = user_id
    
    def register_user(self, username):
        """Register a new user or return existing user ID"""
        if username in self.users:
            return self.users[username]
        
        # Generate a new user ID
        user_id = str(uuid.uuid4())[:8]
        self.users[username] = user_id
        
        # Save to file
        with open(self.users_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if os.path.getsize(self.users_file) == 0:
                writer.writerow(['user_id', 'username'])
            writer.writerow([user_id, username])
        
        return user_id
    
    def get_user_id(self, username):
        """Get user ID for a registered username"""
        return self.users.get(username, None)


class AirWritingSystem(QMainWindow):
    """Main application window for the Air Writing System"""
    def __init__(self):
        super().__init__()
        
        # Set up the main window
        self.setWindowTitle("Bengali Air Writing System")
        self.setMinimumSize(900, 600)
        
        # Initialize serial reader thread
        self.serial_reader = SerialReaderThread()
        self.serial_reader.data_received.connect(self.process_serial_data)
        self.serial_reader.connection_error.connect(self.show_error_message)
        
        # Initialize data recorder and user manager
        self.data_recorder = DataRecorder()
        self.user_manager = UserManager()
        
        # Set up the UI
        self.setup_ui()
        
        # Start the UI update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(50)  # 20 Hz refresh rate
        
        # Start the serial reader thread
        self.serial_reader.start()
        
        # Initialize available ports
        self.refresh_serial_ports()

    def setup_ui(self):
        """Set up the user interface"""
        # Apply modern style
        self.setup_style()
        
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create top section with connection controls
        connection_group = self.create_connection_group()
        
        # Create middle section with visualization
        visualization_group = self.create_visualization_group()
        
        # Create bottom section with recording controls
        recording_group = self.create_recording_group()
        
        # Add sections to main layout
        main_layout.addWidget(connection_group)
        main_layout.addWidget(visualization_group, 1)  # 1 = stretch factor
        main_layout.addWidget(recording_group)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
    def setup_style(self):
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F5;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
            }
            QPushButton#startButton {
                background-color: #2ECC71;
            }
            QPushButton#startButton:hover {
                background-color: #27AE60;
            }
            QPushButton#stopButton {
                background-color: #E74C3C;
            }
            QPushButton#stopButton:hover {
                background-color: #C0392B;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
            }
            QLabel {
                color: #2C3E50;
            }
            QStatusBar {
                background-color: #ECF0F1;
                color: #2C3E50;
            }
        """)
    
    def create_connection_group(self):
        group_box = QGroupBox("Device Connection")
        layout = QHBoxLayout()
        
        # Port selection
        port_label = QLabel("COM Port:")
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(150)
        
        # Connect/Disconnect button
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.toggle_connection)
        
        # Refresh ports button
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_serial_ports)
        
        # Connection status
        self.connection_status = QLabel("Disconnected")
        self.connection_status.setStyleSheet("color: #E74C3C; font-weight: bold;")
        
        # Add widgets to layout
        layout.addWidget(port_label)
        layout.addWidget(self.port_combo)
        layout.addWidget(self.connect_button)
        layout.addWidget(refresh_button)
        layout.addStretch()
        layout.addWidget(QLabel("Status:"))
        layout.addWidget(self.connection_status)
        
        group_box.setLayout(layout)
        return group_box
    
    def create_visualization_group(self):
        """Create the visualization group"""
        group_box = QGroupBox("Real-time Data Visualization")
        layout = QVBoxLayout()
        
        # Create the plot canvas
        self.plot_canvas = AccelGyroPlot(self, width=8, height=6)
        
        # Add canvas to layout
        layout.addWidget(self.plot_canvas)
        
        group_box.setLayout(layout)
        return group_box
    
    def create_recording_group(self):
        """Create the recording controls group"""
        group_box = QGroupBox("Recording Controls")
        layout = QGridLayout()
        
        # User registration
        username_label = QLabel("Username:")
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter your name")
        
        register_button = QPushButton("Register")
        register_button.clicked.connect(self.register_user)
        
        user_id_label = QLabel("User ID:")
        self.user_id_display = QLabel("Not registered")
        
        # Alphabet selection
        alphabet_label = QLabel("Bengali Alphabet:")
        self.alphabet_combo = QComboBox()
        self.alphabet_combo.addItems(BENGALI_ALPHABETS)
        self.alphabet_combo.setMinimumWidth(80)
        
        # Hand preference
        hand_label = QLabel("Hand Preference:")
        self.hand_combo = QComboBox()
        self.hand_combo.addItems(HAND_PREFERENCES)
        
        # Recording buttons
        self.start_button = QPushButton("Start Recording")
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self.start_recording)
        
        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        
        # Recording status
        self.recording_status = QLabel("Not Recording")
        self.recording_status.setStyleSheet("font-weight: bold;")
        
        # Data count
        self.data_count_label = QLabel("Data points: 0")
        
        # Add widgets to layout
        layout.addWidget(username_label, 0, 0)
        layout.addWidget(self.username_input, 0, 1)
        layout.addWidget(register_button, 0, 2)
        layout.addWidget(user_id_label, 0, 3)
        layout.addWidget(self.user_id_display, 0, 4)
        
        layout.addWidget(alphabet_label, 1, 0)
        layout.addWidget(self.alphabet_combo, 1, 1)
        layout.addWidget(hand_label, 1, 2)
        layout.addWidget(self.hand_combo, 1, 3)
        
        layout.addWidget(self.start_button, 2, 0, 1, 2)
        layout.addWidget(self.stop_button, 2, 2, 1, 2)
        
        layout.addWidget(QLabel("Status:"), 3, 0)
        layout.addWidget(self.recording_status, 3, 1)
        layout.addWidget(self.data_count_label, 3, 2, 1, 2)
        
        group_box.setLayout(layout)
        return group_box
    
    def refresh_serial_ports(self):
        """Refresh the list of available serial ports"""
        self.port_combo.clear()
        
        # Get all available ports
        ports = serial.tools.list_ports.comports()
        
        # Add port names to combo box
        for port in ports:
            self.port_combo.addItem(port.device)
            
        # Disable connect button if no ports available
        self.connect_button.setEnabled(self.port_combo.count() > 0)
        
        if self.port_combo.count() == 0:
            self.status_bar.showMessage("No serial ports found. Connect device and refresh.")
    
    def toggle_connection(self):
        """Connect to or disconnect from the selected serial port"""
        if self.serial_reader.serial_conn is not None and self.serial_reader.serial_conn.is_open:
            # Disconnect
            self.serial_reader.stop()
            self.serial_reader = SerialReaderThread()  # Create new thread instance
            self.serial_reader.data_received.connect(self.process_serial_data)
            self.serial_reader.connection_error.connect(self.show_error_message)
            self.serial_reader.start()
            
            self.connect_button.setText("Connect")
            self.connection_status.setText("Disconnected")
            self.connection_status.setStyleSheet("color: #E74C3C; font-weight: bold;")
            self.status_bar.showMessage("Device disconnected")
            self.start_button.setEnabled(False)
        else:
            # Connect
            port = self.port_combo.currentText()
            if port:
                if self.serial_reader.connect_to_device(port):
                    self.connect_button.setText("Disconnect")
                    self.connection_status.setText("Connected")
                    self.connection_status.setStyleSheet("color: #2ECC71; font-weight: bold;")
                    self.status_bar.showMessage(f"Connected to {port}")
                    if self.user_id_display.text() != "Not registered":
                        self.start_button.setEnabled(True)
                else:
                    self.status_bar.showMessage(f"Failed to connect to {port}")
            else:
                self.status_bar.showMessage("No port selected")
    
    def register_user(self):
        """Register a new user or get existing user ID"""
        username = self.username_input.text().strip()
        if not username:
            self.show_error_message("Please enter a username")
            return
            
        user_id = self.user_manager.register_user(username)
        self.user_id_display.setText(user_id)
        
        # Enable start button if device is connected
        if (self.serial_reader.serial_conn is not None and 
            self.serial_reader.serial_conn.is_open):
            self.start_button.setEnabled(True)
            
        self.status_bar.showMessage(f"User registered: {username} (ID: {user_id})")
    
    def process_serial_data(self, data):
        """Process data received from the serial port"""
        # Update the plot with new data
        self.plot_canvas.update_data(data)
        
        # If recording, add data to recording
        if self.data_recorder.recording:
            self.data_recorder.add_data_point(data)
            
            # Update data count label
            self.data_count_label.setText(f"Data points: {len(self.data_recorder.recorded_data)}")
    
    def update_ui(self):
        """Update the UI elements"""
        # Update the plot
        self.plot_canvas.update_plot()
        
        # Check if serial connection is still active
        if self.serial_reader.serial_conn and not self.serial_reader.serial_conn.is_open:
            self.connection_status.setText("Disconnected")
            self.connection_status.setStyleSheet("color: #E74C3C; font-weight: bold;")
            self.connect_button.setText("Connect")
            self.start_button.setEnabled(False)
    
    def start_recording(self):
        """Start recording data for the selected alphabet"""
        if not self.serial_reader.serial_conn or not self.serial_reader.serial_conn.is_open:
            self.show_error_message("Device not connected. Please connect first.")
            return
            
        if self.user_id_display.text() == "Not registered":
            self.show_error_message("Please register a user first.")
            return
            
        alphabet = self.alphabet_combo.currentText()
        if not alphabet:
            self.show_error_message("Please select an alphabet before recording.")
            return
            
        user_id = self.user_id_display.text()
        hand_preference = self.hand_combo.currentText()
        
        # Start recording
        if self.data_recorder.start_recording(alphabet, user_id, hand_preference):
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.alphabet_combo.setEnabled(False)
            self.hand_combo.setEnabled(False)
            self.username_input.setEnabled(False)
            self.recording_status.setText(f"Recording {alphabet} ({hand_preference})...")
            self.recording_status.setStyleSheet("color: #E74C3C; font-weight: bold;")
            self.data_count_label.setText("Data points: 0")
            self.status_bar.showMessage(f"Recording data for alphabet: {alphabet}")
    
    def stop_recording(self):
        """Stop the current recording session"""
        if not self.data_recorder.recording:
            return
            
        # Stop recording
        if self.data_recorder.stop_recording():
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.alphabet_combo.setEnabled(True)
            self.hand_combo.setEnabled(True)
            self.username_input.setEnabled(True)
            self.recording_status.setText("Not Recording")
            self.recording_status.setStyleSheet("font-weight: bold;")
            
            # Ask user if they want to save the data
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setWindowTitle("Save Recording")
            msg_box.setText(f"Save recording for alphabet '{self.data_recorder.current_alphabet}'?")
            msg_box.setInformativeText(f"Contains {len(self.data_recorder.recorded_data)} data points.")
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg_box.setDefaultButton(QMessageBox.Yes)
            
            if msg_box.exec_() == QMessageBox.Yes:
                self.save_recording()
            else:
                self.status_bar.showMessage("Recording discarded")
    
    def save_recording(self):
        """Save the current recording session"""
        success, result = self.data_recorder.save_recording()
        
        if success:
            self.status_bar.showMessage(f"Recording saved and synced: {result}")
            QMessageBox.information(self, "Success", "Data saved and synchronized with GitHub")
        else:
            self.show_error_message(f"Failed to save recording: {result}")
    
    def show_error_message(self, message):
        """Show an error message to the user"""
        self.status_bar.showMessage(message)
        
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(message)
        msg_box.exec_()
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop the serial reader thread
        if self.serial_reader.isRunning():
            self.serial_reader.stop()
            
        # Accept the close event
        event.accept()


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Create and run the application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Use Fusion style for better cross-platform appearance
    
    window = AirWritingSystem()
    window.show()
    
    sys.exit(app.exec_())