import os
import sys
import json
import csv
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QFileDialog, QMessageBox, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QIcon, QKeyEvent
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from collections import deque

class DataVisualizer(FigureCanvas):
    """Canvas for plotting accelerometer and gyroscope data"""
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor('#F5F5F5')
        
        # Create subplots
        self.accel_ax = self.fig.add_subplot(211)
        self.gyro_ax = self.fig.add_subplot(212)
        
        # Set titles and labels
        self.accel_ax.set_title('Accelerometer Data', fontsize=10)
        self.gyro_ax.set_title('Gyroscope Data', fontsize=10)
        self.accel_ax.set_ylabel('Value (G)', fontsize=8)
        self.gyro_ax.set_ylabel('Value (°/s)', fontsize=8)
        self.gyro_ax.set_xlabel('Time (samples)', fontsize=8)
        
        # Initialize empty plots
        self.accel_lines = []
        self.gyro_lines = []
        colors = ["#2C3E50", "#E74C3C", "#3498DB"]  # X, Y, Z colors
        
        for i in range(3):
            acc_line, = self.accel_ax.plot([], [], lw=1.5, color=colors[i], 
                                         label=['X', 'Y', 'Z'][i])
            gyro_line, = self.gyro_ax.plot([], [], lw=1.5, color=colors[i],
                                         label=['X', 'Y', 'Z'][i])
            self.accel_lines.append(acc_line)
            self.gyro_lines.append(gyro_line)
            
        self.accel_ax.legend(loc='upper right', fontsize='small')
        self.gyro_ax.legend(loc='upper right', fontsize='small')
        self.fig.tight_layout()
        
        super().__init__(self.fig)
        self.setParent(parent)
    
    def plot_data(self, data):
        """Plot the loaded data"""
        timestamps = [point['timestamp'] for point in data]
        
        # Extract accelerometer data
        acc_x = [point['acc_x'] for point in data]
        acc_y = [point['acc_y'] for point in data]
        acc_z = [point['acc_z'] for point in data]
        
        # Extract gyroscope data
        gyro_x = [point['gyro_x'] for point in data]
        gyro_y = [point['gyro_y'] for point in data]
        gyro_z = [point['gyro_z'] for point in data]
        
        # Update accelerometer plot
        self.accel_lines[0].set_data(timestamps, acc_x)
        self.accel_lines[1].set_data(timestamps, acc_y)
        self.accel_lines[2].set_data(timestamps, acc_z)
        
        # Update gyroscope plot
        self.gyro_lines[0].set_data(timestamps, gyro_x)
        self.gyro_lines[1].set_data(timestamps, gyro_y)
        self.gyro_lines[2].set_data(timestamps, gyro_z)
        
        # Adjust axes
        self.accel_ax.relim()
        self.accel_ax.autoscale_view()
        self.gyro_ax.relim()
        self.gyro_ax.autoscale_view()
        
        self.draw()

class DataExplorer(QMainWindow):
    """Main application window for exploring recorded data"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Air Writing Data Explorer")
        self.setMinimumSize(1000, 700)
        
        # Data variables
        self.current_folder = ""
        self.data_files = []
        self.current_index = -1
        self.current_data = []
        self.metadata = {}
        
        # UI Setup
        self.setup_ui()
        self.setup_style()
        
    def setup_ui(self):
        """Set up the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        
        # File navigation controls
        nav_group = QGroupBox("File Navigation")
        nav_layout = QHBoxLayout()
        
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        
        self.open_button = QPushButton("Open Folder")
        self.open_button.clicked.connect(self.open_folder)
        
        self.prev_button = QPushButton("← Previous (Left Arrow)")
        self.prev_button.clicked.connect(self.prev_file)
        
        self.next_button = QPushButton("Next (Right Arrow) →")
        self.next_button.clicked.connect(self.next_file)
        
        self.delete_button = QPushButton("Delete Current File")
        self.delete_button.clicked.connect(self.delete_current_file)
        self.delete_button.setStyleSheet("background-color: #E74C3C; color: white;")
        
        nav_layout.addWidget(self.open_button)
        nav_layout.addWidget(self.folder_label, 1)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        nav_layout.addWidget(self.delete_button)
        
        nav_group.setLayout(nav_layout)
        main_layout.addWidget(nav_group)
        
        # Data visualization
        vis_group = QGroupBox("Data Visualization")
        vis_layout = QVBoxLayout()
        
        self.visualizer = DataVisualizer(self, width=10, height=6)
        vis_layout.addWidget(self.visualizer)
        
        vis_group.setLayout(vis_layout)
        main_layout.addWidget(vis_group, 1)
        
        # Metadata display
        meta_group = QGroupBox("Recording Information")
        meta_layout = QGridLayout()
        
        self.meta_labels = {
            'letter': QLabel("Letter: -"),
            'user_id': QLabel("User ID: -"),
            'hand': QLabel("Hand: -"),
            'date': QLabel("Date: -"),
            'duration': QLabel("Duration: -"),
            'samples': QLabel("Samples: -")
        }
        
        row = 0
        col = 0
        for label in self.meta_labels.values():
            label.setStyleSheet("font-weight: bold;")
            meta_layout.addWidget(label, row, col)
            col += 1
            if col > 2:
                col = 0
                row += 1
        
        meta_group.setLayout(meta_layout)
        main_layout.addWidget(meta_group)
        
    def setup_style(self):
        """Set up the application style"""
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
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
            }
            QLabel {
                color: #2C3E50;
                padding: 5px;
            }
        """)
    
    def keyPressEvent(self, event):
        """Handle keyboard navigation"""
        if event.key() == Qt.Key_Left:
            self.prev_file()
        elif event.key() == Qt.Key_Right:
            self.next_file()
        else:
            super().keyPressEvent(event)
    
    def open_folder(self):
        """Open a folder dialog and load data files"""
        folder = QFileDialog.getExistingDirectory(self, "Select Data Folder")
        if folder:
            self.current_folder = folder
            self.folder_label.setText(f"Folder: {folder}")
            
            # Find all CSV files in the folder and subfolders
            self.data_files = []
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith('.csv'):
                        self.data_files.append(os.path.join(root, file))
            
            if self.data_files:
                self.current_index = 0
                self.load_current_file()
            else:
                self.current_index = -1
                self.clear_display()
                QMessageBox.information(self, "No Files", "No CSV files found in the selected folder")
    
    def load_current_file(self):
        """Load the current file's data"""
        if 0 <= self.current_index < len(self.data_files):
            file_path = self.data_files[self.current_index]
            
            try:
                # Read CSV file
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    self.current_data = [row for row in reader]
                
                # Extract metadata from filename
                filename = os.path.basename(file_path)
                parts = filename.split('_')
                
                self.metadata = {
                    'letter': parts[0] if len(parts) > 0 else "Unknown",
                    'user_id': parts[1] if len(parts) > 1 else "Unknown",
                    'hand': parts[2] if len(parts) > 2 else "Unknown",
                    'date': parts[3].split('.')[0] if len(parts) > 3 else "Unknown",
                    'duration': float(self.current_data[-1]['timestamp']) if self.current_data else 0,
                    'samples': len(self.current_data)
                }
                
                # Update UI
                self.update_display()
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not load file: {str(e)}")
                self.clear_display()
    
    def update_display(self):
        """Update all UI elements with current data"""
        # Update metadata labels
        self.meta_labels['letter'].setText(f"Letter: {self.metadata['letter']}")
        self.meta_labels['user_id'].setText(f"User ID: {self.metadata['user_id']}")
        self.meta_labels['hand'].setText(f"Hand: {self.metadata['hand']}")
        self.meta_labels['date'].setText(f"Date: {self.metadata['date']}")
        self.meta_labels['duration'].setText(f"Duration: {self.metadata['duration']:.2f}s")
        self.meta_labels['samples'].setText(f"Samples: {self.metadata['samples']}")
        
        # Convert data to numeric format for plotting
        numeric_data = []
        for point in self.current_data:
            numeric_data.append({
                'timestamp': float(point['timestamp']),
                'acc_x': float(point['acc_x']),
                'acc_y': float(point['acc_y']),
                'acc_z': float(point['acc_z']),
                'gyro_x': float(point['gyro_x']),
                'gyro_y': float(point['gyro_y']),
                'gyro_z': float(point['gyro_z'])
            })
        
        # Update visualization
        self.visualizer.plot_data(numeric_data)
        
        # Update navigation buttons
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.data_files) - 1)
        self.delete_button.setEnabled(True)
    
    def clear_display(self):
        """Clear all displayed data"""
        for label in self.meta_labels.values():
            label.setText(label.text().split(':')[0] + ": -")
        
        # Clear plots
        for line in self.visualizer.accel_lines + self.visualizer.gyro_lines:
            line.set_data([], [])
        self.visualizer.draw()
        
        self.delete_button.setEnabled(False)
    
    def prev_file(self):
        """Navigate to previous file"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_file()
    
    def next_file(self):
        """Navigate to next file"""
        if self.current_index < len(self.data_files) - 1:
            self.current_index += 1
            self.load_current_file()
    
    def delete_current_file(self):
        """Delete the currently displayed file"""
        if 0 <= self.current_index < len(self.data_files):
            file_path = self.data_files[self.current_index]
            
            reply = QMessageBox.question(
                self, 'Confirm Delete',
                f"Are you sure you want to delete:\n{file_path}?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    os.remove(file_path)
                    
                    # Remove from file list
                    del self.data_files[self.current_index]
                    
                    # Adjust current index
                    if self.current_index >= len(self.data_files):
                        self.current_index = len(self.data_files) - 1
                    
                    if self.data_files:
                        self.load_current_file()
                    else:
                        self.current_index = -1
                        self.clear_display()
                        QMessageBox.information(self, "Info", "No more files in folder")
                    
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Could not delete file: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application font
    font = QFont("Arial", 10)
    app.setFont(font)
    
    window = DataExplorer()
    window.show()
    sys.exit(app.exec_())