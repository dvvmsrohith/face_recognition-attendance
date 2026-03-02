import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from PIL import Image, ImageTk
import pickle
import hashlib
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class ModernFaceRecognitionSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Face Recognition Attendance System")
        self.root.geometry("1200x700")
        self.root.configure(bg='#2c3e50')
        
        # Apply modern style
        self.setup_styles()
        
        # Initialize variables
        self.running = False
        self.cap = None
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.attendance_data = []
        self.logged_today = set()
        self.face_data_path = "face_data"
        self.students_data = self.load_students()
        self.current_frame = None
        self.recognition_thread = None
        
        # Create directories
        os.makedirs(self.face_data_path, exist_ok=True)
        
        # Setup UI
        self.setup_ui()
        
        # Load existing data
        self.load_attendance_data()
        self.load_trained_model()
        
    def setup_styles(self):
        """Configure modern styles for the application"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        self.colors = {
            'bg': '#2c3e50',
            'fg': '#ecf0f1',
            'accent': '#3498db',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'surface': '#34495e'
        }
        
    def setup_ui(self):
        """Create modern user interface"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Video feed
        left_frame = ttk.Frame(main_frame, width=600, height=500)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        left_frame.pack_propagate(False)
        
        # Video label
        self.video_label = ttk.Label(left_frame, text="Camera Feed", anchor='center')
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Right panel - Controls
        right_frame = ttk.Frame(main_frame, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(5, 0))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Registration
        self.create_registration_tab(notebook)
        
        # Tab 2: Attendance
        self.create_attendance_tab(notebook)
        
        # Tab 3: Reports
        self.create_reports_tab(notebook)
        
        # Status bar
        self.create_status_bar()
        
    def create_registration_tab(self, notebook):
        """Create student registration tab"""
        reg_frame = ttk.Frame(notebook)
        notebook.add(reg_frame, text="Registration")
        
        # Form fields
        fields = [
            ("Student ID:", "entry_id"),
            ("Full Name:", "entry_name"),
            ("Department:", "entry_dept"),
            ("Year:", "entry_year"),
            ("Email:", "entry_email")
        ]
        
        self.reg_entries = {}
        
        for i, (label, name) in enumerate(fields):
            ttk.Label(reg_frame, text=label).grid(row=i, column=0, sticky='w', pady=5, padx=10)
            entry = ttk.Entry(reg_frame, width=30)
            entry.grid(row=i, column=1, pady=5, padx=10)
            self.reg_entries[name] = entry
        
        # Department dropdown
        dept_values = ["Computer Science", "Engineering", "Business", "Arts", "Science"]
        self.reg_entries["entry_dept"] = ttk.Combobox(reg_frame, values=dept_values, width=27)
        self.reg_entries["entry_dept"].grid(row=2, column=1, pady=5, padx=10)
        
        # Year dropdown
        year_values = ["1st Year", "2nd Year", "3rd Year", "4th Year"]
        self.reg_entries["entry_year"] = ttk.Combobox(reg_frame, values=year_values, width=27)
        self.reg_entries["entry_year"].grid(row=3, column=1, pady=5, padx=10)
        
        # Buttons
        button_frame = ttk.Frame(reg_frame)
        button_frame.grid(row=len(fields), column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Capture Face Data", 
                  command=self.capture_face_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Train Model", 
                  command=self.train_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", 
                  command=self.clear_registration_form).pack(side=tk.LEFT, padx=5)
        
        # Progress bar for capture
        self.capture_progress = ttk.Progressbar(reg_frame, length=300, mode='determinate')
        self.capture_progress.grid(row=len(fields)+1, column=0, columnspan=2, pady=10, padx=10)
        
        # Status label
        self.reg_status = ttk.Label(reg_frame, text="", foreground=self.colors['accent'])
        self.reg_status.grid(row=len(fields)+2, column=0, columnspan=2, pady=5)
        
    def create_attendance_tab(self, notebook):
        """Create attendance tracking tab"""
        att_frame = ttk.Frame(notebook)
        notebook.add(att_frame, text="Attendance")
        
        # Control buttons
        control_frame = ttk.Frame(att_frame)
        control_frame.pack(fill=tk.X, pady=10)
        
        self.start_btn = ttk.Button(control_frame, text="Start Recognition", 
                                    command=self.start_recognition)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Recognition", 
                                   command=self.stop_recognition, state='disabled')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Manual Entry", 
                  command=self.manual_attendance).pack(side=tk.LEFT, padx=5)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(att_frame, text="Today's Statistics")
        stats_frame.pack(fill=tk.X, pady=10, padx=10)
        
        self.stats_text = tk.Text(stats_frame, height=4, width=40, bg=self.colors['surface'], 
                                  fg=self.colors['fg'], font=('Arial', 10))
        self.stats_text.pack(padx=10, pady=10)
        self.stats_text.insert('1.0', "No data for today")
        self.stats_text.config(state='disabled')
        
        # Recent attendance list
        list_frame = ttk.LabelFrame(att_frame, text="Recent Attendance")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        # Treeview for attendance
        columns = ('Time', 'Name', 'Department', 'Status')
        self.attendance_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.attendance_tree.heading(col, text=col)
            self.attendance_tree.column(col, width=80)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.attendance_tree.yview)
        self.attendance_tree.configure(yscrollcommand=scrollbar.set)
        
        self.attendance_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_reports_tab(self, notebook):
        """Create reports and export tab"""
        report_frame = ttk.Frame(notebook)
        notebook.add(report_frame, text="Reports")
        
        # Date range selection
        date_frame = ttk.LabelFrame(report_frame, text="Select Date Range")
        date_frame.pack(fill=tk.X, pady=10, padx=10)
        
        ttk.Label(date_frame, text="From:").grid(row=0, column=0, padx=5, pady=5)
        self.from_date = ttk.Entry(date_frame, width=15)
        self.from_date.grid(row=0, column=1, padx=5, pady=5)
        self.from_date.insert(0, datetime.now().strftime("%Y-%m-%d"))
        
        ttk.Label(date_frame, text="To:").grid(row=0, column=2, padx=5, pady=5)
        self.to_date = ttk.Entry(date_frame, width=15)
        self.to_date.grid(row=0, column=3, padx=5, pady=5)
        self.to_date.insert(0, datetime.now().strftime("%Y-%m-%d"))
        
        # Filter options
        filter_frame = ttk.Frame(report_frame)
        filter_frame.pack(fill=tk.X, pady=10, padx=10)
        
        ttk.Label(filter_frame, text="Department:").pack(side=tk.LEFT, padx=5)
        self.filter_dept = ttk.Combobox(filter_frame, values=["All", "Computer Science", "Engineering", "Business"], width=20)
        self.filter_dept.pack(side=tk.LEFT, padx=5)
        self.filter_dept.set("All")
        
        # Buttons
        btn_frame = ttk.Frame(report_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Generate Report", 
                  command=self.generate_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Export to Excel", 
                  command=self.export_to_excel).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Export to PDF", 
                  command=self.export_to_pdf).pack(side=tk.LEFT, padx=5)
        
        # Preview area
        preview_frame = ttk.LabelFrame(report_frame, text="Report Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        self.preview_text = tk.Text(preview_frame, height=15, bg=self.colors['surface'], 
                                    fg=self.colors['fg'], font=('Courier', 9))
        scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.preview_text.yview)
        self.preview_text.configure(yscrollcommand=scrollbar.set)
        
        self.preview_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_status_bar(self):
        """Create status bar at bottom"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(status_frame, text="System Ready", 
                                      relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.time_label = ttk.Label(status_frame, text="", relief=tk.SUNKEN, width=20)
        self.time_label.pack(side=tk.RIGHT)
        
        # Update time
        self.update_time()
        
    def update_time(self):
        """Update time in status bar"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)
        
    def load_students(self):
        """Load student data from file"""
        students_file = os.path.join(self.face_data_path, "students.pkl")
        if os.path.exists(students_file):
            with open(students_file, 'rb') as f:
                return pickle.load(f)
        return {}
        
    def save_students(self):
        """Save student data to file"""
        students_file = os.path.join(self.face_data_path, "students.pkl")
        with open(students_file, 'wb') as f:
            pickle.dump(self.students_data, f)
            
    def load_attendance_data(self):
        """Load existing attendance data"""
        if os.path.exists("attendance_log.xlsx"):
            try:
                df = pd.read_excel("attendance_log.xlsx")
                self.attendance_data = df.to_dict('records')
            except:
                self.attendance_data = []
                
    def load_trained_model(self):
        """Load trained model if exists"""
        if os.path.exists("face_recognizer.yml"):
            try:
                self.recognizer.read("face_recognizer.yml")
                self.update_status("Model loaded successfully")
            except:
                self.update_status("Failed to load model")
                
    def capture_face_data(self):
        """Capture face data for registration"""
        student_id = self.reg_entries["entry_id"].get()
        name = self.reg_entries["entry_name"].get()
        dept = self.reg_entries["entry_dept"].get()
        year = self.reg_entries["entry_year"].get()
        email = self.reg_entries["entry_email"].get()
        
        if not all([student_id, name, dept, year]):
            messagebox.showerror("Error", "Please fill all required fields")
            return
            
        if not student_id.isdigit():
            messagebox.showerror("Error", "Student ID must be a number")
            return
            
        face_id = int(student_id)
        
        # Save student info
        self.students_data[face_id] = {
            'name': name,
            'department': dept,
            'year': year,
            'email': email,
            'registered_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save_students()
        
        # Capture face data
        self.capture_progress['value'] = 0
        self.reg_status.config(text="Capturing... Look at the camera")
        self.root.update()
        
        def capture_thread():
            student_dir = os.path.join(self.face_data_path, str(face_id))
            os.makedirs(student_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
                
            count = 0
            max_images = 100
            
            while count < max_images:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    count += 1
                    face_img = gray[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (200, 200))
                    cv2.imwrite(f"{student_dir}/{count}.jpg", face_img)
                    
                    # Update progress
                    progress = int((count / max_images) * 100)
                    self.capture_progress['value'] = progress
                    self.root.update()
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Capturing: {count}/{max_images}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow("Capturing Face Data", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            self.reg_status.config(text=f"Captured {count} images successfully!")
            messagebox.showinfo("Success", f"Face data captured for {name}")
            
        threading.Thread(target=capture_thread, daemon=True).start()
        
    def train_model(self):
        """Train the face recognition model"""
        if not self.students_data:
            messagebox.showerror("Error", "No students registered")
            return
            
        self.update_status("Training model...")
        
        def train_thread():
            faces = []
            ids = []
            
            for face_id in os.listdir(self.face_data_path):
                face_dir = os.path.join(self.face_data_path, face_id)
                if not os.path.isdir(face_dir):
                    continue
                    
                for img_file in os.listdir(face_dir):
                    if img_file.endswith('.jpg'):
                        img_path = os.path.join(face_dir, img_file)
                        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if gray_img is not None:
                            faces.append(gray_img)
                            ids.append(int(face_id))
            
            if not faces:
                messagebox.showerror("Error", "No training data found")
                return
                
            # Train recognizer
            self.recognizer.train(faces, np.array(ids))
            self.recognizer.save("face_recognizer.yml")
            
            self.update_status("Model training complete!")
            messagebox.showinfo("Success", f"Model trained with {len(faces)} images")
            
        threading.Thread(target=train_thread, daemon=True).start()
        
    def start_recognition(self):
        """Start face recognition"""
        if not os.path.exists("face_recognizer.yml"):
            messagebox.showerror("Error", "Please train the model first")
            return
            
        self.running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.logged_today.clear()
        
        self.update_status("Recognition started")
        
        self.recognition_thread = threading.Thread(target=self.recognition_loop, daemon=True)
        self.recognition_thread.start()
        
    def stop_recognition(self):
        """Stop face recognition"""
        self.running = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
        if self.cap:
            self.cap.release()
            
        self.update_status("Recognition stopped")
        
    def recognition_loop(self):
        """Main recognition loop"""
        self.cap = cv2.VideoCapture(0)
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face_region = gray[y:y+h, x:x+w]
                face_region = cv2.resize(face_region, (200, 200))
                
                try:
                    face_id, confidence = self.recognizer.predict(face_region)
                    confidence_percent = 100 - confidence
                    
                    if confidence_percent > 50:
                        # Known face
                        student_info = self.students_data.get(face_id, {})
                        name = student_info.get('name', f"ID: {face_id}")
                        dept = student_info.get('department', 'Unknown')
                        
                        # Log attendance if not already logged today
                        if face_id not in self.logged_today:
                            self.log_attendance(face_id, name, dept, confidence_percent)
                            self.logged_today.add(face_id)
                            
                        color = (0, 255, 0)
                        label = f"{name} ({confidence_percent:.1f}%)"
                    else:
                        # Unknown face
                        color = (0, 0, 255)
                        label = "Unknown"
                        
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, label, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                except:
                    pass
            
            # Update video feed in GUI
            self.update_video_feed(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        
    def update_video_feed(self, frame):
        """Update video feed in GUI"""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (580, 460))
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
    def log_attendance(self, face_id, name, department, confidence):
        """Log attendance to data structure"""
        timestamp = datetime.now()
        
        attendance_record = {
            'Face ID': face_id,
            'Name': name,
            'Department': department,
            'Date': timestamp.strftime("%Y-%m-%d"),
            'Time': timestamp.strftime("%H:%M:%S"),
            'Confidence': f"{confidence:.1f}%",
            'Status': 'Present'
        }
        
        self.attendance_data.append(attendance_record)
        
        # Update treeview
        self.attendance_tree.insert('', 0, values=(
            timestamp.strftime("%H:%M:%S"),
            name,
            department,
            'Present'
        ))
        
        # Update statistics
        self.update_statistics()
        
        # Save to Excel
        self.save_attendance_to_excel()
        
    def update_statistics(self):
        """Update attendance statistics"""
        today = datetime.now().strftime("%Y-%m-%d")
        today_records = [r for r in self.attendance_data if r['Date'] == today]
        
        total = len(today_records)
        by_dept = defaultdict(int)
        
        for record in today_records:
            by_dept[record['Department']] += 1
            
        self.stats_text.config(state='normal')
        self.stats_text.delete('1.0', tk.END)
        self.stats_text.insert('1.0', f"Total Present Today: {total}\n")
        for dept, count in by_dept.items():
            self.stats_text.insert(tk.END, f"{dept}: {count}\n")
        self.stats_text.config(state='disabled')
        
    def save_attendance_to_excel(self):
        """Save attendance data to Excel"""
        if self.attendance_data:
            df = pd.DataFrame(self.attendance_data)
            df.to_excel("attendance_log.xlsx", index=False)
            
    def manual_attendance(self):
        """Open manual attendance entry dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Manual Attendance Entry")
        dialog.geometry("300x200")
        
        ttk.Label(dialog, text="Select Student:").pack(pady=10)
        
        # Create student list
        student_list = ttk.Combobox(dialog, values=[
            f"{data['name']} (ID: {sid})" 
            for sid, data in self.students_data.items()
        ], width=30)
        student_list.pack(pady=5)
        
        def mark_present():
            selection = student_list.get()
            if selection:
                # Extract ID from selection
                student_id = int(selection.split("ID: ")[1].rstrip(")"))
                student = self.students_data[student_id]
                
                self.log_attendance(
                    student_id,
                    student['name'],
                    student['department'],
                    100.0
                )
                
                messagebox.showinfo("Success", f"Marked {student['name']} as present")
                dialog.destroy()
                
        ttk.Button(dialog, text="Mark Present", command=mark_present).pack(pady=20)
        
    def generate_report(self):
        """Generate attendance report"""
        from_date = self.from_date.get()
        to_date = self.to_date.get()
        dept_filter = self.filter_dept.get()
        
        # Filter data
        filtered_data = []
        for record in self.attendance_data:
            if from_date <= record['Date'] <= to_date:
                if dept_filter == "All" or record['Department'] == dept_filter:
                    filtered_data.append(record)
                    
        # Display in preview
        self.preview_text.config(state='normal')
        self.preview_text.delete('1.0', tk.END)
        
        if filtered_data:
            self.preview_text.insert('1.0', "ATTENDANCE REPORT\n")
            self.preview_text.insert(tk.END, "="*50 + "\n")
            self.preview_text.insert(tk.END, f"Period: {from_date} to {to_date}\n")
            if dept_filter != "All":
                self.preview_text.insert(tk.END, f"Department: {dept_filter}\n")
            self.preview_text.insert(tk.END, "="*50 + "\n\n")
            
            for record in filtered_data:
                self.preview_text.insert(tk.END, 
                    f"{record['Date']} {record['Time']} - {record['Name']} "
                    f"({record['Department']}) - {record['Status']}\n")
                    
            self.preview_text.insert(tk.END, f"\nTotal Records: {len(filtered_data)}")
        else:
            self.preview_text.insert('1.0', "No data found for selected criteria")
            
        self.preview_text.config(state='disabled')
        
    def export_to_excel(self):
        """Export report to Excel"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if filename:
            df = pd.DataFrame(self.attendance_data)
            df.to_excel(filename, index=False)
            messagebox.showinfo("Success", f"Report exported to {filename}")
            
    def export_to_pdf(self):
        """Export report to PDF"""
        messagebox.showinfo("Info", "PDF export feature coming soon!")
        
    def clear_registration_form(self):
        """Clear registration form fields"""
        for entry in self.reg_entries.values():
            if isinstance(entry, ttk.Entry):
                entry.delete(0, tk.END)
                
    def update_status(self, message):
        """Update status bar message"""
        self.status_label.config(text=message)
        
# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = ModernFaceRecognitionSystem(root)
    root.mainloop()