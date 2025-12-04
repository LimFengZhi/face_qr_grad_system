import sqlite3
import pandas as pd
import os
from datetime import datetime
from typing import Optional, Dict, List

class StudentDatabase:
    """SQLite database for graduation ceremony attendance"""
    
    def __init__(self, db_path: str = "students.db"):
        self.db_path = db_path
        self._create_tables()
    
    def _get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def _create_tables(self):
        """Create tables if they don't exist"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Student table with attendance status
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                cgpa REAL,
                faculty TEXT,
                course TEXT,
                email TEXT,
                attended INTEGER DEFAULT 0,
                attendance_time DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✓ Database initialized: {self.db_path}")
    
    # ==================== STUDENT OPERATIONS ====================
    
    def add_student(self, student_id: str, name: str, cgpa: float = None, 
                    faculty: str = None, course: str = None, email: str = None) -> bool:
        """Add a new student"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO students (student_id, name, cgpa, faculty, course, email, attended, attendance_time)
                VALUES (?, ?, ?, ?, ?, ?, 0, NULL)
            ''', (student_id, name, cgpa, faculty, course, email))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding student: {e}")
            return False
    
    def get_student(self, student_id: str) -> Optional[Dict]:
        """Get student by ID"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM students WHERE student_id = ?', (student_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'student_id': row[0],
                    'name': row[1],
                    'cgpa': row[2],
                    'faculty': row[3],
                    'course': row[4],
                    'email': row[5],
                    'attended': bool(row[6]),
                    'attendance_time': row[7]
                }
            return None
        except Exception as e:
            print(f"Error getting student: {e}")
            return None
    
    def get_all_students(self) -> List[Dict]:
        """Get all students"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM students ORDER BY name')
            rows = cursor.fetchall()
            conn.close()
            
            students = []
            for row in rows:
                students.append({
                    'student_id': row[0],
                    'name': row[1],
                    'cgpa': row[2],
                    'faculty': row[3],
                    'course': row[4],
                    'email': row[5],
                    'attended': bool(row[6]),
                    'attendance_time': row[7]
                })
            return students
        except Exception as e:
            print(f"Error getting students: {e}")
            return []
    
    def update_student(self, student_id: str, **kwargs) -> bool:
        """Update student info"""
        try:
            allowed_fields = ['name', 'cgpa', 'faculty', 'course', 'email']
            updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
            
            if not updates:
                return False
            
            set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
            values = list(updates.values()) + [student_id]
            
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(f'UPDATE students SET {set_clause} WHERE student_id = ?', values)
            conn.commit()
            success = cursor.rowcount > 0
            conn.close()
            return success
        except Exception as e:
            print(f"Error updating student: {e}")
            return False
    
    def delete_student(self, student_id: str) -> bool:
        """Delete a student"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM students WHERE student_id = ?', (student_id,))
            conn.commit()
            success = cursor.rowcount > 0
            conn.close()
            return success
        except Exception as e:
            print(f"Error deleting student: {e}")
            return False
    
    def search_students(self, query: str) -> List[Dict]:
        """Search students by name or ID"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            search_term = f"%{query}%"
            cursor.execute('''
                SELECT * FROM students 
                WHERE student_id LIKE ? OR name LIKE ?
                ORDER BY name
            ''', (search_term, search_term))
            rows = cursor.fetchall()
            conn.close()
            
            students = []
            for row in rows:
                students.append({
                    'student_id': row[0],
                    'name': row[1],
                    'cgpa': row[2],
                    'faculty': row[3],
                    'course': row[4],
                    'email': row[5],
                    'attended': bool(row[6]),
                    'attendance_time': row[7]
                })
            return students
        except Exception as e:
            print(f"Error searching students: {e}")
            return []
    
    # ==================== ATTENDANCE OPERATIONS ====================
    
    def mark_attended(self, student_id: str) -> bool:
        """Mark student as attended"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute('''
                UPDATE students 
                SET attended = 1, attendance_time = ?
                WHERE student_id = ?
            ''', (current_time, student_id))
            conn.commit()
            success = cursor.rowcount > 0
            conn.close()
            return success
        except Exception as e:
            print(f"Error marking attendance: {e}")
            return False
    
    def mark_not_attended(self, student_id: str) -> bool:
        """Mark student as not attended (reset)"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE students 
                SET attended = 0, attendance_time = NULL
                WHERE student_id = ?
            ''', (student_id,))
            conn.commit()
            success = cursor.rowcount > 0
            conn.close()
            return success
        except Exception as e:
            print(f"Error resetting attendance: {e}")
            return False
    
    def is_attended(self, student_id: str) -> bool:
        """Check if student has attended"""
        student = self.get_student(student_id)
        if student:
            return student['attended']
        return False
    
    def get_attended_students(self) -> List[Dict]:
        """Get all students who have attended"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM students 
                WHERE attended = 1
                ORDER BY attendance_time
            ''')
            rows = cursor.fetchall()
            conn.close()
            
            students = []
            for row in rows:
                students.append({
                    'student_id': row[0],
                    'name': row[1],
                    'cgpa': row[2],
                    'faculty': row[3],
                    'course': row[4],
                    'email': row[5],
                    'attended': bool(row[6]),
                    'attendance_time': row[7]
                })
            return students
        except Exception as e:
            print(f"Error getting attended students: {e}")
            return []
    
    def get_not_attended_students(self) -> List[Dict]:
        """Get all students who have NOT attended"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM students 
                WHERE attended = 0
                ORDER BY name
            ''')
            rows = cursor.fetchall()
            conn.close()
            
            students = []
            for row in rows:
                students.append({
                    'student_id': row[0],
                    'name': row[1],
                    'cgpa': row[2],
                    'faculty': row[3],
                    'course': row[4],
                    'email': row[5],
                    'attended': bool(row[6]),
                    'attendance_time': row[7]
                })
            return students
        except Exception as e:
            print(f"Error getting not attended students: {e}")
            return []
    
    def reset_all_attendance(self) -> bool:
        """Reset attendance for all students (for new ceremony)"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('UPDATE students SET attended = 0, attendance_time = NULL')
            conn.commit()
            conn.close()
            print("✓ All attendance reset")
            return True
        except Exception as e:
            print(f"Error resetting all attendance: {e}")
            return False
    
    def get_attendance_stats(self) -> Dict:
        """Get attendance statistics"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM students')
            total = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM students WHERE attended = 1')
            attended = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total': total,
                'attended': attended,
                'not_attended': total - attended,
                'attendance_rate': round((attended / total * 100), 1) if total > 0 else 0
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {'total': 0, 'attended': 0, 'not_attended': 0, 'attendance_rate': 0}
    
    # ==================== IMPORT/EXPORT ====================
    
    def import_from_csv(self, csv_path: str) -> int:
        """Import students from CSV file"""
        try:
            if not os.path.exists(csv_path):
                print(f"CSV file not found: {csv_path}")
                return 0
            
            df = pd.read_csv(csv_path)
            
            # Map column names (handle different formats)
            column_mapping = {
                'student_id': ['student_id', 'id', 'ID', 'StudentID'],
                'name': ['name', 'Name', 'student_name'],
                'cgpa': ['cgpa', 'CGPA', 'gpa'],
                'faculty': ['faculty', 'Faculty'],
                'course': ['course', 'Course', 'program'],
                'email': ['email', 'Email']
            }
            
            # Find actual column names
            actual_columns = {}
            for target, possibilities in column_mapping.items():
                for col in possibilities:
                    if col in df.columns:
                        actual_columns[target] = col
                        break
            
            count = 0
            conn = self._get_connection()
            cursor = conn.cursor()
            
            for _, row in df.iterrows():
                try:
                    student_id = str(row.get(actual_columns.get('student_id', ''), ''))
                    name = str(row.get(actual_columns.get('name', ''), ''))
                    
                    if not student_id or not name:
                        continue
                    
                    cgpa = row.get(actual_columns.get('cgpa', ''), None)
                    faculty = row.get(actual_columns.get('faculty', ''), None)
                    course = row.get(actual_columns.get('course', ''), None)
                    email = row.get(actual_columns.get('email', ''), None)
                    
                    # Handle NaN values
                    if pd.isna(cgpa):
                        cgpa = None
                    if pd.isna(faculty):
                        faculty = None
                    if pd.isna(course):
                        course = None
                    if pd.isna(email):
                        email = None
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO students (student_id, name, cgpa, faculty, course, email, attended, attendance_time)
                        VALUES (?, ?, ?, ?, ?, ?, 0, NULL)
                    ''', (student_id, name, cgpa, faculty, course, email))
                    count += 1
                    
                except Exception as e:
                    print(f"Error importing row: {e}")
                    continue
            
            conn.commit()
            conn.close()
            print(f"✓ Imported {count} students from {csv_path}")
            return count
            
        except Exception as e:
            print(f"Error importing CSV: {e}")
            return 0
    
    def export_to_csv(self, csv_path: str) -> bool:
        """Export students with attendance to CSV file"""
        try:
            students = self.get_all_students()
            if not students:
                print("No students to export")
                return False
            
            df = pd.DataFrame(students)
            df.to_csv(csv_path, index=False)
            print(f"✓ Exported {len(students)} students to {csv_path}")
            return True
        except Exception as e:
            print(f"Error exporting CSV: {e}")
            return False
    
    def export_attendance_report(self, csv_path: str = "attendance_report.csv") -> bool:
        """Export attendance report"""
        try:
            students = self.get_all_students()
            if not students:
                print("No students to export")
                return False
            
            # Format for report
            report_data = []
            for s in students:
                report_data.append({
                    'Student ID': s['student_id'],
                    'Name': s['name'],
                    'Faculty': s['faculty'],
                    'Course': s['course'],
                    'CGPA': s['cgpa'],
                    'Attended': 'Yes' if s['attended'] else 'No',
                    'Check-in Time': s['attendance_time'] if s['attendance_time'] else '-'
                })
            
            df = pd.DataFrame(report_data)
            df.to_csv(csv_path, index=False)
            print(f"✓ Exported attendance report to {csv_path}")
            return True
        except Exception as e:
            print(f"Error exporting report: {e}")
            return False
    
    def get_student_count(self) -> int:
        """Get total number of students"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM students')
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except Exception as e:
            print(f"Error getting count: {e}")
            return 0
    
    # ==================== LOOKUP (for fast access in app) ====================
    
    def build_lookup(self) -> Dict[str, Dict]:
        """Build a dictionary for O(1) student lookup"""
        students = self.get_all_students()
        return {s['student_id']: s for s in students}


# ==================== HELPER FUNCTIONS ====================

def init_database(csv_path: str = "student_list.csv", db_path: str = "students.db") -> StudentDatabase:
    """Initialize database and import from CSV if database is empty"""
    db = StudentDatabase(db_path)
    
    # If database is empty, import from CSV
    if db.get_student_count() == 0 and os.path.exists(csv_path):
        print(f"Database empty, importing from {csv_path}...")
        db.import_from_csv(csv_path)
    
    return db

def graduation_level(cgpa):
    if cgpa >= 3.67:
        return "Distinction"
    elif cgpa >= 2.67:
        return "Merit"
    elif cgpa >= 2.0:
        return "Pass"
    else:
        return "Fail"


# ==================== TEST ====================
if __name__ == "__main__":
    # Test the database
    db = init_database("student_list.csv")
    
    print(f"\n📊 Total students: {db.get_student_count()}")
    
    # Test get student
    student = db.get_student("24WMR08011")
    if student:
        print(f"\nFound student: {student['name']}")
    
    # Test search
    results = db.search_students("Lim")
    print(f"\nSearch 'Lim': {len(results)} results")
    for s in results:
        print(f"  - {s['student_id']}: {s['name']}")