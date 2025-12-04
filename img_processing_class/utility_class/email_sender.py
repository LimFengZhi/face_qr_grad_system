import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
import threading
from typing import Tuple, Optional

# ==================== HARDCODED EMAIL CREDENTIALS ====================
# For Gmail: Use App Password (not regular password)
# 1. Enable 2-Factor Authentication on Gmail
# 2. Go to: https://myaccount.google.com/apppasswords
# 3. Generate App Password and paste below

SENDER_EMAIL = "targradce@gmail.com"  # <-- Change this to your Gmail
SENDER_PASSWORD = "jxyr rzsf hvlx nftz"  # <-- Change this to your App Password

# =====================================================================

class EmailSender:
    """Send graduation QR code emails to students"""
    
    def __init__(self, smtp_server: str = "smtp.gmail.com", smtp_port: int = 587):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = SENDER_EMAIL
        self.sender_password = SENDER_PASSWORD
        self.qr_folder = "data/qr_codes"
    
    def find_qr_image(self, student_id: str) -> Optional[str]:
        """Find QR code image for student"""
        qr_path = f"{self.qr_folder}/qr_{student_id}.png"
        if os.path.exists(qr_path):
            return qr_path
        return None
    
    def send_qr_email(self, student: dict) -> Tuple[bool, str]:
        """
        Send QR code email to student.
        
        Args:
            student: dict with student_id, name, email, course, faculty
            
        Returns:
            (success: bool, message: str)
        """
        if not self.sender_email or not self.sender_password or self.sender_email == "your_email@gmail.com":
            return False, "Email credentials not configured in email_sender.py"
        
        student_email = student.get('email')
        if not student_email or student_email == 'tt@gmail.com':
            return False, "Invalid student email"
        
        student_id = student.get('student_id')
        student_name = student.get('name')
        course = student.get('course', 'N/A')
        faculty = student.get('faculty', 'N/A')
        
        # Find QR code
        qr_path = self.find_qr_image(student_id)
        if not qr_path:
            return False, f"QR code not found for {student_id}"
        
        try:
            # Create message
            msg = MIMEMultipart('related')
            msg['Subject'] = f"🎓 Graduation Ceremony - Your QR Code ({student_id})"
            msg['From'] = self.sender_email
            msg['To'] = student_email
            
            # HTML body
            html = f"""
            <html>
            <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px; text-align: center; color: white;">
                    <h1>🎓 Congratulations!</h1>
                    <h2>{student_name}</h2>
                </div>
                
                <div style="padding: 20px; background: #f9f9f9; border-radius: 10px; margin-top: 20px;">
                    <h3>Your Graduation Details:</h3>
                    <p><strong>Student ID:</strong> {student_id}</p>
                    <p><strong>Faculty:</strong> {faculty}</p>
                    <p><strong>Course:</strong> {course}</p>
                </div>
                
                <div style="text-align: center; padding: 30px; background: white; border-radius: 10px; margin-top: 20px; border: 2px dashed #667eea;">
                    <h3>Your QR Code</h3>
                    <p>Please present this QR code at the graduation ceremony for verification.</p>
                    <img src="cid:qrcode" style="max-width: 250px; margin: 20px auto;" />
                </div>
                
                <div style="padding: 20px; text-align: center; color: #666; font-size: 12px;">
                    <p>This is an automated message from the Graduation Ceremony System.</p>
                    <p>Please do not reply to this email.</p>
                </div>
            </body>
            </html>
            """
            
            msg_alternative = MIMEMultipart('alternative')
            msg.attach(msg_alternative)
            msg_alternative.attach(MIMEText(html, 'html'))
            
            # Attach QR code image
            with open(qr_path, 'rb') as f:
                qr_image = MIMEImage(f.read())
                qr_image.add_header('Content-ID', '<qrcode>')
                qr_image.add_header('Content-Disposition', 'inline', filename=f'qr_{student_id}.png')
                msg.attach(qr_image)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            return True, f"Email sent to {student_email}"
            
        except smtplib.SMTPAuthenticationError:
            return False, "Email authentication failed. Check credentials."
        except Exception as e:
            return False, f"Failed to send email: {str(e)}"
    
    def send_qr_email_async(self, student: dict, callback=None):
        """Send email in background thread"""
        def _send():
            success, message = self.send_qr_email(student)
            if callback:
                callback(success, message, student)
            else:
                if success:
                    print(f"✉️ {message}")
                else:
                    print(f"❌ Email failed: {message}")
        
        thread = threading.Thread(target=_send, daemon=True)
        thread.start()
        return thread


# Singleton instance
_email_sender = None

def get_email_sender() -> EmailSender:
    """Get or create email sender instance"""
    global _email_sender
    if _email_sender is None:
        _email_sender = EmailSender()
    return _email_sender