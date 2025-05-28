import smtplib
from email.message import EmailMessage
import mimetypes
import os

# --- CONFIGURATION ---
# Replace with your email and app password
sender_email = 'emmutab@gmail.com'
app_password = 'usro suuz epeo caih'

# Replace with the recipient's email
recipient_email = 'vimalamathew@gmail.com'

# Email content

subject = '‼️‼️‼️Alert: Camera Being Disturbed'
body = 'The image from the feed is attached below.'

# Path to the image you want to attach
image_path = '/Users/emmanuelgeorgep/Documents/Internship/Alert Mechanism/image.jpg'    
# --- CREATE EMAIL MESSAGE ---
msg = EmailMessage()
msg['From'] = sender_email
msg['To'] = recipient_email
msg['Subject'] = subject
msg.set_content(body)

# --- ATTACH IMAGE FILE ---
# Guess the MIME type and encoding
mime_type, _ = mimetypes.guess_type(image_path)
if mime_type is None:
    mime_type = 'application/octet-stream'
maintype, subtype = mime_type.split('/')

with open(image_path, 'rb') as img:
    img_data = img.read()
    img_name = os.path.basename(image_path)
    msg.add_attachment(img_data, maintype=maintype, subtype=subtype, filename=img_name)

# --- SEND EMAIL VIA GMAIL SMTP ---
smtp_server = 'smtp.gmail.com'
smtp_port = 587

try:
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()  # Secure the connection
        server.login(sender_email, app_password)
        server.send_message(msg)
        print("✅ Email sent successfully!")
except Exception as e:
    print(f"❌ Failed to send email: {e}")