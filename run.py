import subprocess
import smtplib
import os
import platform
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.mime.text import MIMEText


# Configuration
MATHEMATICA_SCRIPT_1 = "Dysprosium_in_3D_harmonic_trap.nb"  # Path to the first Mathematica notebook
MATHEMATICA_SCRIPT_2 = "Dysprosium 3D scattering.nb"  # Path to the second Mathematica notebook
PYTHON_SCRIPT = "plots.py"  # Path to the Python script
FILES_TO_SEND = ["3D_Bound_state_energy.png", "3D_Bound_state_energy_before_scat.png",
                 "Drop_3D_30_07.csv", "Scattered_02.csv", "Spectral_colored.csv",
                 "Before_scat_colored.csv"]  # List of files to send via email
EMAIL_ADDRESS = "michalzkod@gmail.com"  # Your email address
EMAIL_PASSWORD = "Kapibara"  # Your email password
RECIPIENT_EMAIL = "m.zdziennick@student.uw.edu.pl"  # Recipient's email address
SMTP_SERVER = "smtp.gmail.com"  # SMTP server, e.g., "smtp.gmail.com" for Gmail
SMTP_PORT = 587  # SMTP port, usually 587 for TLS

def run_mathematica_notebook(notebook_path):
    """Runs a Mathematica notebook using wolframscript."""
    try:
        result = subprocess.run(['wolframscript', '-file', notebook_path], check=True, capture_output=True, text=True)
        print(f"Successfully ran {notebook_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {notebook_path}: {e.stderr}")

def run_python_script(script_path):
    """Runs a Python script."""
    try:
        result = subprocess.run(['python', script_path], check=True, capture_output=True, text=True)
        print(f"Successfully ran {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e.stderr}")

def send_email(files):
    """Sends an email with attachments."""
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = "Obliczenia"

    body = "So simple."
    msg.attach(MIMEText(body, 'plain'))

    for file_path in files:
        if os.path.exists(file_path):
            attachment = MIMEBase('application', 'octet-stream')
            with open(file_path, 'rb') as file:
                attachment.set_payload(file.read())
            encoders.encode_base64(attachment)
            attachment.add_header('Content-Disposition', f'attachment; filename={os.path.basename(file_path)}')
            msg.attach(attachment)
        else:
            print(f"File not found: {file_path}")

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

def shutdown_system():
    """Shuts down the Windows system."""
    try:
        # Use the Windows shutdown command to shut down the system immediately
        subprocess.run(['shutdown', '/s', '/t', '0'], check=True)
        print("System will shut down now.")
    except Exception as e:
        print(f"Failed to shut down the system: {e}")

def main():
    # # Prompt user for shutdown
    # shutdown_after = input("Do you want to shut down the system after the script completes? [Y/n]: ").strip().lower() == 'y'

    # # Step 1: Run Mathematica notebooks
    # run_mathematica_notebook(MATHEMATICA_SCRIPT_1)
    # run_mathematica_notebook(MATHEMATICA_SCRIPT_2)

    # # Step 2: Run Python script
    # run_python_script(PYTHON_SCRIPT)

    # Step 3: Send files via email
    send_email(FILES_TO_SEND)

    # # Step 4: Conditionally shut down the system if Windows
    # if shutdown_after and platform.system() == 'Windows':
    #     shutdown_system()

if __name__ == "__main__":
    main()
