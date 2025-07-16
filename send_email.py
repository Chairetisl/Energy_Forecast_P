import smtplib
from email.message import EmailMessage
import os

sender = os.getenv("EMAIL_USER")
password = os.getenv("EMAIL_PASS")
receiver = "trading.power@depa.gr"  # άλλαξε το αν θες

msg = EmailMessage()
msg["Subject"] = "Daily Energy Forecast"
msg["From"] = sender
msg["To"] = receiver
msg.set_content("Σας επισυνάπτω το ημερήσιο αρχείο πρόβλεψης.")

with open("predictions_temperature_hour_wind_consumption.xlsx", "rb") as f:
    msg.add_attachment(f.read(), maintype="application",
                       subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       filename="predictions.xlsx")

with smtplib.SMTP("smtp.office365.com", 587) as smtp:
    smtp.starttls()
    smtp.login(sender, password)
    smtp.send_message(msg)

print("✅ Email sent successfully!")
