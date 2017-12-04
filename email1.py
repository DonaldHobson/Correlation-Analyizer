# Import smtplib for the actual sending function
print(-1)
import smtplib
print(0)

me = "dph39@cam.ac.uk"#donaldphobson@yahoo.co.uk"

#import smtplib
print(1)
server = smtplib.SMTP('smtp.hermes.cam.ac.uk')
print(2)
server.starttls()
print(3)
server.login(me, "fjkJTV4VJF8$ychY")#"YOUR PASSWORD")"awefHRoinl%8islD")#
print(4)
msg = "YOUR MESSAGE!"
server.sendmail("YOUR EMAIL ADDRESS", "THE EMAIL ADDRESS TO SEND TO", msg)
print(5)
server.quit()
print(6)
###import smtplib
##
##sender = me
##receivers = [me]
##
##message = """From: From Person <donaldphobson@yahoo.co.uk>
##To: To Person <donaldphobson@yahoo.co.uk>
##Subject: SMTP e-mail test
##
##This is a test e-mail message.
##"""
##
##try:
##   smtpObj = smtplib.SMTP('localhost')
##   smtpObj.sendmail(sender, receivers, message)         
##   print( "Successfully sent email")
##except SMTPException:
##   print( "Error: unable to send email")

   
##server = smtplib.SMTP('pop.mail.yahoo.com', 995)
##print(1)
###Next, log in to the server
##me = "donaldphobson@yahoo.co.uk"
##server.login(me, "8ckewmtkapqnn3")
##print(2)
###Send the mail
##msg = "\nHello! Dad. Email sent " # The /n separates the message from the headers
##server.sendmail(me, me, msg)
##print(3)

# Import the email modules we'll need
##from email.mime.text import MIMEText
##
### Open a plain text file for reading.  For this example, assume that
### the text file contains only ASCII characters.
###with open(textfile) as fp:
##    # Create a text/plain message
##msg = MIMEText("hello")
##
##me = "donaldphobson@yahoo.co.uk"
##you =me
##msg['Subject'] = 'The contents'
##msg['From'] = me
##msg['To'] = you
##
### Send the message via our own SMTP server.
##print(2)
###s = smtplib.SMTP('localhost')
##s = smtplib.SMTP('pop.mail.yahoo.com', 995)
##print(3)
##s.send_message(msg)
##print(4)
##s.quit()
