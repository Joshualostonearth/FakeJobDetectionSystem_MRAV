import os
from dotenv import load_dotenv
from twilio.rest import Client

load_dotenv(override=True)
client = Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))

to_num = "+17736932394"
from_num = os.getenv("TWILIO_PHONE_NUMBER")

try:
    print("Testing Twilio with twiml...")
    client.calls.create(
        twiml="<Response><Say>Test</Say></Response>",
        to=to_num,
        from_=from_num
    )
    print("twiml Success!")
except Exception as e:
    print(f"twiml Failed: {e}")

try:
    print("\nTesting Twilio with url...")
    client.calls.create(
        url="http://demo.twilio.com/docs/voice.xml",
        to=to_num,
        from_=from_num
    )
    print("url Success!")
except Exception as e:
    print(f"url Failed: {e}")
