"""
Full end-to-end test: build location context for a known Nairobi location
and send it to Gemini with a real question.
"""
import os, sys
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Patch path so we can import helpers from app.py
sys.path.insert(0, os.path.dirname(__file__))
from app import build_location_context, get_session

# Test coords: near Odeon Cinema / CBD
LAT, LNG = -1.2828, 36.8250

print("=== Location context block ===")
ctx = build_location_context(LAT, LNG)
print(ctx)

print("\n=== Gemini answer ===")
enriched = ctx + "\n\nUSER MESSAGE: I want to go to Naivas Supermarket in Ruaka. Which matatu do I take and how much?"
session = get_session("test-session-001")
reply = session.send_message(enriched)
print(reply.text)
