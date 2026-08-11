import sys
import os

# Ensure project root is in sys.path so 'app' package can be imported on Vercel
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

# Export app ASGI instance for Vercel Serverless Functions
app = app
