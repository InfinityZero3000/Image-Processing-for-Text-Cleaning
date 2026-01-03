"""
Vercel API endpoint for DocCleaner AI
Entry point for serverless deployment
"""

from Backend.app import app

# Vercel requires the app to be named 'app' or be exported
# This file acts as the entry point for Vercel's Python runtime
handler = app
