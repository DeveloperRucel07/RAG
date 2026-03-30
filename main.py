import os
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from google import genai

SUPABASE_URL = ""
SUPABASE_KEY = ""
GEMINI_API = ""