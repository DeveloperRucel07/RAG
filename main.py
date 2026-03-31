import os
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from google import genai
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("API_KEY")


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API = os.getenv("GEMINI_API")

client = genai.Client(api_key=GEMINI_API)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embedding_model = SentenceTransformer("Supabase/gte-small")


def generate_embedding(text):
    return embedding_model.encode(text, normalize_embeddings=True).tolist()


def supabase_document_upload(text: str, title: str | None = None):
    supabase.table("documents").insert(
        {
            "title": title,
            "body": text,
            "embedding": generate_embedding(text)
        }
    ).execute()


def textfile_to_string(file):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
    title = os.path.splitext(os.path.basename(file))[0].lower()
    supabase_document_upload(text, title)


def cleanJson(text):
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```")
        text = text.rsplit("```", 1)[0]
    return text.strip()


def get_prompt_text(userinput, context):
    return f"""
        Du bist ein präziser, sachlicher Textassistent. 

        **Frage eines Users:**
        {userinput}
        
        **Dokumente (Kontext für die Antwort):**
        {context}
   
        **Anweisungen:**
        - Verwende NUR Informationen aus den Dokumenten.
        - Erfinde keine Details.
        - Wenn keine relevanten Informationen enthalten sind, schreibe: "Keine inhaltliche Beschreibung möglich."
        
        Antworte jetzt:
        """


def sending_to_LLM(documents, userinput):
    context = "\n".join(
        [f"Titel: {doc['title']}\nInhalt:\n{doc['body']}" for doc in documents]
    )

    prompt = get_prompt_text(userinput, context)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return (cleanJson(response.text.strip()))

    except Exception as e:
        raise RuntimeError(f"Error updating product description: {e}")
    


if __name__ == "__main__":
    for file in os.listdir("documents"):
        if file.endswith(".txt"):
            textfile_to_string(os.path.join("documents", file))
            print(f"Uploaded {file} to Supabase.")
    print("All documents uploaded to Supabase.")