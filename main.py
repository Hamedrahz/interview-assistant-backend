import json
import queue
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import chromadb
from fastapi import FastAPI, WebSocket
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load Vosk Speech Recognition Model
model = Model("vosk-model-small-en-us-0.15")  # Download & extract this model
recognizer = KaldiRecognizer(model, 16000)
audio_queue = queue.Queue()

# Load ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="qa_collection")

# Load Sentence Transformer for semantic search
encoder = SentenceTransformer("all-MiniLM-L6-v2")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected...")

    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16', channels=1,
                           callback=lambda indata, frames, time, status: audio_queue.put(bytes(indata))):
        while True:
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                question_text = result["text"]

                if question_text.strip():
                    print(f"Recognized: {question_text}")

                    # Search the best matching answer in ChromaDB
                    answer = search_answer(question_text)

                    response = {"question": question_text, "answer": answer}
                    await websocket.send_json(response)

def search_answer(user_question):
    """Search for the best-matching answer in ChromaDB"""
    query_embedding = encoder.encode(user_question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=1)

    if results["documents"] and len(results["documents"][0]) > 0:
        return results["documents"][0][0]
    
    return "Sorry, I don't have an answer for that."

# Function to add QA pairs to ChromaDB (Run this once to populate)
def populate_chroma_db():
    qa_pairs = [
        {"question": "Tell me about your experience in UX design.",
         "answer": "I have 7 years of experience in UX design, working with companies like Verizon and Fandom."},
        {"question": "What is your experience in crypto?",
         "answer": "I have worked with Consensys and Ether Capital, designing blockchain-based products."},
        {"question": "How do you approach design challenges?",
         "answer": "I use a user-centered approach, conducting research, prototyping, and iterating based on feedback."}
    ]
    
    for item in qa_pairs:
        embedding = encoder.encode(item["question"]).tolist()
        collection.add(documents=[item["answer"]], embeddings=[embedding], ids=[item["question"]])

# Uncomment this line and run `python main.py` once to populate ChromaDB
# populate_chroma_db()
