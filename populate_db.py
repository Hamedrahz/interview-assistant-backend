import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="qa_collection")

# Load sentence transformer for embeddings
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Define your predefined questions and answers
qa_pairs = [
    {"question": "Tell me about your experience in UX design.",
     "answer": "I have 7 years of experience in UX design, working with companies like Verizon and Fandom."},
    
    {"question": "What is your experience in crypto?",
     "answer": "I have worked with Consensys and Ether Capital, designing blockchain-based products."},
    
    {"question": "How do you approach design challenges?",
     "answer": "I use a user-centered approach, conducting research, prototyping, and iterating based on feedback."},

    {"question": "What are your strengths as a product designer?",
     "answer": "My strengths include user research, interaction design, and prototyping with a focus on accessibility."},

    {"question": "How do you handle user feedback?",
     "answer": "I analyze user feedback, prioritize issues, and iterate designs based on real-world insights."},

    {"question": "What tools do you use for product design?",
     "answer": "I primarily use Figma, but I also work with Framer, Sketch, and prototyping tools like Principle."},

    {"question": "Have you worked with design systems?",
     "answer": "Yes, I have experience building and maintaining design systems for large organizations."},

    {"question": "How do you handle tight deadlines?",
     "answer": "I prioritize tasks, break work into sprints, and collaborate effectively with teams to meet deadlines."},

    {"question": "What is your process for creating a design from scratch?",
     "answer": "I start with research, define user personas, create wireframes, iterate prototypes, and refine based on feedback."}
]

# Populate ChromaDB with Q&A pairs
for item in qa_pairs:
    embedding = encoder.encode(item["question"]).tolist()
    collection.add(documents=[item["answer"]], embeddings=[embedding], ids=[item["question"]])

print("✅ ChromaDB has been populated with Q&A pairs!")

