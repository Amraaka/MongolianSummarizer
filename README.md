# Mongolian Summerizer

**Mongolian Summerizer** is a simple web application that summarizes lengthy Mongolian text into a short, clear, and understandable version.

## ✨ Features

- 📝 Summarize raw Mongolian text easily.
- 📄 Upload PDF or DOC files and extract their text.
- ⚙️ Uses an enhanced text summarization algorithm with:
  - Sentence position scoring
  - Phrase extraction

## 🖥️ Tech Stack

- **Frontend:** React + Vite  
  - Clean interface to input raw text or upload files.
- **Backend:** Python Flask  
  - Custom text summarization algorithm for Mongolian language.

## 🚀 Getting Started

### Frontend

```bash
cd frontend
npm install
npm run dev

cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
