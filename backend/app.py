from flask import Flask, request, jsonify
from flask_cors import CORS
from hybrid import summarize_text
import os
import PyPDF2
import docx
import tempfile

app = Flask(__name__)
CORS(app)

def extract_from_pdf(file_path):
    """Extract text from PDF files"""
    text = ""
    with open(file_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_from_docx(file_path):
    """Extract text from DOCX files"""
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_from_txt(file_path):
    """Extract text from TXT files"""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

@app.route("/extract", methods=["POST"])
def extract_text():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        file.save(tmp.name)
        file_path = tmp.name
    
    try:
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext == ".pdf":
            text = extract_from_pdf(file_path)
        elif file_ext == ".docx":
            text = extract_from_docx(file_path)
        elif file_ext == ".txt":
            text = extract_from_txt(file_path)
        else:
            os.unlink(file_path)  # Delete the temporary file
            return jsonify({"error": "Unsupported file format"}), 400
        
        os.unlink(file_path)
        
        return jsonify({"text": text})
    
    except Exception as e:
        if os.path.exists(file_path):
            os.unlink(file_path)
        return jsonify({"error": str(e)}), 500

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "")
    sentences = data.get("sentences", 3)  # Get the desired number of sentences, default to 3
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Pass the number of sentences to your summarizer function
    summary = summarize_text(text, sentences=sentences) 
    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(debug=True)