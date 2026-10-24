# AI Resume Information Extractor

## Project Overview
This project uses a Large Language Model (LLM) to extract structured information from PDF resumes and return the results in JSON format.

## Features
- Extracts text from PDF resumes.
- Uses Google Gemini to identify resume information.
- Extracts name, email, phone, education, and skills.
- Uses Pydantic to validate the extracted data.
- Returns formatted JSON output.
- Avoids inventing missing information.

## Technologies Used
- Python
- Google Gemini API
- Pydantic
- PyPDF
- python-dotenv

## Project Workflow
1. User provides a PDF resume path.
2. PyPDF extracts the resume text.
3. Gemini processes the text using a structured prompt.
4. Pydantic validates the extracted information.
5. The program displays the results as JSON.

## Setup Instructions
1. Install Python.
2. Clone or download this repository.
3. Create a virtual environment.
4. Install the required packages:
   `pip install google-genai pydantic python-dotenv pypdf`
5. Create a `.env` file in the project folder and add:
   `GEMINI_API_KEY=your_api_key_here`
6. Run the program:
   `python main.py`

## Security
Keep your Gemini API key private. Never upload your `.env` file.

## Author
AI & Data Science Engineering Student