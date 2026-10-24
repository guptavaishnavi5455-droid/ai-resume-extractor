import os
import json
from pypdf import PdfReader
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

client = genai.Client(api_key=api_key)


class ResumeData(BaseModel):
    name: str = Field(description="Candidate's full name")
    email: str = Field(description="Email address or Not found")
    phone: str = Field(description="Phone number or Not found")
    education: list[str]
    skills: list[str]


def extract_resume_text(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join(
        page.extract_text() or "" for page in reader.pages
    )


def extract_with_gemini(resume_text):
    prompt = f"""
You are a resume information extraction assistant.

Extract information from the resume below.
Return only information supported by the resume.
Do not invent missing details. Use "Not found" for a missing
name, email, or phone number. Use empty lists when education
or skills are not found.

Return valid JSON with these exact fields:
name, email, phone, education, skills.

RESUME TEXT:
{resume_text}
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ResumeData,
        },
    )

    return ResumeData.model_validate_json(response.text)


if __name__ == "__main__":
    pdf_path = input(
        "Enter the full path of your PDF resume: "
    ).strip().strip('"')

    try:
        resume_text = extract_resume_text(pdf_path)

        if not resume_text.strip():
            raise ValueError(
                "No text found. The PDF may be scanned or image-only."
            )

        resume = extract_with_gemini(resume_text)

        print("\n--- Extracted Resume JSON ---")
        print(json.dumps(resume.model_dump(), indent=4))
        print("\nLLM extraction and Pydantic validation successful!")

    except ValidationError as error:
        print("Validation error:", error)
    except Exception as error:
        print("Error:", error)
        