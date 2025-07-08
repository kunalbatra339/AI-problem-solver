---
title: AI Problem Solver Backend
emoji: 💡
colorFrom: teal
colorTo: emerald
sdk: flask
app_file: app.py
python_version: 3.12
---

# AI Problem Solver Backend

This is the backend service for the AI Problem Solver application, deployed on Hugging Face Spaces.

It provides an API endpoint to:
- Receive an image and user context.
- Utilize a PyTorch ResNet model to identify issues (e.g., plant diseases).
- Generate solutions using the Gemini API.

**API Endpoint:** `/identify_issue` (POST request)
- **Form Data:**
    - `image`: The image file to analyze.
    - `context`: Additional text context from the user.

**Model:** `plant_disease_resnet18.pth`
**Class Names:** `class_names.json`

## Setup (for local development)

1. Clone the repository:
   `git clone https://github.com/kunalbatra339/AI-problem-solver.git`
   `cd AI-problem-solver`

2. Create a virtual environment:
   `python -m venv venv`
   `source venv/bin/activate` (Linux/macOS) or `.\venv\Scripts\activate` (Windows)

3. Install dependencies:
   `pip install -r requirements.txt`

4. Set environment variables (e.g., in a `.env` file):
   `MONGO_URI="your_mongodb_connection_string"`
   `GEMINI_API_KEY="your_gemini_api_key"`

5. Run the Flask app:
   `python app.py`

The API will be available at `http://localhost:7860` (or `5000` if you change the default).