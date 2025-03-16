# AI Resume Scanning & Candidate Ranking System

## Overview
This system scans and ranks resumes based on a given job description. It uses AI to match resume content with job requirements and assigns a match score.

## Features
- Upload multiple resumes (PDF format).
- Enter job descriptions to find the best matches.
- Automated ranking based on match percentage.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <project_directory>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```

## Requirements
Create a `requirements.txt` file and add the following dependencies:
```
Flask
PyPDF2
pdfminer.six
scikit-learn
spaCy
```

## Usage
1. Start the application.
2. Enter a job description.
3. Upload resumes (PDF format).
4. Click 'Rank Resumes' to see the best-matching candidates.

## License
This project is open-source under the MIT License.
