# Clinical Trial AI Assistant

This project provides an AI-powered tool for querying clinical trial data and generating insights. Leveraging SQL and AI-driven natural language processing, it enables users to explore clinical trial data, ask questions, and obtain meaningful responses.

## Features

- **Interactive Clinical Trial Querying**: Users can ask questions related to clinical trials through natural language prompts.
- **Database Connectivity**: Connect to a MySQL database containing clinical trial data for real-time querying.
- **Data Viewing**: Users can view specific clinical trial tables and examine the underlying data.

## Setup

### Prerequisites

- **Python**
- **MySQL Database** with clinical trial data
- **API Key for LLM**: Ensure you have a valid API key for Groq or similar generative language models.

### Installation

1. **Clone the Repository**:

Install Dependencies:

pip install -r requirements.txt
Environment Variables:

Create a .env file with the following variables:
makefile

GROQ_API_KEY=<your_groq_api_key>
Configure Database Connection: When running the app, provide MySQL connection details in the Streamlit sidebar.

Database Configuration
The application connects to a MySQL database containing clinical trial data. Tables should include fields for trials, participants, treatments, and outcomes as defined in the sample SQL schema.

Running the Application
Run the application using Streamlit:

streamlit run app.py
Usage
Ask AI about Clinical Trials:
Enter a clinical trial-related question in the chat interface. The application uses prompts from CLINICAL_TRIAL_PROMPTS to contextualize responses.
View Clinical Trial Data:
Select a clinical trial table to view its data in a tabular format.
Prompts
The assistant can respond to questions such as:

"What are the demographics of patients enrolled in the clinical trial for [specific treatment]?"
"How many patients were enrolled in the [specific trial name] and what were the enrollment criteria?"
"Can you summarize the primary and secondary outcomes of the clinical trial named [specific trial name]?"
For a complete list of sample prompts, please take a look at the CLINICAL_TRIAL_PROMPTS variable in app.py.

Troubleshooting
Database Connection Errors: Ensure correct MySQL credentials and that the database server is running.
API Errors: Verify that your API key is valid and that the environment variable is correctly set.
