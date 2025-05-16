# 🔗 PDF ChatBot 

A ChatBot allowing the user to upload any PDF of their liking and iteract with it by a chat interface, allowing them to get their FAQ's right away without actually reading the entire PDF. This allows the user to retrive any relevent information that they what from the PDF in matter of seconds.

## 📦 Features

- Upload the PDF to the Chat bot interface.
- Choose the model of their liking which they would want to use for generating responses. The available models are Llama-3, Llama-4, GPT-4o, Gemini-2.5-Flash, Gemma-3.
- A chat bot interface which keeps track of the past user chats in the current chat window.
---

## 🛠️ Tech Stack

- **Python**
- **LangChain**
- **FAISS - Facebook AI Similarity Search**
- **RAGAS - RAG Model Evaluation Framework**

---

## 🚀 Getting Started (Local Setup)

### 1. Extract out the project from the zip file
Start by setting up a python virtual enviorment (preferably select Python 3.11.12 or above for smoother setup)
```bash
python -m venv .venv
```

### 2. Install Dependency Packages
```bash
pip install -r requirements.txt
```

### 3. Setting up .env file
create a new .env file and setup the API keys in the bellow variables
```bash
touch .env
```
Copy the below and populate
```bash
LANGSMITH_TRACING=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=

OPENAI_API_KEY=
GOOGLE_API_KEY=
GROQ_API_KEY=
MISTRAL_API_KEY=
```
### 4. Starting up the Steamlit App
Run the below command to setup the streamlit applilcation
```bash
streamlit run streamlit_app.py
```

### 5. Start up an evaluation chain
On running the ```evaluate.py``` script would start the evaluation process and then create two .xlsx files - generative_evals.xlsx and retriever_evals.xlsx, containing the both retriever evaluation metrics and generative evaluation metrics.
```bash
python evaluate.py
```

### 6. Visualzing the evaluation metrics
You can run the cells in the ```visualize.ipynb``` file to visualize the evaluation metrics in a tabular metrics and also using a bar chart comparing all models.

