# FraudLens - An Responsible AI Fraud Explanation Chatbot

FraudLens is an interpretable fraud-analysis assistant designed to help analysts understand why a transaction is flagged. They system combines a surrogate Explainable Boosting Machine (EBM), SHAP explanations, and a FAISS-based RAG knowledge layer to turn complex model behavior and rule-based fraud signals into clear, human-readable narratives. It is built for transparency, regulatory alignment, and rapid analyst review in modern fraud-detection workflows.

## Key features

- **LLM-powered interaction & explanation** – routes natural-language questions through OpenAI's `gpt-3.5-turbo` (or any compatible model) via the `OPENAI_API_KEY` environment variable, allowing investigators to request justifications, metrics, or transformations and receive grounded narrative explanations that blend EBM scores, SHAP contributors, and retrieved domain definitions.
- **Explainable ML scoring** – loads an EBM model artifact from `_model_/ebm_fraud_model.pkl` to approximate internal rule-based fraud logic. The surrogate model produces stable, interpretable feature-level risk contributions and transparent per-transaction scores.
- **Local SHAP explanation layer** – computes SHAP values for every scored transaction, revealing how each feature increases or decreases the predicted risk relative to the model baseline. This exposes the top positive and negative drivers behind each decision.
- **Retrieval-Augmented explanations (FAISS RAG)** – builds a FAISS vector index from feature definitions and curated fraud-risk patterns. Retrieved context is injected into the LLM so explanations remain accurate, grounded, and domain-aligned instead of generic.
- **Streamlit interface** – provides a clean, user-friendly chat-style UI for inputs, model outputs, SHAP plots, and explanation text, with clear prompts or warnings when required data or model files are missing.

## Repository layout

```
├── app_chat_indicators.py              # Main Streamlit application (FraudLens UI, chat, SHAP, scoring)
├── requirements.txt         # Python dependencies needed to run the application
├── _model_/                 # Stores serialized EBM model artifacts (ebm_fraud_model.pkl)
├── _data_/_merge_/          # Contains the merged_data.csv feature table used for inference/explanations
├── .doc/                    # RAG knowledge sources
├── image/                   # Logos and supporting imagery referenced by the UI
└── launch_win.bat           # Helper scripts for launching Streamlit locally
```

## Getting started

### Prerequisites
- A valid OpenAI API key with access to the chosen chat-completion model
- EBM model artifact saved at `_model_/ebm_fraud_model.pkl`
- Merged transaction dataset saved at `_data_/_merge_/merged_data.csv`
- RAG knowledge saved at `.doc/fraud_knowledge_base_features.csv` and `.doc/fraud_knowledge_base_patterns.md`

### Installation
1. Create and activate a virtual environment (optional but recommended).
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file (or export shell variables) with your OpenAI key:
   ```bash
   echo "OPENAI_API_KEY=sk-..." >> .env
   ```

### Running the app
Start Streamlit and point it at the main application module:
```bash
streamlit run app_chat.py
```
The server will launch at `http://localhost:8501` by default. Make sure the expected `_model_`, `_data_/_merge_` and `.doc` artifacts exist before launching. Otherwise the interface will raise warnings and disable certain features.

## Development tips
- Update `requirements.txt` whenever new libraries are introduced.
- The SHAP explainer caches background samples via `@st.cache_resource`; restart Streamlit if you need to invalidate cached data after modifying the dataset.

## Data Source
- Kaggle open source datasets: https://www.kaggle.com/code/yashjadwani/financial-fraud-analysis/notebook.
- It includes `transactions_data.csv`, `users_data.csv`, `cards_data.csv`, `mcc_codes.json`, `train_fraud_labels.json`.

## License
This project has not specified a public license. Treat the contents as confidential unless instructed otherwise.
