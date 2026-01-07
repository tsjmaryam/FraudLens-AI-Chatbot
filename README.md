<p align="center">
  <img src="https://raw.githubusercontent.com/tsjmaryam/FraudLens-AI-Chatbot/main/logo.png" alt="FraudLens Logo" width="260">
</p>

<p align="center">
  © 2025 Maryam Shahbaz Ali, Pingyi Xu, Tyler Ordiway, Zihui Chen  
  <br>
  Released under the MIT License.
</p>

---
# FraudLens - From Alerts to Insights: Responsible AI for Fraud Detection

FraudLens is an interpretable fraud-analysis assistant designed to help analysts understand why a transaction is flagged. They system combines a surrogate Explainable Boosting Machine (EBM), SHAP explanations, and a TF-IDF based RAG knowledge layer to turn complex model behavior and rule-based fraud signals into clear, human-readable narratives. It is built for transparency, regulatory alignment, and rapid analyst review in modern fraud-detection workflows.

## Key features

- **LLM-powered interaction & explanation** – routes natural-language questions through OpenAI's `gpt-3.5-turbo` (or any compatible model) via the `OPENAI_API_KEY` environment variable, allowing investigators to request justifications, metrics, or transformations and receive grounded narrative explanations that blend EBM scores, SHAP contributors, and retrieved domain definitions.
- **Explainable ML scoring** – loads an EBM model artifact from `_model_/ebm_fraud_model.pkl` to approximate internal rule-based fraud logic. The surrogate model produces stable, interpretable feature-level risk contributions and transparent per-transaction scores.
- **Local SHAP explanation layer** – computes SHAP values for every scored transaction, revealing how each feature increases or decreases the predicted risk relative to the model baseline. This exposes the top positive and negative drivers behind each decision.
- **Retrieval-Augmented explanations (TF-IDF RAG)** – builds a TF-IDF vector from feature definitions and curated fraud-risk patterns. Retrieved context is injected into the LLM so explanations remain accurate, grounded, and domain-aligned instead of generic.
- **Streamlit interface** – provides a clean, user-friendly chat-style UI for inputs, model outputs, SHAP plots, and explanation text, with clear prompts or warnings when required data or model files are missing.

## Repository layout

```FraudLens-AI-Chatbot/
│
├── .doc/                                # RAG knowledge base (features + patterns)
│   ├── fraud_knowledge_base_features.csv
│   └── fraud_knowledge_base_patterns.md
│
├── Project_readme/                      # Extended project documentation
│   └── README.md
│
├── _model_/                             # Trained EBM model artifacts
│   └── EBM_fraud_model.pkl
│
├── evaluation_graphs/                   # Model evaluation visuals
│   ├── Calibration curve.png
│   ├── ROC curve.png
│   ├── confusion matrix.png
│   ├── percision recall curve.png
│   └── README.md
│
├── fraud_plots/                         # EDA visualizations
│   ├── README.md
│   ├── cardtype.png
│   ├── chipusage.png
│   ├── clientnetwork.png
│   ├── dayoftheweek.png
│   ├── fraudarte.png
│   ├── fraudrateby_darkweb.png
│   ├── houroftheday.png
│   ├── merchanype.png
│   ├── Detect unusual merchants or transactions.png
│   └── Detect whether the transaction ...png
│
├── full_dataset/                        # All dataset files used for FraudLens
│   ├── README.md
│   ├── cards_data.csv
│   ├── df_merged_small.csv
│   ├── transactions_data.csv
│   └── users_data.csv
│
├── image/                               # UI and logo images
│   ├── 1.png
│   ├── GWSB Short White.png
│   └── logo option2.png
│
├── requirements/                        # Requirements explanation document
│   └── README.md
│
├── Youtube_channel/                     # Demo videos, YouTube links, media assets
│   └── README.md                        # Description + links to video demos
│
│── .gitattributes                       # Git LFS tracking
│── Fraud_EBM_Final.ipynb                # Jupyter Notebook for training the EBM model
│── LICENSE                              # MIT License
│── README.md                            # Main repository README (homepage)
│── app_chat.py                          # Main Streamlit Chatbot UI
│── app_chat_indicators.py               # Indicators-based version of Chatbot
│── app_chat_indicators_decision.py      # Most updated version of our Chatbot
│── app_chat_updated.py                  # Updated chatbot with behavioral narrative
│── fraudlens.py                         # Fraud scoring and logic helpers
│── launch_win.bat                       # Windows launcher for the app
│── logo.png                             # Project logo and GWSB logo
│── requirements.txt                     # Python dependencies (actual package list)
│
├── Architectural_Diagram/               # System's architectural diagram
│   ├── Architectural Diagram.png        # Final architecture visual
│   └── README.md                        # Explanation of the diagram
│
└── poster_and_slides/                   # Poster and presentation materials
    ├── README.md                        # Description of poster + slides
    ├── Poster.png                       # Poster image
    ├── Poster.pdf                       # Final poster
    ├── Draft Presentation Deck.pptx     # Draft deck
    └── Final Presentation Deck.pdf      # Final deck
```
## Getting started

### Prerequisites
- A valid OpenAI API key with access to the chosen chat-completion model
- EBM model artifact saved at `_model_/EBM_fraud_model.pkl`
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
streamlit run app_chat_indicators_decision.py
```
The server will launch at `http://localhost:8501` by default. Make sure the expected `_model_`, `_data_/_merge_` and `.doc` artifacts exist before launching. Otherwise the interface will raise warnings and disable certain features.

## Development tips
- Update `requirements.txt` whenever new libraries are introduced.
- The SHAP explainer caches background samples via `@st.cache_resource`; restart Streamlit if you need to invalidate cached data after modifying the dataset.

## Data source
- Kaggle open source datasets: https://www.kaggle.com/code/yashjadwani/financial-fraud-analysis/notebook.
- It includes `transactions_data.csv`, `users_data.csv`, `cards_data.csv`, `mcc_codes.json`, `train_fraud_labels.json`.
  
## Acknowledgments
We gratefully acknowledge the guidance of our industry judges and advisors, whose insights shaped the direction and grounding of this project. We thank Maysam Kashipour (Capital One) for helping us narrow our problem framing and design a realistic, analyst-focused solution; Brian Clark and Benjamin Pickett (EY) for their clarity on value, impact, and organizational trust; Matthew Finan (Mizuho Financial Group) for deep feedback on explainability, behavioral patterns, and workflow design; Dillon A. De Benedetto (SMBC) for sharing practical fraud-analysis expertise and real-world investigative patterns; and Steve Deering and Jerome Prieur (FI Consulting) for their emphasis on data quality, simplicity, and user-centered design.

We also express our sincere appreciation to Professor Patrick Hall, our faculty mentor, for his continuous support in responsible AI, model transparency, and rigorous evaluation. His guidance played a foundational role in shaping the principles behind FraudLens.

Their collective expertise made this project stronger, clearer, and more aligned with real analyst needs.

## License
This project is licensed under the MIT License.  
See the [LICENSE](./LICENSE) file for details.
