🌱 Renewable Energy Simulation App
This Streamlit app is designed to simulate and predict key performance indicators (KPIs) for a renewable energy power plant. It uses machine learning to forecast metrics such as grid draw, energy output, consumption, costs, hydrogen production, and CO₂ captured.

🚀 Features
Real-time predictions using trained machine learning models
Interactive visualizations and dashboards
Downloadable results and prediction logs
SHAP-based model explainability (optional)
Simple and intuitive UI with Streamlit
⚙️ Technologies Used
Python 3.10+
Streamlit
Pandas, NumPy
Scikit-learn
Matplotlib / Seaborn
SHAP, XGBoost (if included)
Joblib for model persistence
📁 Project Structure
.
├── streamlit_app.py        # Main Streamlit app
├── requirements.txt        # Dependencies for Streamlit Cloud
└── README.md               # You're reading it!
🧪 How to Run Locally
# Clone the repository
git clone https://github.com/YOUR_USERNAME/renewable-energy-app.git
cd renewable-energy-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
🌐 Deploy on Streamlit Cloud
Push this repository to GitHub.
Go to https://streamlit.io/cloud.
Click "New App" and select this repo.
Set streamlit_app.py as the main file.
Deploy and share your app!
