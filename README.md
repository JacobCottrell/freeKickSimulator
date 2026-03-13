# Free Kick Simulator Web App

This is a Streamlit web app version of your original `freekickv3.py` simulator.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free

### Streamlit Community Cloud
1. Create a new GitHub repo.
2. Upload `app.py` and `requirements.txt`.
3. Go to Streamlit Community Cloud.
4. Create a new app from your GitHub repo.
5. Set the main file path to `app.py`.

### Hugging Face Spaces
1. Create a new Space.
2. Choose **Streamlit**.
3. Upload `app.py` and `requirements.txt`.
4. The Space will build automatically and give you a public link.

## Notes
- This version keeps the original physics model.
- It replaces desktop Matplotlib widgets with browser sliders and Plotly charts.
- The original animation is simplified into an interactive 3D plot for web compatibility.
