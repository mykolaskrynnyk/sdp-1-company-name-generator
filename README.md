# sdp-1-company-name-generator
 A character-level RNN trained on WikiData to generate company names.

  ***

 ## Introduction

  Being the first one in the series of Side Data Projects (SDP), this applied project explores the application of deep learning to natural language modelling. In particular, I collect a sizeable training set of several hundred thousand company names from WikiData using SPARQL to build a character-level language model in TensorFlow. Using a "Long Short-Term Memory" (LSTM) architecture, I train a model capable of generating short (5-30 characters long) and realistic company names.

 ## The Structure

  The repository consists of three notebooks that guide through the whole training pipeline from data acquisition to model training as well as the scripts that contain model definition and helper functions.

  - `sdp-1-i-data-acquisition.ipynb` ([open in nbviewer](https://nbviewer.jupyter.org/github/mykolaskrynnyk/sdp-1-company-name-generator/blob/main/notebooks/sdp-1-i-data-acquisition.ipynb))
  - `sdp-1-ii-data-preparation.ipynb` ([open in nbviewer](https://nbviewer.jupyter.org/github/mykolaskrynnyk/sdp-1-company-name-generator/blob/main/notebooks/sdp-1-ii-data-preparation.ipynb))
  - `sdp-1-iii-language-modelling.ipynb` ([open in nbviewer](https://nbviewer.jupyter.org/github/mykolaskrynnyk/sdp-1-company-name-generator/blob/main/notebooks/sdp-1-iii-language-modelling.ipynb))

 ## The App

  The model is deployed online and can be accessed at streamlit [here](https://share.streamlit.io/mykolaskrynnyk/sdp-1-company-name-generator/main/app/streamlit_app.py).
