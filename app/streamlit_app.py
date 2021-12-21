# standard library
import sys, os

# if executed on Streamlit
if 'models' in os.listdir('./'):
    sys.path.append('./src')
    model_dir = './models'

# if run locally
else:
    sys.path.append('../src')
    model_dir = '../models'

# database
from pymongo import MongoClient

# deep learning
import tensorflow as tf

# streamlit
import streamlit as st

# local packages
from myutils import generate

# other settings
REGION = 'Default'
RNN_TYPE = 'lstm'

# page config
st.set_page_config(
     page_title = 'SDP-1 Company Name Generator',
     page_icon = '🧊',
     layout = 'centered',
     initial_sidebar_state = 'auto',
     menu_items = {
         'Get help': None,
         'Report a Bug': None,
         'About': 'https://www.github.com/mykolaskrynnyk',
     }
 )

# functions and callbacks
@st.cache
def load_model():
    model_name = f'company-generator-{RNN_TYPE.lower()}-{REGION.lower()}'
    model = tf.saved_model.load(os.path.join(model_dir, model_name))
    return model

if 'labels' not in st.session_state:
    client = MongoClient(os.environ.get('MONGODB_CONN'))
    db = client['sdp-1']
    collection = db['labels-v21-12-20']
    cursor = collection.find(filter = {'region': 'Default'}, projection = {'label': 1})
    st.session_state['labels'] = set([x['label'] for x in cursor])
    client.close()

def generate_name() -> str:
    """
    Generates a full company name using the RNN model.
    If Exclude actual names is selected, generates up to 10 names
    stopping if a novel name is found.
    """
    for _ in range(10):
        prediction = generate(model, st.session_state.seed)
        prediction = prediction[1: -1]

        if not st.session_state.exclude:
            break

        if prediction not in st.session_state.labels:
            break

    else:
        prediction = 'Try another one...'

    return prediction

model = load_model()

# app definition
st.title('SDP-1: Company Name Generator')
st.header('IV. Model Deployment')
st.markdown('''
_This application allows users to interact with a Recurrent Neural Network
trained on several hundred thousand company names from WikiData.
The model can be used to generate novel company names either completely from scratch
or by using a seed sequence. Ticking the Exclude actual names checkbox will make sure
that a generated name is novel, i.e. it does not appear in the training set.
This, of course, does not guarantee that the name is not being used by
some actual company. Find more details about the project in
[SDP-1: Company Name Generator](https://www.github.com/mykolaskrynnyk/sdp-1-company-name-generator),
a dedicated repository on GitHub._
''')

with st.expander('About the model'):
     st.markdown('''
         ### The Data

         I train the model on 280k+ examples of company names from WikiData. Each name consists of up to three words and
         is between 5 and 30 characters long. Names are allowed to contain only ASCII characters, numbers, commas, dots, dashes and ampersands.

         ### The Architecture

         The model consists of a 256-dimensional `Embedding` layer, an `LSTM` layer with 1024 hidden units and a Dense output layer.
         Overall, the model has more than 5 million parameters. I use `SparseCategoricalCrossentropy` as the loss function and `Adam`
         optimised with the default learning rate (`.001`). The model is trained using `TensorFlow` (`2.6.0`).
     ''')

col1, col2 = st.columns(2)

with col1:
    st.subheader('User Input')
    st.text_input('Enter a short seed sequence or leave the field empty', key = 'seed')
    st.checkbox('Exclude actual names', key = 'exclude')
    st.button('Generate!', key = 'generate', on_click = generate_name)

with col2:
    st.subheader('Model Output')
    st.write(f'### Seed: {st.session_state.seed if st.session_state.seed else "<None>"}')
    st.write(f'### Prediction: {generate_name()}')
