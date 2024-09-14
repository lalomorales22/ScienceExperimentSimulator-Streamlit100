import streamlit as st
import ollama
import time
import json
import os
from datetime import datetime
from openai import OpenAI

# List of available models
MODELS = [
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",  # OpenAI models
    "llama3.1:8b", "gemma2:2b", "mistral-nemo:latest", "phi3:latest",  # Ollama models
]

# Science fields and experiment types
SCIENCE_FIELDS = [
    "Physics", "Chemistry", "Biology", "Earth Science", "Astronomy"
]

EXPERIMENT_TYPES = [
    "Observation", "Measurement", "Hypothesis Testing", "Controlled Experiment",
    "Comparative Study", "Field Study", "Simulation"
]

def get_ai_response(messages, model):
    if model.startswith("gpt-"):
        return get_openai_response(messages, model)
    else:
        return get_ollama_response(messages, model)

def get_openai_response(messages, model):
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, 0, 0

def get_ollama_response(messages, model):
    try:
        response = ollama.chat(
            model=model,
            messages=messages
        )
        return response['message']['content'], response['prompt_eval_count'], response['eval_count']
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, 0, 0

def stream_response(messages, model):
    if model.startswith("gpt-"):
        return stream_openai_response(messages, model)
    else:
        return stream_ollama_response(messages, model)

def stream_openai_response(messages, model):
    client = OpenAI()
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        return stream
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def stream_ollama_response(messages, model):
    try:
        stream = ollama.chat(
            model=model,
            messages=messages,
            stream=True
        )
        return stream
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def save_experiment(messages, filename):
    experiment = {
        "timestamp": datetime.now().isoformat(),
        "messages": messages
    }
    
    os.makedirs('experiments', exist_ok=True)
    file_path = os.path.join('experiments', filename)
    
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                experiments = json.load(f)
        else:
            experiments = []
    except json.JSONDecodeError:
        experiments = []
    
    experiments.append(experiment)
    
    with open(file_path, 'w') as f:
        json.dump(experiments, f, indent=2)

def load_experiments(uploaded_file):
    if uploaded_file is not None:
        try:
            experiments = json.loads(uploaded_file.getvalue().decode("utf-8"))
            return experiments
        except json.JSONDecodeError:
            st.error(f"Error decoding the uploaded file. The file may be corrupted or not in JSON format.")
            return []
    else:
        st.warning("No file was uploaded.")
        return []

def main():
    st.set_page_config(layout="wide")
    st.title("Science Experiment Simulator: Virtual Lab Experiments Guided by AI")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "token_count" not in st.session_state:
        st.session_state.token_count = {"prompt": 0, "completion": 0}

    if "user_name" not in st.session_state:
        st.session_state.user_name = "Researcher"

    st.session_state.user_name = st.text_input("Enter your name:", value=st.session_state.user_name)

    st.sidebar.title("Experiment Settings")
    model = st.sidebar.selectbox("Choose a model", MODELS)

    custom_instructions = st.sidebar.text_area("Custom Instructions", 
        """You are an advanced Science Experiment Simulator AI. Your role is to guide users through virtual lab experiments, helping them understand scientific concepts, experimental design, and data analysis. You should provide detailed explanations, ask thought-provoking questions, and offer step-by-step guidance throughout the experimental process.

Your capabilities include:
1. Explaining scientific concepts and theories related to experiments
2. Guiding users through experimental setup and procedures
3. Simulating experimental outcomes based on user inputs
4. Assisting with data analysis and interpretation
5. Encouraging critical thinking and scientific reasoning
6. Providing safety information and best practices for lab work

When interacting:
- Adapt your explanations to the user's level of understanding
- Use clear, concise language while maintaining scientific accuracy
- Encourage users to form hypotheses and predict outcomes
- Guide users through the scientific method
- Offer suggestions for further experiments or areas of study
- Provide context on real-world applications of the experiments

Remember, your goal is to create an engaging and educational virtual lab experience, helping users develop their scientific skills and understanding across various fields of science.""")

    science_field = st.sidebar.selectbox("Choose science field", SCIENCE_FIELDS)
    experiment_type = st.sidebar.selectbox("Select experiment type", EXPERIMENT_TYPES)

    theme = st.sidebar.selectbox("Choose a theme", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown("""
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    if st.sidebar.button("Start New Experiment"):
        st.session_state.messages = []
        st.session_state.token_count = {"prompt": 0, "completion": 0}

    st.sidebar.subheader("Experiment Management")
    save_name = st.sidebar.text_input("Save experiment as:", f"{science_field.lower()}_{experiment_type.lower()}_experiment.json")
    if st.sidebar.button("Save Experiment"):
        save_experiment(st.session_state.messages, save_name)
        st.sidebar.success(f"Experiment saved to experiments/{save_name}")

    st.sidebar.subheader("Load Experiment")
    uploaded_file = st.sidebar.file_uploader("Choose a file to load experiments", type=["json"], key="experiment_uploader")
    
    if uploaded_file is not None:
        try:
            experiments = load_experiments(uploaded_file)
            if experiments:
                st.sidebar.success(f"Loaded {len(experiments)} experiments from the uploaded file")
                selected_experiment = st.sidebar.selectbox(
                    "Select an experiment to load",
                    range(len(experiments)),
                    format_func=lambda i: experiments[i]['timestamp']
                )
                if st.sidebar.button("Load Selected Experiment"):
                    st.session_state.messages = experiments[selected_experiment]['messages']
                    st.sidebar.success("Experiment loaded successfully!")
            else:
                st.sidebar.error("No valid experiments found in the uploaded file.")
        except Exception as e:
            st.sidebar.error(f"Error loading experiments: {str(e)}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your experiment step or question:"):
        st.session_state.messages.append({"role": "user", "content": f"{st.session_state.user_name}: {prompt}"})
        with st.chat_message("user"):
            st.markdown(f"{st.session_state.user_name}: {prompt}")

        field_instruction = f"Focus on {science_field} experiments. "
        type_instruction = f"Guide the {experiment_type} process. "
        ai_messages = [
            {"role": "system", "content": custom_instructions + field_instruction + type_instruction},
            {"role": "system", "content": "Remember to guide the user through the scientific method, encourage critical thinking, and provide detailed explanations of scientific concepts."},
        ] + st.session_state.messages

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in stream_response(ai_messages, model):
                if chunk:
                    if model.startswith("gpt-"):
                        full_response += chunk.choices[0].delta.content or ""
                    else:
                        full_response += chunk['message']['content']
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.05)
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        _, prompt_tokens, completion_tokens = get_ai_response(ai_messages, model)
        st.session_state.token_count["prompt"] += prompt_tokens
        st.session_state.token_count["completion"] += completion_tokens

    st.sidebar.subheader("Token Usage")
    st.sidebar.write(f"Prompt tokens: {st.session_state.token_count['prompt']}")
    st.sidebar.write(f"Completion tokens: {st.session_state.token_count['completion']}")
    st.sidebar.write(f"Total tokens: {sum(st.session_state.token_count.values())}")

if __name__ == "__main__":
    main()
