# Install Required Libraries
!pip install transformers gradio accelerate huggingface_hub

# Import Libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login
import gradio as gr

# Login to Hugging Face
login()

# IBM Granite Model ID
model_id = "ibm-granite/granite-3.3-2b-instruct"

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load AI Model
model = AutoModelForCausalLM.from_pretrained(model_id)

# Create Pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Disease Prediction Function
def predict_disease(symptoms):

    prompt = f"""
You are an AI Healthcare Assistant.

A patient has the following symptoms:
{symptoms}

Predict the possible disease and suggest general precautions.
Mention that this is not a medical diagnosis and advise consulting a doctor.
"""

    response = pipe(
        prompt,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    return response[0]["generated_text"]


# Home Remedy Function
def home_remedy(disease):

    prompt = f"""
You are an AI Healthcare Assistant.

A user is suffering from:
{disease}

Suggest some general home remedies.
Also mention that if symptoms become severe, the user should consult a doctor.
"""

    response = pipe(
        prompt,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    return response[0]["generated_text"]


# Gradio User Interface
with gr.Blocks(
    title="AI Healthcare Assistant",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("""
# 🩺 AI Healthcare Assistant

### Predict diseases based on symptoms and get general home remedies using IBM Granite AI.

⚠️ Disclaimer:
This application is for educational purposes only.
It is not a substitute for professional medical advice.
""")

    with gr.Tab("🩺 Disease Prediction"):

        symptoms = gr.Textbox(
            label="Enter Your Symptoms",
            placeholder="Example: Fever, Cough, Headache",
            lines=4
        )

        disease_output = gr.Textbox(
            label="AI Prediction",
            lines=12
        )

        predict_btn = gr.Button("🔍 Predict Disease")

        predict_btn.click(
            fn=predict_disease,
            inputs=symptoms,
            outputs=disease_output
        )

    with gr.Tab("🏠 Home Remedies"):

        disease = gr.Textbox(
            label="Enter Disease Name",
            placeholder="Example: Common Cold",
            lines=2
        )

        remedy_output = gr.Textbox(
            label="Suggested Home Remedies",
            lines=12
        )

        remedy_btn = gr.Button("🌿 Get Home Remedies")

        remedy_btn.click(
            fn=home_remedy,
            inputs=disease,
            outputs=remedy_output
        )

    gr.Markdown("""
---
### 👨‍💻 Developed By

Prasanna

### Technologies Used

- Python
- IBM Granite AI
- Hugging Face
- Transformers
- Gradio
""")

# Launch Application
demo.launch()
