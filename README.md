# Science Experiment Simulator: Virtual Lab Experiments Guided by AI

Science Experiment Simulator is a Streamlit-based web application that provides virtual lab experiments guided by AI. This interactive tool uses various language models to simulate scientific experiments, offer explanations, and guide users through the scientific process across different fields of science.

## Features

- Interactive chat interface for conducting virtual lab experiments
- Support for multiple AI models, including OpenAI's GPT models and Ollama's local models
- Customizable science field and experiment type selection
- Dark/Light theme toggle
- Experiment saving and loading functionality
- Token usage tracking

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/science-experiment-simulator.git
   cd science-experiment-simulator
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```

4. (Optional) If you want to use Ollama models, make sure you have Ollama installed and running on your system.

## Usage

1. Run the Streamlit app:
   ```
   streamlit run science_experiment_simulator.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Enter your name, select the science field and experiment type, and start your virtual experiment!

## Customization

- You can modify the `SCIENCE_FIELDS` and `EXPERIMENT_TYPES` lists in the code to add or remove fields and types of experiments.
- The custom instructions for the AI can be adjusted in the sidebar of the application.

## Contributing

Contributions to improve the Science Experiment Simulator are welcome! Please feel free to submit pull requests or open issues to discuss potential enhancements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
