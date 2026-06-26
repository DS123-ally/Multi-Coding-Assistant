import streamlit as st
from agents import run_workflow
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Gemma 4 Multi-Agent Assistant", page_icon="🤖", layout="wide")

st.title("🤖 Gemma 4 Multi-Agent Coding Assistant")
st.markdown("A Streamlit app where an Orchestrator coordinates Planner, Coder, and Reviewer agents.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    st.info("Ensure you have set up your API keys in the `.env` file or here.")
    api_key = st.text_input("OpenAI-compatible API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    base_url = st.text_input("API Base URL (for local models like Ollama)", value=os.getenv("OPENAI_BASE_URL", ""))
    model_name = st.text_input("Model Name", value=os.getenv("MODEL_NAME", "gemma"))
    
    if st.button("Save Config"):
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["MODEL_NAME"] = model_name
        st.success("Configuration updated!")

# Main chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Enter your coding task (e.g., 'Build a simple web scraper in Python')...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        st.markdown("### Agent Workflow 🚀")
        
        # We will use an expander to show the agent thinking process
        status_placeholder = st.empty()
        
        # Callback to update the UI
        def ui_callback(agent_name, status):
            status_placeholder.info(f"**{agent_name}:** {status}")
            
        with st.spinner("Processing..."):
            results = run_workflow(user_input, callback=ui_callback)
            
        status_placeholder.success("Workflow Complete!")
        
        # Display results in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Final Output", "Console Output", "Orchestrator Analysis", "Plan", "Raw Code"])
        
        with tab1:
            st.markdown("#### Final Reviewed Code")
            if not results.get("success", True):
                st.warning("⚠️ The Tester Agent could not get the code to run successfully after maximum retries.")
            st.markdown(results.get("final_code", "No final code generated."))
            
        with tab2:
            st.markdown("#### Tester Sandbox Console")
            if results.get("success"):
                st.success("Code executed successfully!")
            else:
                st.error("Code execution failed.")
            st.code(results.get("console_output", ""), language="text")

        with tab3:
            st.json(results.get("orchestrator_analysis", {}))
            
        with tab4:
            st.markdown(results.get("plan", "No plan generated."))
            
        with tab5:
            st.markdown("#### Code before review")
            st.markdown(results.get("code", "No code generated."))
            
        # Add assistant response to chat history
        assistant_reply = f"Here is the final result for your task:\n\n{results.get('final_code', '')}"
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
