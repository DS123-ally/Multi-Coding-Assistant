import os
import json
import subprocess
import tempfile
import re
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Setup OpenAI client (configurable for local Gemma via Ollama or Groq/Google)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    base_url=os.getenv("OPENAI_BASE_URL")
)
MODEL = os.getenv("MODEL_NAME", "gemma")

def call_llm(system_prompt: str, user_prompt: str, token_callback=None) -> str:
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            stream=bool(token_callback)
        )
        if token_callback:
            full_content = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    token_callback(content)
            return full_content
        else:
            return response.choices[0].message.content
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

def extract_python_code(text: str) -> str:
    """Helper to extract python code block from markdown."""
    pattern = r"```(?:python|py)?(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()

class Orchestrator:
    def __init__(self):
        self.system_prompt = """You are the Orchestrator agent. Your job is to break down the user's coding request and coordinate the Planner, Coder, and Reviewer.
You must output a JSON object containing the plan and delegating the task to the Planner.
Example output:
{
    "analysis": "The user wants to build a simple snake game.",
    "next_agent": "Planner",
    "instructions_for_agent": "Create a detailed step-by-step architecture for a Python snake game."
}"""

    def run(self, user_request: str, token_callback=None):
        return call_llm(self.system_prompt, f"User Request: {user_request}\n\nPlease analyze and delegate to the Planner.", token_callback)

class Planner:
    def __init__(self):
        self.system_prompt = """You are the Planner agent. Your job is to create a detailed technical plan for the given task.
Break the task down into clear implementation steps, specifying the files to create, architecture, and libraries.
Return ONLY the technical plan in markdown format."""

    def run(self, instructions: str, token_callback=None):
        return call_llm(self.system_prompt, instructions, token_callback)

class Coder:
    def __init__(self):
        self.system_prompt = """You are the Coder agent. Your job is to write the actual code based on the Planner's plan.
Provide clean, modular, and well-commented code. Output the code block(s) clearly.
CRITICAL: You are writing Python. Do NOT use C-style comments (/* */). Use ONLY Python comments (#). Ensure it is 100% valid Python.
IMPORTANT: The code will be tested in an automated sandbox. DO NOT use `input()`, infinite loops, or blocking GUI calls (like `plt.show()` or `mainloop()`). The code must execute and terminate automatically."""

    def run(self, plan: str, token_callback=None):
        return call_llm(self.system_prompt, f"Here is the plan:\n{plan}\n\nPlease implement this code.", token_callback)

class Reviewer:
    def __init__(self):
        self.system_prompt = """You are the Reviewer agent. Your job is to review the code.
If there are errors from the Tester, you MUST fix them.
Look for bugs, clean code practices, and security issues. 
Return the final code in a markdown code block.
CRITICAL: You are writing Python. Do NOT use C-style comments (/* */). Use ONLY Python comments (#). Ensure it is 100% valid Python syntax.
IMPORTANT: The code will be tested in an automated sandbox. DO NOT use `input()`, infinite loops, or blocking GUI calls (like `plt.show()` or `mainloop()`). The code must execute and terminate automatically."""

    def run(self, code: str, error_log: str = None, token_callback=None):
        prompt = f"Please review and finalize the following code:\n{code}"
        if error_log:
            prompt += f"\n\nCRITICAL: The code failed when tested. Fix these errors:\n{error_log}"
        return call_llm(self.system_prompt, prompt, token_callback)

class Tester:
    def __init__(self):
        pass

    def run(self, raw_markdown_code: str):
        """Runs the python code and returns (success_bool, console_output)"""
        code = extract_python_code(raw_markdown_code)
        
        # Don't try to run empty code
        if not code or len(code) < 5:
            return False, "Error: No Python code found to test."

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w", encoding="utf-8") as temp_file:
            temp_file.write(code)
            temp_path = temp_file.name

        try:
            # Run the python script
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=10 # 10 seconds timeout
            )
            success = result.returncode == 0
            
            output = f"--- STDOUT ---\n{result.stdout}\n"
            if result.stderr:
                output += f"--- STDERR ---\n{result.stderr}\n"
                
            return success, output
        except subprocess.TimeoutExpired:
            return False, "Error: Code execution timed out (infinite loop or long process)."
        except Exception as e:
            return False, f"System Error executing code: {str(e)}"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

def run_workflow(user_request: str, status_callback=None, token_callback=None):
    # This function orchestrates the whole flow
    orchestrator = Orchestrator()
    planner = Planner()
    coder = Coder()
    reviewer = Reviewer()
    tester = Tester()
    
    if status_callback: status_callback("Orchestrator", "Analyzing the user request...")
    orch_response = orchestrator.run(user_request, token_callback)
    
    # Simple fallback parsing
    orch_data = {}
    try:
        orch_str = orch_response
        if "```json" in orch_str:
            orch_str = orch_str.split("```json")[1].split("```")[0]
        elif "```" in orch_str:
            orch_str = orch_str.split("```")[1].split("```")[0]
        orch_data = json.loads(orch_str.strip())
    except:
        orch_data = {
            "analysis": "Could not parse JSON.",
            "next_agent": "Planner",
            "instructions_for_agent": orch_response
        }

    if status_callback: status_callback("Planner", "Creating technical plan...")
    plan = planner.run(orch_data.get("instructions_for_agent", user_request), token_callback)
    
    if status_callback: status_callback("Coder", "Writing the code...")
    code = coder.run(plan, token_callback)
    
    # Reviewer and Tester Feedback Loop
    max_retries = 2
    attempts = 0
    final_code = code
    console_output = "Code not tested."
    
    while attempts <= max_retries:
        if status_callback: status_callback("Reviewer", f"Reviewing code (Attempt {attempts + 1})...")
        final_code = reviewer.run(final_code, error_log=console_output if attempts > 0 else None, token_callback=token_callback)
        
        if status_callback: status_callback("Tester", f"Running code in sandbox...")
        success, console_output = tester.run(final_code)
        
        if success:
            if status_callback: status_callback("Tester", f"Code passed all tests!")
            break
        else:
            if status_callback: status_callback("Tester", f"Code failed. Sending error back to Reviewer...")
            attempts += 1

    return {
        "orchestrator_analysis": orch_data,
        "plan": plan,
        "code": code,
        "final_code": final_code,
        "console_output": console_output,
        "success": attempts <= max_retries
    }
