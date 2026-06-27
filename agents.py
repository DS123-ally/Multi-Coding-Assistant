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

def get_workspace_context(directory: str) -> str:
    if not os.path.exists(directory):
        return "Workspace is currently empty."
    
    context = ""
    for root, dirs, files in os.walk(directory):
        # Ignore hidden and venv dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('venv', '__pycache__')]
        for file in files:
            if not file.startswith('.'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, directory)
                context += f"\n--- {rel_path} ---\n"
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if len(content) > 5000:
                            content = content[:5000] + "\n...[TRUNCATED]..."
                        context += content + "\n"
                except Exception:
                    context += "[Could not read file]\n"
    return context.strip() if context else "Workspace is currently empty."


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

def extract_files(text: str) -> list:
    """Extracts a list of (filepath, code) from markdown text."""
    pattern = r"```(?:python|py)?(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    files = []
    
    for match in matches:
        code = match.strip()
        filepath = "main.py" # default fallback
        
        # Look for # FILE: path/to/file.py at the start of the code
        lines = code.split("\n")
        if lines and lines[0].strip().startswith("# FILE:"):
            filepath = lines[0].split("FILE:", 1)[1].strip()
            code = "\n".join(lines[1:]).strip()
            
        if code:
            files.append((filepath, code))
            
    return files

def save_files_to_disk(raw_markdown_code: str, workspace_dir: str):
    files = extract_files(raw_markdown_code)
    saved_paths = []
    
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)
        
    for filepath, code in files:
        # Prevent directory traversal
        safe_path = os.path.normpath(filepath)
        if safe_path.startswith("..") or os.path.isabs(safe_path):
            safe_path = os.path.basename(filepath)
            
        full_path = os.path.join(workspace_dir, safe_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(code)
        saved_paths.append(safe_path)
        
    return saved_paths

class Orchestrator:
    def __init__(self):
        self.system_prompt = """You are the Orchestrator agent. Your job is to break down the user's coding request and coordinate the Planner, Coder, and Reviewer.
You will be provided with the current context of the user's workspace. Analyze the workspace and the user's request.
You must output a JSON object containing the plan and delegating the task to the Planner.
Example output:
{
    "analysis": "The user wants to refactor main.py to add a new function.",
    "next_agent": "Planner",
    "instructions_for_agent": "Create a detailed step-by-step architecture to refactor main.py."
}"""

    def run(self, user_request: str, workspace_context: str, token_callback=None):
        prompt = f"Workspace Context:\n<workspace_context>\n{workspace_context}\n</workspace_context>\n\nUser Request: {user_request}\n\nPlease analyze and delegate to the Planner."
        return call_llm(self.system_prompt, prompt, token_callback)

class Planner:
    def __init__(self):
        self.system_prompt = """You are the Planner agent. Your job is to create a detailed technical plan for the given task.
Break the task down into clear implementation steps, specifying the exact files to create or modify, architecture, and libraries.
Return ONLY the technical plan in markdown format."""

    def run(self, instructions: str, token_callback=None):
        return call_llm(self.system_prompt, instructions, token_callback)

class Coder:
    def __init__(self):
        self.system_prompt = """You are the Coder agent. Your job is to write the actual code based on the Planner's plan.
Provide clean, modular, and well-commented code. Output the code block(s) clearly.
CRITICAL: You are writing Python. Do NOT use C-style comments (/* */). Use ONLY Python comments (#). Ensure it is 100% valid Python.
IMPORTANT: The code will be tested in an automated sandbox. DO NOT use `input()`, infinite loops, or blocking GUI calls. The code must execute and terminate automatically.
EXTREMELY CRITICAL: For every code block you write, the very first line inside the python code block MUST be a comment specifying the filepath, exactly like this:
# FILE: filename.py"""

    def run(self, plan: str, token_callback=None):
        return call_llm(self.system_prompt, f"Here is the plan:\n{plan}\n\nPlease implement this code.", token_callback)

class Reviewer:
    def __init__(self):
        self.system_prompt = """You are the Reviewer agent. Your job is to review the code.
If there are errors from the Tester, you MUST fix them.
Look for bugs, clean code practices, and security issues. 
Return the final code in a markdown code block.
CRITICAL: You are writing Python. Do NOT use C-style comments (/* */). Use ONLY Python comments (#). Ensure it is 100% valid Python syntax.
IMPORTANT: The code will be tested in an automated sandbox. DO NOT use `input()`, infinite loops, or blocking GUI calls. The code must execute and terminate automatically.
EXTREMELY CRITICAL: For every code block you write, the very first line inside the python code block MUST be a comment specifying the filepath, exactly like this:
# FILE: filename.py"""

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
        files = extract_files(raw_markdown_code)
        if not files:
            return False, "Error: No Python code blocks found to test."

        with tempfile.TemporaryDirectory() as temp_dir:
            entrypoint = None
            for filepath, code in files:
                safe_path = os.path.basename(filepath)
                if not entrypoint:
                    entrypoint = safe_path
                full_path = os.path.join(temp_dir, safe_path)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(code)

            if not entrypoint:
                return False, "Error: Could not determine entrypoint."

            try:
                result = subprocess.run(
                    [sys.executable, os.path.join(temp_dir, entrypoint)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=temp_dir
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

def run_workflow(user_request: str, workspace_dir: str, status_callback=None, token_callback=None):
    orchestrator = Orchestrator()
    planner = Planner()
    coder = Coder()
    reviewer = Reviewer()
    tester = Tester()
    
    if status_callback: status_callback("Orchestrator", "Analyzing workspace and request...")
    workspace_context = get_workspace_context(workspace_dir)
    orch_response = orchestrator.run(user_request, workspace_context, token_callback)
    
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

    # Save to disk if successful
    saved_files = []
    if attempts <= max_retries:
        if status_callback: status_callback("File Manager", "Saving files to workspace...")
        saved_files = save_files_to_disk(final_code, workspace_dir)

    return {
        "orchestrator_analysis": orch_data,
        "plan": plan,
        "code": code,
        "final_code": final_code,
        "console_output": console_output,
        "success": attempts <= max_retries,
        "saved_files": saved_files
    }
