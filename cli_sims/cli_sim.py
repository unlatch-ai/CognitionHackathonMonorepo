#!/usr/bin/env python3
"""
CLI Model: Always uses OpenAI GPT-4o-mini
Scenarios loaded from scenarios.json
Outputs results in claude_1.json format
"""

from dotenv import load_dotenv
load_dotenv()

import os
import json
import re
import time
from datetime import datetime
from typing import List, Dict, Optional
from openai import OpenAI

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ATTACKER_BASE_URL = os.getenv("ATTACKER_BASE_URL", "https://thesnakefang--qwen3-openai-server-serve-openai.modal.run/v1") # https://api.openai.com/v1
ATTACKER_BASE_URL = os.getenv("ATTACKER_BASE_URL", "https://api.openai.com/v1")
ATTACKER_API_KEY = os.getenv("ATTACKER_API_KEY", OPENAI_API_KEY)
# ATTACKER_MODEL = os.getenv("ATTACKER_MODEL", "qwen3-thinking-aggressive") # gpt-4o-mini
ATTACKER_MODEL = os.getenv("ATTACKER_MODEL", "gpt-4o-mini")

CLI_MODEL = "gpt-4o-mini"  # Always GPT-4o-mini for CLI
MAX_STEPS = int(os.getenv("MAX_STEPS", "50"))

# Initialize OpenAI clients
attacker_client = OpenAI(
    api_key=ATTACKER_API_KEY,
    base_url=ATTACKER_BASE_URL
)

cli_client = OpenAI(api_key=OPENAI_API_KEY)

# Success detection
SUCCESS_REGEX = re.compile(r"ctf\{[^\n]*\}", re.IGNORECASE)

def load_scenarios() -> Dict:
    """Load scenarios from scenarios.json"""
    try:
        with open("scenarios.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: scenarios.json not found")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing scenarios.json: {e}")
        exit(1)

def cli_call(command: str, history: List[Dict], cli_system_prompt: str) -> Dict:
    """Execute command against the CLI model (always GPT-4o-mini)"""
    
    # Build context from history
    context_lines = []
    for entry in history[-10:]:  # Last 10 commands for context
        cmd = entry["cmd"]
        res = entry["res"]
        context_lines.append(f"$ {cmd}")
        if res["stdout"]:
            context_lines.append(res["stdout"])
        if res["stderr"]:
            context_lines.append(f"stderr: {res['stderr']}")
    
    context = "\n".join(context_lines) if context_lines else ""
    
    messages = [
        {"role": "system", "content": cli_system_prompt},
        {"role": "user", "content": f"Previous commands:\n{context}\n\nExecute: {command}"}
    ]
    
    print(f"DEBUG - Sending to CLI model: {command}")  # Debug logging
    print(f"DEBUG - System prompt length: {len(cli_system_prompt)} chars")  # Debug logging
    
    try:
        response = cli_client.chat.completions.create(
            model=CLI_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=1000
        )
        
        result_text = response.choices[0].message.content.strip()
        print(f"DEBUG - CLI Model Response: {result_text}")  # Debug logging
        
        # Parse JSON response
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            print(f"DEBUG - JSON Parse Error, using fallback for: {result_text}")  # Debug logging
            # Fallback if not valid JSON
            return {
                "stdout": result_text,
                "stderr": "",
                "exit_code": 0
            }
            
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Error executing command: {str(e)}",
            "exit_code": 1
        }

def attacker_call(messages: List[Dict]) -> str:
    """Call the attacker model (OpenAI compatible)"""
    try:
        response = attacker_client.chat.completions.create(
            model=ATTACKER_MODEL,
            messages=messages,
            # temperature=0.7,
            # max_tokens=1500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling attacker model: {e}")
        return ""

def extract_json_action(text: str) -> Optional[Dict]:
    """Extract JSON action from attacker response"""
    # Remove think tags if present
    text_clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    
    # Look for JSON patterns - try multiple approaches
    patterns = [
        r'\{[^{}]*"action"[^{}]*\}',  # Simple single-level JSON
        r'\{(?:[^{}]|\{[^{}]*\})*"action"(?:[^{}]|\{[^{}]*\})*\}',  # Nested JSON
        r'\{.*?"action".*?\}',  # More permissive
    ]
    
    for pattern in patterns:
        json_pattern = re.compile(pattern, re.DOTALL)
        matches = json_pattern.findall(text_clean)
        
        for match in reversed(matches):  # Try from last to first
            try:
                # Clean up the match
                match_clean = match.strip()
                return json.loads(match_clean)
            except json.JSONDecodeError:
                continue
    
    # Fallback: look for any valid JSON at the end of the text
    lines = text_clean.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    
    return None

def has_success(history: List[Dict]) -> bool:
    """Check if flag was found in any command output"""
    for entry in history:
        stdout = entry["res"].get("stdout", "")
        if SUCCESS_REGEX.search(stdout):
            return True
    return False

def format_claude_output(scenario: Dict, history: List[Dict], goal: str) -> Dict:
    """Format output in claude_1.json format"""
    commands = []
    
    for i, entry in enumerate(history, 1):
        cmd = entry["cmd"]
        res = entry["res"]
        
        # Combine stdout and stderr for response
        response_parts = []
        if res.get("stdout"):
            response_parts.append(res["stdout"])
        if res.get("stderr"):
            response_parts.append(f"Error: {res['stderr']}")
        
        response = "\n".join(response_parts) if response_parts else "<Empty Response>"
        
        commands.append({
            "id": i,
            "command": cmd,
            "full_prompt": f"Execute command: {cmd}",
            "response": response
        })
    
    return {
        "system_prompt": scenario.get("cli_system_prompt", ""),
        "goal": goal,
        "commands": commands
    }

def save_output(data: Dict, scenario_id: str):
    """Save output to timestamped file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"red_blue_output_scenario_{scenario_id}_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nOutput saved to: {filename}")

def run_session(scenario_id: str):
    """Run a red vs blue session"""
    scenarios_data = load_scenarios()
    
    if scenario_id not in scenarios_data["scenarios"]:
        print(f"Error: Scenario {scenario_id} not found")
        return
    
    scenario = scenarios_data["scenarios"][scenario_id]
    goal = scenario["goal"]
    attacker_system_prompt = scenario["attacker_system_prompt"].format(goal=goal)
    cli_system_prompt = scenario["cli_system_prompt"]
    
    print(f"=== Red vs Blue Clone - Scenario {scenario_id} ===")
    print(f"Goal: {goal}")
    print(f"Attacker Model: {ATTACKER_MODEL}")
    print(f"CLI Model: {CLI_MODEL}")
    print("=" * 50)
    
    history = []
    attacker_messages = [
        {"role": "system", "content": attacker_system_prompt}
    ]
    
    for step in range(MAX_STEPS):
        print(f"\n--- Step {step + 1} ---")
        
        # Get attacker action
        response = attacker_call(attacker_messages)
        if not response:
            print("Error: No response from attacker model")
            break
        
        print(f"Attacker response: {response}")
        
        # Extract action
        action = extract_json_action(response)
        if not action:
            print("Error: Could not parse action JSON")
            attacker_messages.append({"role": "assistant", "content": response})
            attacker_messages.append({"role": "user", "content": "Please provide a valid JSON action with 'action' field."})
            continue
        
        action_type = action.get("action")
        
        if action_type == "run_shell":
            cmd = action.get("args", {}).get("cmd", "").strip()
            if not cmd:
                print("Error: Empty command")
                continue
            
            print(f"Executing: {cmd}")
            
            # Execute command
            result = cli_call(cmd, history, cli_system_prompt)
            history.append({"cmd": cmd, "res": result})
            
            print(f"Result: {result}")
            
            # Update attacker context
            attacker_messages.append({"role": "assistant", "content": response})
            attacker_messages.append({"role": "user", "content": f"Command result: {json.dumps(result)}"})
            
            # Check for success
            if has_success(history):
                print("\n🎉 SUCCESS: Flag found!")
                attacker_messages.append({"role": "user", "content": "Success! Flag found. Provide a final summary."})
                
                # Get final summary
                final_response = attacker_call(attacker_messages)
                print(f"Final summary: {final_response}")
                break
        
        elif action_type == "final":
            answer = action.get("args", {}).get("answer", "No answer provided")
            print(f"\n=== FINAL REPORT ===")
            print(answer)
            break
        
        else:
            print(f"Unknown action: {action_type}")
            attacker_messages.append({"role": "assistant", "content": response})
            attacker_messages.append({"role": "user", "content": "Unknown action. Use 'run_shell' or 'final'."})
    
    # Generate and save output
    output_data = format_claude_output(scenario, history, goal)
    save_output(output_data, scenario_id)
    
    print(f"\nSession completed. Total commands executed: {len(history)}")
    if history:
        print(f"Last command output: {history[-1]['res']['stdout']}")

def main():
    """Main entry point"""
    import sys
    
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY not set")
        exit(1)
    
    if not ATTACKER_API_KEY:
        print("Error: ATTACKER_API_KEY not set")
        exit(1)
    
    print("Red vs Blue Clone")
    print(f"Attacker Model: {ATTACKER_MODEL}")
    print(f"CLI Model: {CLI_MODEL}")
    
    # Load available scenarios
    scenarios_data = load_scenarios()
    available_scenarios = list(scenarios_data["scenarios"].keys())
    
    print(f"Available scenarios: {', '.join(available_scenarios)}")
    
    # Check for command line argument
    if len(sys.argv) > 1:
        scenario_id = sys.argv[1]
    else:
        try:
            scenario_id = input("Enter scenario ID (default: 1): ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            print("\nUsing default scenario: 1")
            scenario_id = "1"
    
    if scenario_id not in available_scenarios:
        print(f"Invalid scenario ID. Available: {', '.join(available_scenarios)}")
        exit(1)
    
    run_session(scenario_id)

if __name__ == "__main__":
    main()
