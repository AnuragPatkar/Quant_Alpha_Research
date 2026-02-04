import os
import subprocess
import sys

def run_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return e.stderr.strip()

def main():
    print("🚑 STARTING EMERGENCY GIT REPAIR...")

    # 1. Verify Project Root
    if not os.path.exists(".git"):
        print("❌ ERROR: No .git folder found. Are you in the project root?")
        return

    # 2. Force Write UTF-8 .gitignore
    gitignore_content = (
        "venv/\n"
        "venv\n"
        "__pycache__/\n"
        "*.pyc\n"
        ".env\n"
        "data/\n"
        "models/\n"
        "results/\n"
        "config/api_keys.py\n"
        ".vscode/\n"
        ".idea/\n"
    )
    
    try:
        # Force UTF-8 encoding (Crucial for Windows)
        with open(".gitignore", "w", encoding="utf-8") as f:
            f.write(gitignore_content)
        print("✅ .gitignore re-written with UTF-8 encoding.")
    except Exception as e:
        print(f"❌ Failed to write .gitignore: {e}")
        return

    # 3. Diagnose: Does Git see the rule?
    print("🔍 Testing Git Ignore Rules...")
    # We ask git: "Would you ignore venv/Scripts/python.exe?"
    check = run_command("git check-ignore -v venv/Scripts/python.exe")
    
    if ".gitignore" in check:
        print(f"✅ SUCCESS: Git is reading the file! Rule found: {check}")
    else:
        print("❌ FAILURE: Git is NOT ignoring venv. The file might still be corrupt or path is wrong.")
        print(f"Debug Output: {check}")
        # Stop here if it fails
        return

    # 4. Clean and Reset
    print("🧹 Cleaning Git Index (This takes a moment)...")
    run_command("git rm -r --cached .")
    
    print("➕ Re-adding files...")
    run_command("git add .")
    
    # 5. Final Status
    status = run_command("git status")
    if "new file:   venv/" in status:
        print("\n❌ FAILED. 'venv' is still present.")
    else:
        print("\n🏆 REPAIR COMPLETE. 'venv' is gone!")
        print("You can now commit.")

if __name__ == "__main__":
    main()