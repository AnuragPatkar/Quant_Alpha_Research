import os
import subprocess

def force_exclude():
    print("🛡️ Applying Git Internal Override...")
    
    # 1. Locate the secret exclude file
    git_dir = ".git"
    if not os.path.exists(git_dir):
        print("❌ Error: .git folder not found. Make sure you are in the project root.")
        return

    exclude_file = os.path.join(git_dir, "info", "exclude")
    
    # 2. Write the rules directly into Git's internal config
    rules = [
        "venv/",
        "venv",
        "__pycache__/",
        "*.pyc",
        ".env",
        "data/",
        "models/",
        "results/",
        "config/api_keys.py",
        ".vscode/"
    ]
    
    try:
        with open(exclude_file, "a") as f:
            f.write("\n" + "\n".join(rules) + "\n")
        print(f"✅ Rules injected into {exclude_file}")
    except Exception as e:
        print(f"❌ Failed to write to exclude file: {e}")
        return

    # 3. Aggressive Cleanup
    print("🧹 Force cleaning Git index...")
    subprocess.run("git rm -r -f --cached .", shell=True) 
    
    # 4. Re-add
    print("➕ Re-adding files...")
    subprocess.run("git add .", shell=True)
    
    # 5. Status
    print("📊 Status Check:")
    subprocess.run("git status", shell=True)

if __name__ == "__main__":
    force_exclude()