import subprocess
import sys
import os


def run_command(command):
    print(f"Running: {' '.join(command)}")
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        return False
    return True


def main():
    print("--- Git Hooks Setup Script ---")

    # 1. Check if git is initialized
    if not os.path.exists(".git"):
        print("Error: .git directory not found. Are you in the root of the repository?")
        sys.exit(1)

    # 2. Install pre-commit package
    print("\nStep 1: Installing pre-commit python package...")
    if not run_command([sys.executable, "-m", "pip", "install", "pre-commit"]):
        sys.exit(1)

    # 3. Install hooks
    print("\nStep 2: Installing Git hooks...")

    # Install standard pre-commit hooks
    if not run_command([sys.executable, "-m", "pre_commit", "install"]):
        print("Warning: Failed to install standard pre-commit hooks.")

    # Install commit-msg hooks (for gitlint)
    if not run_command(
        [sys.executable, "-m", "pre_commit", "install", "--hook-type", "commit-msg"]
    ):
        print("Warning: Failed to install commit-msg hooks.")

    print("\nSuccess! Git hooks are now installed.")
    print("These hooks will run automatically when you commit.")
    print("Note: If you use GitHub Desktop, it will now respect these rules.")


if __name__ == "__main__":
    main()
