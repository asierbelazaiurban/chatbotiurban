import subprocess

def run_tests():
    subprocess.run(["python", "run_tests.py"], check=True)

if __name__ == "__main__":
    run_tests()