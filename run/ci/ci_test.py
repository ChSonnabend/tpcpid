import subprocess
import sys
import os

print("Running CI tests...")

subprocess.run(
    ["python3", "-u", "run/run.py", "--config", "run/ci/ciconfig.json", "--ci-run", "1", "--skip-question", "1"],
    stderr=subprocess.STDOUT,
    check=True
)

if os.environ.get("TPCPID_RUN_O2PHYSICS_CI", "0") == "1":
    subprocess.run(
        ["python3", "-u", "run/ci/o2physics_pid_test.py", "--required"],
        stderr=subprocess.STDOUT,
        check=True
    )

print("CI tests completed successfully.")
