"""Legacy entrypoint shim.

This file used to host an older demo with obsolete import paths.
Use the unified CLI in run.py for all current workflows.
"""

from run import main as cli_main


if __name__ == "__main__":
    cli_main()
