"""
Luxanix Studio Pro — Entrypoint
Autonomous AI Video Editor & Remastering Suite with NVIDIA RTX 50 Blackwell support.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ui.desktop_app import LuxanixDesktopApp

def main():
    app = LuxanixDesktopApp()
    app.mainloop()

if __name__ == "__main__":
    main()
