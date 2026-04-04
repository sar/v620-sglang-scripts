#!/usr/bin/env python3
"""
Entry point for SGLang in ROCm Docker. Native RoPE workaround is applied in
sitecustomize.py (every process); this script only forwards to launch_server.
"""
from __future__ import annotations

import runpy


def main() -> None:
    runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
