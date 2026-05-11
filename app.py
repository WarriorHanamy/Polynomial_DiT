#!/usr/bin/env python3
"""CLI entrypoint for Polynomial DiT demo."""

import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: demo <points|trajectory>", file=sys.stderr)
        sys.exit(1)

    subcommand = sys.argv[1]

    if subcommand == "points":
        from demo_points import create_simplified_points_demo

        create_simplified_points_demo()
    elif subcommand == "trajectory":
        from demo_polynomial import create_simplified_demo

        create_simplified_demo()
    else:
        print(f"Unknown subcommand: {subcommand}", file=sys.stderr)
        print("Usage: demo <points|trajectory>", file=sys.stderr)
        sys.exit(1)
