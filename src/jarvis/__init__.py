"""JARVIS: Korean/English voice agent for Windows that fights in Roblox *untitled boxing game*.

The game-facing code only ever observes the real Roblox window (Windows Graphics Capture) and
acts through real OS input (Win32 ``SendInput``). Nothing in this package injects into, reads the
memory of, or otherwise tampers with the Roblox process.
"""

__version__ = "0.1.0"
