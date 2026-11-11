@echo off
rem train-watch: start pc_agent (machine name / paths are in watch\pc_config.json).
rem Double-click, or put a shortcut to this file in shell:startup to start at login.
rem Restarts pc_agent 30 s after it exits. Close this window (or Ctrl-C) to stop.
cd /d "%~dp0.."
if not exist watch\pc_config.json (
  echo watch\pc_config.json not found.
  echo Copy watch\pc_config.example.json to watch\pc_config.json and edit machine / repo_dir / data_dir / launch_shell.
  pause
  exit /b 1
)
title spica-watch pc_agent
rem Prefer the Python launcher (py -3); fall back to python. "python" alone may be the Microsoft Store alias.
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
:loop
%PY% watch\pc_agent.py loop
echo pc_agent exited with code %errorlevel%. Restarting in 30 s ...
timeout /t 30
goto loop
