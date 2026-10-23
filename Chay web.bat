@echo off
title Smoke Guard AI
cd /d "%~dp0"
start "" http://localhost:5000
py serve.py
pause
