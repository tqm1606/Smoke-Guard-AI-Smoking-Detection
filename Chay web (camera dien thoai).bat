@echo off
title Smoke Guard AI - HTTPS cho camera dien thoai
cd /d "%~dp0"
py serve.py --https
pause
