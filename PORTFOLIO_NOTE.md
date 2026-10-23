# Portfolio context

Smoke Guard AI demonstrates a Python/Flask web application for analyzing images, video files and live camera feeds with a customized YOLO11 detection checkpoint (EMA + LSKA).

## Stack
Python, Ultralytics YOLO11, OpenCV, Flask, Waitress, HTML/CSS/JavaScript.

## Run locally
1. Install packages: `python -m pip install -r requirements.txt`.
2. Start: `python serve.py` (local webcam and uploaded files).
3. For phone-camera access on the same trusted local network: `python serve.py --https`.
4. Open the address printed in the terminal.

## Attribution
This file is a project summary for a personal portfolio. Specify the original project authors, dataset/model sources, licenses, and your own contributions before publishing. Do not present team work as solely your own.

## Security
This repository does not include local uploads, generated results, private keys, or runtime caches. The supplied checkpoint is retained; verify model provenance and redistributability before making the repository public.
