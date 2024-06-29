# How To Use the Code

***Disclamer***
The functionality of this project is not guaranteed. While I strive to ensure the reliability and accuracy of the applications, I cannot address all possible usage combinations.
If you encounter any issues, please raise an issue on GitHub.

These applications are designed to assist in avoiding the complexity of setting up a Python environment. However, I do not guarantee that they will work on all platforms.

## Using compiled app

Compiled apps are available in the release section of this repository. Download the app for your operating system and run it.
[You can click here](https://github.com/Sakagraisse/The-great-drift/releases/tag/1.0)

### Windows (only x86_64)
1. Download the .exe file
2. Run the .exe file

notes : 
- If you get a warning from Windows Defender, click on "More info" and then "Run anyway"
- If you get a warning from SmartScreen, click on "More info" and then "Run anyway"
- If you get a warning from your antivirus, you can either disable it or add an exception for the app
- The app may be slow to launch on some computers, please be patient
- On the first simulation, part of the code is autocompiled, this may take a few seconds

### Linux (only x86_64)
1. The tarball is available in the release section of this repository. Download the .tar.gz file
2. Extract the .tar.gz file
3. Run the ./The-great-drift file

notes :
- If you get a warning about the file being executable, you can either run chmod +x ./The-great-drift or run the file with sh ./The-great-drift
- The app may be slow to launch on some computers, please be patient
- On the first simulation, part of the code is autocompiled, this may take a few seconds
- If you get a warning about missing libraries, you may need to install them. For example, on Ubuntu, you can run sudo apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

### MacOS (only arm64)
1. The "The-great-drift.app" is available in the release section of this repository. Download the .app file
2. Run the .app file

notes :
- If you get a warning about the app being from an unidentified developer, you can either run the app with right-click -> Open or go to System Preferences -> Security & Privacy -> General and click on "Open Anyway"
- The app may be slow to launch on some computers, please be patient
- On the first simulation, part of the code is autocompiled, this may take a few seconds

## Using the source code

### Requirements
- Python 3.11
- Using a conda environment is recommended
- The following packages are required:
  - numpy
  - pyqt5
  - pyqtgraph
  - pyopengl
  - pyopengl-accelerate
  - pyinstaller (only for compiling the app)

### Installation
1. Clone the repository
2. Install the required packages
    1. you can use pip : pip install -r requirements.txt
3. Run the GUI.py file


