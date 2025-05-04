# AskDOC: Your AI literature review assistant
---
AskDOC is an AI-powered tool designed to help users extract, search, and understand content from any files. Leveraging advanced machine learning models, Askdoc enables fast and efficient querying of document text, localisation of content, and identification of sources, offering an intuitive interface for easy document navigation. Perfect for anyone looking to analyse large documents quickly.
---
#### Clone the repo:

Clone the repo using ```git clone https://github.com/EmmAnkit/AskDOC.git```
---
#### Install Pandoc and LibreOffice on your device to read .docx, .pptx files:
---
##### Pandoc:
---
On Mac: ```brew install pandoc```
On Linux: ```sudo apt-get install pandoc```
On Windows: You need to manually download and install from [Pandoc Installer](https://pandoc.org/installing.html)
---
##### LibreOffice:
---
On Mac: ```brew install --cask libreoffice```
On Linux: ```sudo apt-get install libreoffice```
On Windows: You need to manually download and install from [LibreOffice Installer](https://www.libreoffice.org/get-help/install-howto/windows/)
---
### Make sure every file in the repo is in the same directory.
---
##### We have created a bash script to run the code. Following is the instruction on how to run it:

On Mac: ```bash run.sh```

On Linux: ```bash run.sh```

On Windows (if you have git installed): ```sh run.sh```
---

If none of these work, input all of these in the terminal manually:

```python3 -m pip install -r requirements.txt```

```streamlit run app.py```

