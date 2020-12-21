# CSdigest

Dear friend, hopefully you already know what this repo is about from the message, so here is a quick start guide!

# Getting started

1. download or clone this repo: `git clone https://github.com/phildong/csdigest.git`
2. open the folder: `cd csdigest`
3. create an empty file named `token` and paste **Access Token** into its content.
4. download the `model` file from our shared google drive, under `/Phil/foodnet/model`, and put it under `/foodnet/` in the current folder.
5. create conda environment: `conda env create -f environment.yml`
6. activate conda environment: `conda activate digest`
7. run the script: `python csdigest.py`
8. you will see the newsletter as `csdigest.html` which can be opened with your favorite browser. Enjoy!

# Features
- parse messages from `#general` that's worth celebrating.
- extract pictures of foods from `#general` and `#homesanity`.
- parse all `#quotablequotes`.
