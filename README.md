# BrainTumorDetectionUsingDeepLearning
The repository features the source code of a Neural Network Model to predict Brain Tumors. It is a 'Team Project' to test the coding and software developing skills.
## Screenshots

Website Look
![App Screenshot](https://github.com/AnkitRupal/ConfigPhotos/raw/master/BTD_Intro.png)


## Dataset

We used the Kaggle BR35H dataset for training.
The dataset contains 3000 images, 1500 infected and 
1500 healthy.

We used another datset from Kaggle itself having just 253 images. We used
this dataset for testing purposes. 
## Accuracy

Accuracy on:

    1) TRAINING dataset is   : 99.17%

    2) VALIDATION dataset is : 97.88%

    3) TESTING dataset is    : 98.81%
## Tech Stack

#### NEURAL NETWORK BUILDING
    1) Tensorflow       2) MobileNet V2     3) Tensorflow_hub

#### WEBSITE DEVELOPMENT
    Streamlit


## Run Locally

Clone the project

```bash
  git clone https://github.com/AnkitRupal/BrainTumorDetectionUsingDeepLearning.git
```

Go to the project directory

```bash
  cd BrainTumorDetectionUsingDeepLearning
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  streamlit run app.py
```


## Optimizations

        1) Reduced prediction time to one-fourth the initial runtime, i.e., 4 times reduced.

        2) Improved model loading time by 25%.

        3) Improved Prediction Accuracy from 85% to >97.5%.
        
        4) Reduced backend load


## Running Tests

To run test dataset, run the following commands

```bash
  cd ./testing
  python testing.py
```


## License


![Permissions](https://github.com/AnkitRupal/ConfigPhotos/raw/master/MITLicence.png)

MIT License

Copyright (c) 2022 Ankit Rupal, Karan Dhar, Adil Vinayak and Lalit Kumar.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributors and Contributing

The contributors to the repository are:

    1) Ankit Rupal (@AnkitRupal)
    
    2) Karan Dhar (@Karandhar007)

    3) Adil Vinayak (@adil19-net)

    4) Lalit Kumar (@thelalitkumar)

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.
