# Torchvision object detection tutorial

A Python library based around [this tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) from Torchvision.

## Instructions

```bash
git clone https://github.com/dewcservices/tvtuner.git
cd tvtuner
python -m venv venv --prompt .
source venv/bin/activate
python -m pip install .[dev]

# run examples
cd examples
./download_data.sh  # requires a Kaggle account
python main.py penn_fudan 
python main.py tomatoes
python main.py fixed_wing
python main.py quadcopter

# dockerise and serve
cd examples
docker build . -t tvtuner:latest
docker run -p 8080:8080 tvtuner:latest
curl localhost:8080/models
```
