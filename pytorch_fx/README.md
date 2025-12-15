# Introduction to torch fx

1. basic tools and usage
2. 



# Installation
On mac:

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# check if you have metal acceleration:
python3 -c "import torch; print(torch.backends.mps.is_available())"

pip install tabulate # for print_tabulate in fx
```

