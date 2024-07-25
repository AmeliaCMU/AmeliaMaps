# Installation

Installation can be done by installing the AmeliaScenes repository or the complete Amelia Framework

## Creating an environment

It can be created using either `conda`.

- Conda

```bash
conda create --name amelia python=3.9
conda activate amelia
```

### Installing AmeliaMaps

Download the GitHub repository and install requirements:

```bash
git clone git@github.com:AmeliaCMU/AmeliaMaps.git
cd AmeliaMaps
pip install -e .
```

## Installing the Amelia Framework

Download the `install.sh` script from the [AmeliaScenes](https://github.com/AmeliaCMU/AmeliaScenes/blob/main/install.sh) repository and run it:

```bash
chmod +x install.sh
./install.sh amelia
```
