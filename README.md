# SEST-Chinese

## Installation

```
pip install -r requirements.txt
wget http://ospm9rsnd.bkt.clouddn.com/model/ltp_data_v3.4.0.zip
unzip ltp_data_v3.4.0.zip && rm ltp_data_v3.4.0.zip
```

## Quick Start

```
python main.py skipgram
python main.py SEST
```

## Trouble Shooting

### gcc: error trying to exec 'cc1plus': execvp: No such file or directory?

```
sudo apt-get install --reinstall build-essential
```