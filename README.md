# calvin_env



## Installation
```bash
git clone --recursive https://github.com/mees/calvin_env.git
cd calvin_env/tacto
pip install -e .
cd ..
pip install -e .
```

## Replay dataset
```bash
python calvin_env/envs/play_table_env.py
```

## Replace Panda with UR5
Replace ```panda_longer_finger``` with ```ur5``` in ```calvin_env/conf/config_data_collection.yaml```
