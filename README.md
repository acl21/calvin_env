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

## Robot replacement
Replace ```robot``` in ```calvin_env/conf/config_data_collection.yaml```. Possible options ```panda_longer_finger```, ```ur5```, ```ur5_with_panda_gripper```
