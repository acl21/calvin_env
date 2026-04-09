# calvin_env

Supports CALVIN environment with a human arm in the scene.


|     |
| --- |
|     |


## Installation

```bash
git clone --recursive https://github.com/mees/calvin_env.git
cd calvin_env/tacto
pip install -e .
cd ..
pip install -e .
```

## Human Arm

### To calibrate human arm pose

```bash
python calvin_env/envs/play_env_table_calibrate_armpose.py
```

Use these keys in the PyBullet GUI window:

```
J / L: x - / +
I / K: y + / -
U / O: z + / -
R / F: roll + / -
T / B: pitch + / -
Y / H: yaw + / -
= / -: scale + / -
P: print current xyz/rpy/scale to logs
```

### Regenerate CALVIN data with the human arm

```bash
python scripts/regenerate_calvin_humanarm_data.py
```

The above script can regenerate CALVIN data with UR5 robot in the environment. To choose which data to store, change `DATA_KEYS` variable. Make sure the above script is using the config `config_human_arm.yaml`.

