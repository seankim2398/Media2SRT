# Media2SRT
Converts Video/Audio file into SRT subtitle file

# Quick Start
For Mac/Windows users, there is a known bug, please use the following workaround:
```
# install all dependencies except triton
pip install numba numpy torch tqdm more-itertools tiktoken==0.3.3
# install whisper-at without any dependency
pip install --no-deps whisper-at  
```
Also, install rust: 
```
https://www.rust-lang.org/tools/install
```

# Citation
```
@inproceedings{gong_whisperat,
  author={Gong, Yuan and Khurana, Sameer and Karlinsky, Leonid and Glass, James},
  title={Whisper-AT: Noise-Robust Automatic Speech Recognizers are Also Strong Audio Event Taggers},
  year=2023,
  booktitle={Proc. Interspeech 2023}
}
```
