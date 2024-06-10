# self-supervised-audio
Code for self-supervised learning of audio (waveform, but also maybe spectrograms)

- [ x ] implement VicReg
- [ x ] implement audio dataloaders
- [ x ] visualization & analysis

## current version
- all segments 10ms (~441 samples) in length
- vicreg v1: no overlap between audio segments
- vicreg v2: 50% overlap between audio segments

## next versions?

- [ ] longer segments (30ms? 40?)
- [ ] CPC (with AR context?)
- [ ] vicreg with context?
