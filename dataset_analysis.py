import numpy as np
from pathlib import Path
import mido
from tqdm import tqdm
import matplotlib.pyplot as plt

data_N = 669

is_channel_visited = np.zeros((128), dtype=np.bool)
is_note_visited = np.zeros((128), dtype=np.bool)
lengths = []

for filepath in tqdm(Path("dataset/midi/adjust_tempo").glob("*.mid"), total=data_N):
    midi = mido.MidiFile(filepath)
    lengths.append(midi.length)
    for msg in midi:
        if not msg.is_meta:
            if is_channel_visited[msg.channel] == False:
                is_channel_visited[msg.channel] = True
            if hasattr(msg, 'note') and is_note_visited[msg.note] == False:
                is_note_visited[msg.note] = True

print(f"Channels in use: {np.where(is_channel_visited==True)[0].tolist()}")
print(f"Notes in use: {np.where(is_note_visited==True)[0].tolist()}")
plt.hist(lengths)
plt.show()
