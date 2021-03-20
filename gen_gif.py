from big_sleep import Imagine
import sys

text = sys.argv[1]

train = Imagine(
    text = text,
    lr = 0.07,
    save_every = 2,
    save_progress = True,
    epochs = 2
)

train()
