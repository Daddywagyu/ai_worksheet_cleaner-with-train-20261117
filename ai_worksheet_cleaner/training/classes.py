"""Class definitions shared by training, evaluation and runtime."""
CLASS_NAMES = ["background", "printed", "red_ink", "blue_ink", "pencil", "other_handwriting", "uncertain"]
BACKGROUND, PRINTED, RED, BLUE, PENCIL, OTHER, UNCERTAIN = range(7)
NUM_CLASSES = 7
INK_CLASSES = (RED, BLUE, OTHER)          # Phase 2 (pencil = Phase 3, uncertain = runtime only)
TRAINED_CLASSES = (BACKGROUND, PRINTED, RED, BLUE, OTHER)
IGNORE_INDEX = 255
TRAIN_DPI = 100                           # model is trained on 100-DPI renderings
ALL_INK_CLASSES = (RED, BLUE, PENCIL, OTHER)   # Phase 3: pencil model trains on pens + pencil
