# Phoneme-Forced-Alignment
Comparison of methods to perform forced-alignment of phonemes in English

Start with phoneme_alignment_tutorial.ipynb which provides an explanation of the three different methods I used to solve this problem.

To align phonemes of speech audio to a ground truth transcript, run 'python phoneme_alignment.py --audio_path path/to/audio.mp3 --transcript_path path/to/transcript.txt'
Alignment results are stored in method_1.json (wav2vec2 base 960h), method_2.json (wav2vec2 base 960h fine-tuned on English IPA), and method_3.json (Montreal Forced Alignment)

Methods 1 and 2 are based on the CTC algorithm, Method 3 is not.
Note that MFA requires its own Conda environment -- please refer to the notebook for further details.