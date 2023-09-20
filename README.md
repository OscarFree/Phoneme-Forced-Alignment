# Phoneme-Forced-Alignment
Comparison of methods to perform forced-alignment of phonemes in English

Start with phoneme_alignment_tutorial.ipynb which provides an explanation of the three different methods I used to solve this problem.

To align phonemes of speech audio to a ground truth transcript, run 

    python phoneme_alignment.py --audio_path path/to/audio.mp3 --transcript_path path/to/transcript.txt
    
Alignment results for each method are stored in the output folder as json.

Method 1: wav2vec2 base 960h

Method 2: wav2vec2 base 960h fine-tuned on English IPA

Method 3: Montreal Forced Alignment (MFA)
Note that MFA requires its own Conda environment -- please refer to the notebook for further details.

Methods 1 and 2 are based on the CTC algorithm, Method 3 is not.

To use method 2, you must first download https://huggingface.co/speech31/wav2vec2-large-english-phoneme-v2/blob/main/pytorch_model.bin
