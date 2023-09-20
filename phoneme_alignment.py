# Description: This script aligns phonemes from a ground-truth transcript to a speech audio file. 
# This script requires a path to an audio file and a transcript file in roman alphabet as input. It outputs 2 JSON files containing the phonemes 
# and their timestamps in seconds. The first is using wav2vec2 base 960 and the second is using a fine-tuned IPA wave2vec2 model.

# To call this script, run 'python phoneme_alignment.py --audio_path input/assessment_9.mp3 --transcript_path input/assessment_9.txt'

import torch
import torchaudio
from torchaudio.datasets import CMUDict
import torchaudio.transforms as T
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from dataclasses import dataclass
import IPython
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.figsize"] = [16.0, 4.8]
import json
import re
import os
import string
import copy
import IPython
import tgt
import argparse
import tempfile

from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC

# Parse arguments
parser = argparse.ArgumentParser(
    description='Align phonemes from a ground-truth transcript to a speech audio file. '
                'This script requires a path to an audio file and a transcript file in '
                'roman alphabet as input. It outputs 2 JSON files containing the phonemes '
                'and their timestamps in seconds. The first is using wav2vec2 base 960 '
                'and the second is using a fine-tuned IPA wave2vec2 model.')

parser.add_argument('--audio_path', type=str, help='Path to audio file')
parser.add_argument('--transcript_path', type=str, help='Path to transcript')
parser.add_argument('--g2p_path', type=str, default="./cmudict/g2p.json", help='Path to g2p dictionary')
parser.add_argument('--CMU_phonemes_path', type=str, default="./cmudict/SphinxPhones_40.txt", help='Path to CMU phonemes')

args = parser.parse_args()

audio_file = args.audio_path
transcript_file = args.transcript_path
g2p_path = args.g2p_path
CMU_phonemes_path = args.CMU_phonemes_path

# Helper functions
def load_audio(audio_file):
    root, _ = os.path.splitext(audio_file)
    output_file = f"{root}.wav"

    # Create a temporary output file
    temp_output_file = tempfile.mktemp(suffix=".wav")

    # Convert audio to mono 16khz 16-bit wav format which wav2vec2 models expect
    os.system(f'ffmpeg -y -i {audio_file} -acodec pcm_s16le -ac 1 -ar 16000 {temp_output_file}')

    waveform, sample_rate = torchaudio.load(temp_output_file)
    os.remove(temp_output_file)
    return waveform, sample_rate

def load_transcript(transcript_file):
    with open(transcript_file, "r") as file:
        transcript = file.read()
    return transcript

def format_transcript(transcript): # Remove punctuation, convert each letter to uppercase and join words with '|'
    import string
    translator = str.maketrans('', '', string.punctuation)
    cleaned_string = transcript.translate(translator)
    words = cleaned_string.split()   
    result_string = '|'.join(word.upper() for word in words)
    return result_string

def save_phonemes_JSON(phonemes, output_file):
    phoneme_segments = []
    for phoneme in phonemes:
        phoneme_segments.append({'phoneme': phoneme.label, 'start': phoneme.start, 'end': phoneme.end})
    with open(output_file, 'w') as outfile:
        json.dump(phoneme_segments, outfile)

def get_trellis(emission, tokens, blank_id=0): 
    # Trellis is a matrix of (Frames, Transcript Letters) where each element is the probability of the transcript letter occuring at that frame. 
    # The general idea is that wav2vec2 outputs a probability for each letter in the vocab per frame, but we want to align a ground truth transcript 
    # who's letters may not perfectly match the maximum probabiliity of the wav2vec2 output at each frame.
    # Traversing the most likely path on the trellis will give us the most likely alignment timestamps between the wav2vec2 output and the ground truth transcript.
    # The following functions are adapted from https://pytorch.org/audio/stable/tutorials/forced_alignment_tutorial.html which has further details. 
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    # Trellis has extra dimensions for both time axis and tokens.
    # The extra dim for tokens represents <SoS> (start-of-sentence)
    # The extra dim for time axis is for simplification of the code.
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens],
        )
    return trellis

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

def backtrack(trellis, emission, tokens, blank_id=0):
    # Note:
    # j and t are indices for trellis, which has extra dimensions
    # for time and tokens at the beginning.
    # When referring to time frame index `T` in trellis,
    # the corresponding index in emission is `T-1`.
    # Similarly, when referring to token index `J` in trellis,
    # the corresponding index in transcript is `J-1`.
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()

    path = []
    for t in range(t_start, 0, -1):
        # 1. Figure out if the current position was stay or change
        # Note (again):
        # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
        # Score for token staying the same from time frame J-1 to T.
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        # Score for token changing from C-1 at T-1 to J at T.
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # 2. Store the path with frame-wise probability.
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
        # Return token index and time index in non-trellis coordinate.
        path.append(Point(j - 1, t - 1, prob))

        # 3. Update the token
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        raise ValueError("Failed to align")
    return path[::-1]

@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path, transcript):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words


def find_phonemes(letter_segments, word_segments, g2p):
    # Converting from graphemes to phonemes is not trivial because it is not a one-to-one mapping (it is many-to-many). For example, the word "hello" has 5 letters 
    # but 4 phonemes. In the word "memorable", the letter "l" maps to 2 phonemes - AH and L.
    # This algorithm merges the letters aligned using the CTC algorithm into phonemes using the g2p dictionary. It also adds phoneme separators between phonemes and 
    # word separators between words. Using this algorithm in production require lots of testing.
    # This function will modify letter_segments, so make a copy if you'd like to keep the original
    segment_index = 0
    letter_segments.append(Segment("|", letter_segments[-1].end, letter_segments[-1].end, 1.0)) # add a word separator at the end of the transcript
    for word in word_segments:
        # print(f"Scanning word {word.label}")
        phonemes_map = g2p[word.label.lower()]
        for idx, grapheme in enumerate(phonemes_map['graphemes']):
            if "|" not in grapheme:
                if grapheme != letter_segments[segment_index].label.lower():
                    # print(f"Error: {grapheme} != {letter_segments[segment_index].label}")
                    break
                letter_segments[segment_index].label = phonemes_map['phonemes'][idx]
                segment_index += 1

                if letter_segments[segment_index].label == "|":
                    # print(f"Found word separator after {word.label}")
                    segment_index += 1
                else:
                    # print(f"Adding phoneme separator after {letter_segments[segment_index-1].label}")
                    letter_segments.insert(segment_index, Segment("|", letter_segments[segment_index-1].end, letter_segments[segment_index].start, 1.0))
                    segment_index += 1
            else:
                # print(f"Found 2 letter phoneme {grapheme}")
                for idx2, letter in enumerate(grapheme.split("|")):               
                    letter_segments[segment_index].label = phonemes_map['phonemes'][idx]
                    # print(f"Letter {idx2}: {letter}. Segment: {letter_segments[segment_index].label}")
                    segment_index += 1
                if letter_segments[segment_index].label != "|":
                    # print(f"Adding phoneme separator after {letter_segments[segment_index-1].label}")
                    letter_segments.insert(segment_index, Segment("|", letter_segments[segment_index-1].end, letter_segments[segment_index].start, 1.0))
                    segment_index += 1
                else:
                    # print(f"Found word separator after {word.label}")
                    segment_index += 1

    # remove silent phonemes
    phoneme_segments = [segment for segment in letter_segments if segment.label != "_"]

    # mitigate double phonemes by splitting them evenly in time, this is an approximation which introduces small errors
    split_segments = []
    import numpy as np
    for segment in phoneme_segments:
        if segment.label == "|":
            split_segments.append(segment)
            continue
        phonemes = segment.label.split("|")
        num_parts = len(phonemes)
        if num_parts > 1:
            # print(f"Found {num_parts} phonemes in {segment.label}")
            parts = np.linspace(segment.start, segment.end, num_parts + 1, dtype=int)
            for idx, phoneme in enumerate(phonemes):
                split_segments.append(Segment(phoneme, parts[idx], parts[idx+1], segment.score))
                split_segments.append(Segment("|", parts[idx+1], parts[idx+1], 1.0))
        else:
            split_segments.append(segment)

    while split_segments[-1].label == "|":
        # print(f"Removing trailing word separator at {split_segments[-1].start}")
        split_segments = split_segments[:-1] # remove last "|" segment
    return split_segments

# Merge phonemes
def merge_repeat_phonemes(segments, separator="|"):
    # segments = phoneme_segments[:]
    phonemes = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                if all(seg.label == segs[0].label for seg in segs):
                    phoneme = segs[0].label
                else:
                    phoneme = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                phonemes.append(Segment(phoneme, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return phonemes

sphinx_to_ipa = {
  "'": "'",
  "[PAD]": "[PAD]",
  "[UNK]": "[UNK]",
  "AY": "aɪ",
  "B": "b",
  "C": "c",
  "D": "d",
  "EY": "e",
  "F": "f",
  "G": "g",
  "HH": "h",
  "IY": "i",
  "ZH": "ʒ",
  "K": "k",
  "L": "l",
  "M": "m",
  "N": "n",
  "AW": "o",
  "OW": "oʊ",
  "P": "p",
  "Q": "q",
  "R": "r",
  "S": "s",
  "T": "t",
  "UW": "u",
  "V": "v",
  "W": "w",
  "Y": "y",
  "Z": "z",
  "|": "|",
  "AE": "æ",
  "DH": "ð",
  "NG": "ŋ",
  "AA": "ɑ",
  "AO": "ɔ",
  "AH": "ə",
  "EH": "ɛ",
  "IH": "ɪ",
  "SH": "ʃ",
  "UH": "ʊ",
  "JH": "ʤ",
  "CH": "ʧ",
  "'": "ˈ",
  "TH": "θ",
  "ER": "ər", # pretrained model has no ɝ
  "OY": "ɔɪ"
}

# Main script
waveform, sample_rate = load_audio(audio_file)

transcript = load_transcript(transcript_file)
transcript = format_transcript(transcript)
print(f"Transcript: {transcript}")

print("Aligning phonemes using method 1: Using WAV2VEC2_ASR_BASE_960H")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)
letters = bundle.get_labels()
letters_dictionary = {c: i for i, c in enumerate(letters)} # {letter: index} corresponding to the pretrained wav2vec2 model
tokens = [letters_dictionary[c] for c in transcript] # [index of first letter in transcript, index of second letter in transcript, ...]
# print(f"Transcript tokens: {tokens}")

with torch.inference_mode():
    emissions, _ = model(waveform.to(device))
    emissions = torch.log_softmax(emissions, dim=-1) # According to the tutorial, this avoids numerical instability -- should test whether it reduces performance
emission = emissions[0].cpu().detach() # Shape: (Frames, Letters)

# We use the CTC algorithm to find the most likely alignment timestamps between the wav2vec2 emission and the ground truth transcript.
# torch.nn.CTCLoss will give us the probability of the ground truth transcript matching the wav2vec2 output, but to get the most likely timestamps, we have to implement the CTC algorithm.

trellis = get_trellis(emission, tokens)

path = backtrack(trellis, emission, tokens)
# for p in path:
#     print(f"Letter: {transcript[p.token_index]}, Frame: {p.time_index}, Probability: {p.score}")

letter_segments = merge_repeats(path, transcript)
# for seg in letter_segments:
#     print(f"Letter: {seg.label}, Frame(s): {seg.start} - {seg.end}, Probability: {seg.score}")

# Now, we merge words and average the probability scores
word_segments = merge_words(letter_segments)
# for word in word_segments:
#     print(f"Word: {word.label}, Frame(s): {word.start} - {word.end}, Probability: {word.score}")

# If we wanted forced alignment of letters or words, we would be done. However, we want forced alignment of phonemes, so we need to convert the letters to phonemes.
# We make use of the CMU grapheme 2 phoneme alignment dictionary (aligned using Phonetisaurus) hosted here: https://github.com/ckw017/aligned-cmudict/blob/master/g2p.json

# Let's clean up the g2p dictionary a bit and make sure it's using the Sphinx 40 Phoneme set.

import json
import re
with open(g2p_path, "r") as f:
    g2p = json.load(f)

to_remove = []

unique_phonemes = set()
for key in g2p:
    if not any("foreign" in element for element in g2p[key]['phonemes']) and not any("#" in element for element in g2p[key]['phonemes']) and not any("old" in element for element in g2p[key]['phonemes']):
      for phoneme in g2p[key]['phonemes']:
        for individual_phoneme in phoneme.split('|'):
            unique_phonemes.add(re.sub(r'\d+', '', individual_phoneme))
    else:
      to_remove.append(key)

for key in to_remove:
    del g2p[key]

# Read CMU Phonemes into a list

CMU_phonemes_path = "./cmudict/SphinxPhones_40.txt"
CMU_phonemes = set()
with open(CMU_phonemes_path, "r") as f:
    for line in f:
        CMU_phonemes.add(line.strip())

letter_segments_copy = copy.deepcopy(letter_segments) # This function will modify letter_segments, so make a copy if you'd like to keep the original
phoneme_segments = find_phonemes(letter_segments_copy, word_segments, g2p)
phoneme_segments = merge_repeat_phonemes(phoneme_segments)

ratio = waveform.size()[1] / (trellis.size(0) - 1)
for phoneme in phoneme_segments:
    phoneme.label = re.sub(r'\d', '', phoneme.label) # we can remove the stress markers, but keep in mind they may be useful for future work
    phoneme.start = ratio * phoneme.start / sample_rate # convert to seconds
    phoneme.end = ratio * phoneme.end / sample_rate
    # print(f"Phoneme: {phoneme.label}, Time: {phoneme.start:.3f}s - {phoneme.end:.3f}s")

# Save phonemes to JSON
output_file = "output/method_1.json"
save_phonemes_JSON(phoneme_segments, output_file)

print("Aligned successfully")

# Method 2: Using pretrained wav2vec2 model finetuned on phonemes
print("Aligning phonemes using method 2: Pretrained phoneme model wav2vec2-large-english-phoneme-v2")
# Load the pretrained phoneme recognition model

letters_dictionary = json.load(open("./wav2vec2-large-english-phoneme-v2/vocab.json"))
tokenizer = Wav2Vec2CTCTokenizer("./wav2vec2-large-english-phoneme-v2/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
model = Wav2Vec2ForCTC.from_pretrained("./wav2vec2-large-english-phoneme-v2")

# Run inference on audio file
model.to(device)
input_values = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt").input_values.to(device)

with torch.no_grad():
  emissions = model(input_values).logits
emission = emissions[0].cpu().detach()

# We only need the emissions to perform forced alignment, but we can print the transcript generate by the ASR model if we want.
pred_ids = torch.argmax(emissions, dim=-1)
# print(f"Letters: {' '.join(processor.tokenizer.convert_ids_to_tokens(pred_ids[0].tolist()))}")
pred_str = processor.batch_decode(pred_ids)[0]
# print(f"ASR Model Transcript (CTC Decoded): {pred_str}")

# Now, we convert the ground truth transcript to IPA

# First, convert the CMU dictionary to IPA
CMU = CMUDict(root='./cmudict', download=True)

CMU_dict = {}
for word in CMU:
	if word[0] == 'THE':
		CMU_dict['the'] = 'ðə' # Choose the other pronunciation of "the" to generate the correct IPA transcript for the test example
	else:
		CMU_dict[word[0].lower()] = [sphinx_to_ipa[re.sub(r'\d', '', phoneme.upper())] for phoneme in word[1]] # strip out stress markers

# print(f"Hello: {CMU_dict['hello']}")

ipa_transcript = [] # transcript in modified IPA
for word in transcript.split('|'):
  ipa_transcript += CMU_dict[word.lower()]
  ipa_transcript += ['|']

print(f"IPA Transcript: {''.join(ipa_transcript)}")

# Next, we proceed as in method 1 using the CTC algorithm to find the most likely alignment timestamps between the wav2vec2 output and the ground truth transcript.
with open("./wav2vec2-large-english-phoneme-v2/vocab.json", 'r', encoding='utf-8') as file:
    letters_dictionary = json.load(file)

tokens = [letters_dictionary[c] for c in ''.join(ipa_transcript)]
# print(tokens)
trellis = get_trellis(emission, tokens)
# print(trellis.shape)

path = backtrack(trellis, emission, tokens)
phoneme_segments = merge_repeats(path, ''.join(ipa_transcript))
phoneme_segments = [segment for segment in phoneme_segments if segment.label != "|"]

ratio = waveform.size()[1] / (trellis.size(0) - 1)
for phoneme in phoneme_segments:
    phoneme.start = ratio * phoneme.start / sample_rate # convert to seconds
    phoneme.end = ratio * phoneme.end / sample_rate
    # print(f"Phoneme: {phoneme.label}, Time: {phoneme.start:.3f}s - {phoneme.end:.3f}s")

# Convert the phonemes to Arpabet

ipa_to_sphinx = {v: k for k, v in sphinx_to_ipa.items()}
index = 0
for grapheme in [letter for letter in ipa_transcript if letter != '|']:
    if len(grapheme) == 1:
        phoneme_segments[index].label = ipa_to_sphinx[grapheme]
        index += 1
    else:
        length = len(grapheme)
        phoneme_segments[index].label = ipa_to_sphinx[grapheme]
        phoneme_segments[index].end = phoneme_segments[index + length - 1].end
        del phoneme_segments[index + 1:index + length]
        index += 1

# for phoneme in phoneme_segments:
#     print(f"Phoneme: {phoneme.label}, Time: {phoneme.start:.3f}s - {phoneme.end:.3f}s")

# Save phonemes to JSON
output_file = "output/method_2.json"
save_phonemes_JSON(phoneme_segments, output_file)

print("Aligned successfully")