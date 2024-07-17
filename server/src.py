import random

import torch
from spacy.lang.ar import Arabic
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.ja import Japanese
from spacy.lang.zh import Chinese

def get_spacy_lang(lang):
    if lang == "zh":
        return Chinese()
    elif lang == "ja":
        return Japanese()
    elif lang == "ar":
        return Arabic()
    elif lang == "es":
        return Spanish()
    else:
        # For most languages, Enlish does the job
        return English()

def split_sentence(text, lang):
    text_splits = []
    text_splits.append("")
    nlp = get_spacy_lang(lang)
    nlp.add_pipe("sentencizer")
    doc = nlp(text)
    for sentence in doc.sents:
        text_splits.append(str(sentence))

    if len(text_splits) > 1:
        if text_splits[0] == "":
            del text_splits[0]
    
    return text_splits


def local_generation(speaker_embedding, gpt_cond_latent, model, text, language, silence_dot):
    wavs = []
    text = split_sentence(text, language)
    start = 0
    end = 0
    time_stamp = []

    for sent in text:
        out = model.inference(
            sent,
            language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            enable_text_splitting=False,
        )
        dot_silence = torch.zeros(random.choices(silence_dot))
        dot_silence_length = dot_silence.shape[0] / 24000
        audio_length = out["wav"].shape[0] / 24000

        end = start + audio_length
        time_stamp.append((start, end))
        start = end + dot_silence_length

        wavs.append(torch.from_numpy(out["wav"]))
        wavs.append(dot_silence)
        
        end = start + audio_length
        time_stamp.append((start, end))
        start = end + dot_silence_length
        
        wavs.append(dot_silence)
    
    wav_output = torch.cat(wavs, dim=0)
    
    return wav_output