import random

import torch
from spacy.lang.ar import Arabic
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.ja import Japanese
from spacy.lang.zh import Chinese

silence_dot = [0.1, 0.15, 0.05, 0.2]

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

def convert_seconds(seconds):
    # Calculate hours
    hours = seconds // 3600
    seconds %= 3600

    # Calculate minutes
    minutes = seconds // 60
    seconds %= 60

    # Calculate milliseconds
    milliseconds = (seconds - int(seconds)) * 1000

    # Convert remaining seconds to an integer
    seconds = int(seconds)

    return f"{int(hours):02}:{int(minutes):02}:{seconds:02},{int(milliseconds):03}"

def local_generation(speaker_embedding, gpt_cond_latent, model, text, language, silence_length, temperature, top_k, top_p, speed):
    wavs = []
    text = split_sentence(text, language)
    start = 0
    end = 0
    time_stamp = []
    srt_file = []

    for sent in text:
        out = model.inference(
            sent,
            language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            speed=speed,
            enable_text_splitting=False,
        )
        random_choice = random.choice([0, 1])
        if random_choice == 0:
            silence = random.choices(silence_dot)[0] + silence_length
            silence = int(silence * 24000)
        else:
            silence = random.choices(silence_dot)[0] + silence_length
            silence = int(silence * 24000)

        dot_silence = torch.zeros([silence])
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
    
    for id, (t, segment) in enumerate(zip(text, time_stamp)):
        startime = convert_seconds(segment[0])
        endtime = convert_seconds(segment[1])
        segment_id = str(id+1)
        if isinstance(t, str):
            segment = f"{segment_id}\n{startime} --> {endtime}\n{t}\n\n"
        else:
            t = ", ".join(t)
            segment = f"{segment_id}\n{startime} --> {endtime}\n{t}\n\n"
        
        srt_file.append(segment)
    
    wav_output = torch.cat(wavs, dim=0)
    
    return wav_output, srt_file
