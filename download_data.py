"""
Fetch the documents the chatbot reads.

The PDFs are not in this repository on purpose. They are free to download and
read, but they are other people's work and not mine to republish. So the repo
keeps the list of sources, and everyone downloads their own copies:

    python download_data.py       into data/, skipping anything already there
    python rag.py --rebuild       then embed them (a few minutes)
"""

import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
ARXIV = "https://arxiv.org/pdf/"

# name -> (url, what it is). Three textbooks for the foundations, then papers
# for the parts the books barely cover: vision, audio and multimodal models.
SOURCES = {
    "speech_and_language_processing":
        ("https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf",
         "Jurafsky & Martin, Speech and Language Processing (3rd ed. draft)"),
    "understanding_deep_learning":
        ("https://github.com/udlbook/udlbook/releases/download/v5.00/"
         "UnderstandingDeepLearning_11_21_24_C.pdf",
         "Simon Prince, Understanding Deep Learning (MIT Press)"),
    "foundations_of_large_language_models":
        (ARXIV + "2501.09223", "Xiao & Zhu, Foundations of Large Language Models"),

    "multimodal_foundation_models_survey":
        (ARXIV + "2309.10020", "Li et al. - multimodal foundation models, a survey"),
    "vision_transformer_vit":      (ARXIV + "2010.11929", "ViT - an image is worth 16x16 words"),
    "clip_text_image_pairs":       (ARXIV + "2103.00020", "CLIP - images and text in one space"),
    "llava_visual_instruction_tuning": (ARXIV + "2304.08485", "LLaVA - chatting about an image"),
    "flamingo_visual_language_model":  (ARXIV + "2204.14198", "Flamingo - few-shot vision-language"),
    "ddpm_denoising_diffusion":    (ARXIV + "2006.11239", "DDPM - the original diffusion model"),
    "latent_diffusion_stable_diffusion": (ARXIV + "2112.10752", "Latent diffusion / Stable Diffusion"),
    "whisper_speech_recognition":  (ARXIV + "2212.04356", "Whisper - speech recognition"),
    "wav2vec2_audio_representations": (ARXIV + "2006.11477", "wav2vec 2.0 - self-supervised audio"),
    "ethical_and_social_risks":    (ARXIV + "2112.04359", "Weidinger et al. - risks of harm from LMs"),
}


def fetch(name, url):
    path = os.path.join(DATA_DIR, name + ".pdf")
    if os.path.exists(path):
        return None                      # already here, leave it alone
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (student project)"})
    with urllib.request.urlopen(req) as response, open(path, "wb") as f:
        f.write(response.read())
    return os.path.getsize(path)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, (url, what) in SOURCES.items():
        size = fetch(name, url)
        state = "already there" if size is None else f"{size / 1e6:.1f} MB"
        print(f"   {name:38s} {state:14s} {what}")
    print(f"\n{len(SOURCES)} documents in data/. Next: python rag.py --rebuild")
