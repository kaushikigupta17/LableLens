# LableLens# LabelLens

**Scan any label. Read it clearly. Hear it in your language.**

LabelLens is a web app that reads the small print on medicine, food and household labels, shows it in large readable text, translates it between English and Hindi, and reads it aloud. It is built for people who struggle with tiny, English-only packaging.

**The scanner that never guesses.** LabelLens only repeats what is printed on the label. Text it is unsure about is flagged or hidden instead of being presented as fact.

- **Live demo:** [(https://lablelens-ccakpmadpwgh2zm48ev4ul.streamlit.app/)]
- **Team:** [Infinite Loop]
- **Hackathon:** [WIE-Lumina'26]

---

## The problem

- Labels use tiny, dense print that is hard for older adults and people with low vision.
- Many labels are English-only, so Hindi-speaking users can miss directions, storage instructions, warnings and expiry dates.
- Misreading a label can lead to wrong use or missed warnings.
- Most people never complain. They guess, or ask someone else.

## Who it is for

- Older adults
- People with low vision
- Hindi-speaking users who are not comfortable with English labels
- Caregivers buying medicines or products for family members

## Features

- **Camera scan or photo upload** (works with a phone or laptop camera)
- **OCR in English, with optional Hindi text recognition**
- **Image preparation:** rotation fix, resizing and contrast boost so small print is easier to read
- **Confidence colours:** green means confident and the word is recognised, yellow means low confidence or an unrecognised word, and very unclear lines are hidden with a warning
- **Word-level flags:** in a confident line, only the doubtful word is marked yellow
- **Expiry badge:** shows "Not expired" or "EXPIRED" with the date text found, and says "probable" when no expiry keyword was read
- **Key facts card:** batch number, MRP and manufacturing date when they are read from the label
- **Red warning box detection:** finds red boxes such as "Schedule H prescription drug" and tries to read the light text inside. If it cannot, it shows the enhanced box image so the user can read it
- **Blur warning:** tells the user to retake a blurry photo
- **English to Hindi translation** with the original text shown side by side
- **Read aloud** for the whole label and for each section
- **Section cards:** Directions, Warnings, Storage, Ingredients and Other details
- **Safety notes:** always reminds users to double-check doses and dates and to verify with a pharmacist

## How it works

```
Photo -> Preprocess -> OCR -> Confidence and word check -> Translate -> Speak
```

1. **Preprocess:** fix rotation, resize, grayscale and contrast boost (OpenCV, Pillow)
2. **OCR:** read the text (EasyOCR)
3. **Confidence and word check:** score each line, and check each word against a dictionary (pyspellchecker). Doubtful text is flagged or hidden
4. **Translate:** English to Hindi (deep-translator, Google with MyMemory as fallback)
5. **Speak:** text-to-speech (gTTS)

## Safety principle

LabelLens is a reading aid, not a medical tool.

- It only repeats printed label text.
- It never invents medical advice or changes doses.
- It always shows the original text beside any translation.
- It flags uncertain text and asks the user to check the original packaging.

## Tech stack

Python, Streamlit, EasyOCR, OpenCV, Pillow, NumPy, deep-translator, gTTS, pyspellchecker

## Run it locally

You need Python 3.11 or 3.12 for the easiest install.

```bash
git clone [(https://github.com/kaushikigupta17/LableLens)]
cd [REPO FOLDER]
pip install -r requirements.txt
streamlit run app.py
```

The first scan downloads the OCR model, so it is slower. After that, scanning the same photo again is instant because results are cached.

## Tested on

- Cetaphil lotion (white label)
- A hair serum bottle
- Caberlin 0.25 (Cabergoline) foil strip, the hardest case, with a red warning box, rotated batch text and reflective foil

## Known limitations

- OCR runs on the CPU and can take 15 seconds or more per photo.
- Blurry photos, glare and glossy dark labels reduce accuracy. Sharp phone photos work much better.
- Faint light text on red boxes may not be read. In that case the app shows the box image instead of guessing.
- Rotated text, such as batch and expiry printed sideways, needs a separate "look for rotated text" scan.
- Ingredient lists in tiny print are often unreadable and are hidden or flagged.
- Section cards use simple rules, so some lines may land in the wrong card.
- Free translation and audio services can be rate-limited.

## Roadmap

- "Ask the label": answer questions using only the printed label text, or say "This isn't on the label. Ask your pharmacist."
- Voice-first mode for visually impaired users
- Allergen highlighter based on words printed on the label
- Large-text and high-contrast mode, and a more mobile-friendly layout
- Faster OCR and more Indian languages
- Testing on many more medicine labels

## Team
- Kaushiki Gupta
- Ipsita Kirati
- Vaishnavi Rastogi
- Mahi Singh


## Disclaimer

LabelLens can make mistakes. It is not a substitute for reading the original packaging or asking a pharmacist or doctor. Always double-check doses, dates and warnings.
