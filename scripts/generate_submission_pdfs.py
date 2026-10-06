"""Generate final 3 submission PDFs - simple student writing."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Preformatted,
    Image,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "submission"
OUT.mkdir(parents=True, exist_ok=True)
TITLE = "Continuous Speech Emotion Recognition"


def S():
    b = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "t", parent=b["Heading1"], fontName="Times-Bold", fontSize=13,
            leading=16, alignment=TA_CENTER, spaceAfter=4, textColor="black",
        ),
        "sub": ParagraphStyle(
            "s", parent=b["Normal"], fontName="Times-Roman", fontSize=11,
            leading=13, alignment=TA_CENTER, spaceAfter=12, textColor="black",
        ),
        "h1": ParagraphStyle(
            "h1", parent=b["Heading1"], fontName="Times-Bold", fontSize=12,
            leading=15, spaceBefore=10, spaceAfter=6, textColor="black",
        ),
        "h2": ParagraphStyle(
            "h2", parent=b["Heading2"], fontName="Times-Bold", fontSize=11,
            leading=14, spaceBefore=8, spaceAfter=4, textColor="black",
        ),
        "body": ParagraphStyle(
            "body", parent=b["Normal"], fontName="Times-Roman", fontSize=11,
            leading=15, alignment=TA_JUSTIFY, spaceAfter=7, textColor="black",
        ),
        "bullet": ParagraphStyle(
            "bu", parent=b["Normal"], fontName="Times-Roman", fontSize=11,
            leading=14, leftIndent=14, spaceAfter=3, textColor="black",
        ),
        "ref": ParagraphStyle(
            "ref", parent=b["Normal"], fontName="Times-Roman", fontSize=10,
            leading=13, leftIndent=16, firstLineIndent=-16, spaceAfter=5, textColor="black",
        ),
        "code": ParagraphStyle(
            "code", parent=b["Code"], fontName="Courier", fontSize=8.0,
            leading=10.2, leftIndent=3, spaceAfter=6, textColor="black",
        ),
        "cap": ParagraphStyle(
            "cap", parent=b["Normal"], fontName="Times-Italic", fontSize=10,
            leading=12, alignment=TA_CENTER, spaceBefore=4, spaceAfter=8, textColor="black",
        ),
    }


def head(story, s, label):
    story.append(Paragraph(TITLE, s["title"]))
    story.append(Paragraph(label, s["sub"]))


def pdf1():
    path = OUT / "01_Introduction_and_Literature_Survey.pdf"
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    s = S()
    story = []
    head(story, s, "Introduction and Literature Survey")

    story.append(Paragraph("1. Introduction", s["h1"]))
    story.append(Paragraph(
        "Speech Emotion Recognition means finding emotion from a person's voice. "
        "Usually one audio clip gets one label such as happy, angry or neutral. That works "
        "when the clip is short. In a real conversation the mood can change after some time. "
        "If we keep only one label for the full audio, we miss when the emotion changed.",
        s["body"],
    ))
    story.append(Paragraph(
        "If we cut the audio into many small windows and show every prediction, the output "
        "becomes messy. One wrong window makes the emotion jump again and again. That is "
        "hard for a person to read.",
        s["body"],
    ))
    story.append(Paragraph(
        "This project does Continuous Speech Emotion Recognition. Continuous here means we "
        "look at emotion over time, not only once for the whole file. We predict emotion for "
        "each window, remove weak jumps, join same emotions into stable parts, and mark short "
        "transition parts. The final answer looks like a timeline, for example: 0-4 sec Neutral, "
        "4-5 sec Transition, 5-8 sec Anger.",
        s["body"],
    ))
    story.append(Paragraph(
        "This is useful in call centre checks, teaching apps and chatbots where people want "
        "to know when the mood changed. Public talk data like MELD already has emotion changes "
        "across turns, so we use MELD in this project. Other continuous datasets need college "
        "permission and we do not have that.",
        s["body"],
    ))

    story.append(Paragraph("2. Literature Survey", s["h1"]))
    story.append(Paragraph(
        "We studied five related papers. For each paper we write what it does and what it "
        "does not cover for our problem.",
        s["body"],
    ))

    story.append(Paragraph(
        "Paper 1. A Review on Speech Emotion Recognition: A Survey, Recent Advances, "
        "Challenges, and the Influence of Noise [1]",
        s["h2"],
    ))
    story.append(Paragraph(
        "This review talks about SER steps like features, classifiers, deep learning, "
        "datasets and noise. It gives a clear picture of normal SER work.",
        s["body"],
    ))
    story.append(Paragraph("Drawbacks:", s["body"]))
    story.append(Paragraph("- Mostly talks about one emotion label for a clip.", s["bullet"]))
    story.append(Paragraph("- Less talk about emotion changing inside a long conversation.", s["bullet"]))
    story.append(Paragraph("- Does not show how to make a short timeline from window outputs.", s["bullet"]))

    story.append(Paragraph(
        "Paper 2. Speech Emotion Recognition Approaches: A Systematic Review [2]",
        s["h2"],
    ))
    story.append(Paragraph(
        "This paper compares many SER methods. It covers features, models and the usual "
        "ways papers report scores.",
        s["body"],
    ))
    story.append(Paragraph("Drawbacks:", s["body"]))
    story.append(Paragraph("- Scores are for fixed segments using accuracy or F1.", s["bullet"]))
    story.append(Paragraph("- Timeline type output is not discussed.", s["bullet"]))
    story.append(Paragraph("- Change from one emotion to another is not taken as a separate result.", s["bullet"]))

    story.append(Paragraph(
        "Paper 3. Speech Emotion Recognition Based on Multi-Feature Speed Rate and "
        "Rhythmic Information [3]",
        s["h2"],
    ))
    story.append(Paragraph(
        "This paper uses speaking speed and rhythm with other sound features to get better "
        "emotion classification. It shows that how a person speaks matters a lot.",
        s["body"],
    ))
    story.append(Paragraph("Drawbacks:", s["body"]))
    story.append(Paragraph("- Focus is only on better score for one segment.", s["bullet"]))
    story.append(Paragraph("- No step to join many window outputs into stable emotion parts.", s["bullet"]))
    story.append(Paragraph("- No transition marking between emotions.", s["bullet"]))

    story.append(Paragraph(
        "Paper 4. Speech Emotion Recognition Based on Multi-Dimensional Feature Extraction "
        "and Multi-Scale Feature Fusion [4]",
        s["h2"],
    ))
    story.append(Paragraph(
        "This paper takes many speech features and joins them at different scales. This "
        "often helps accuracy because short time detail and longer context both are used.",
        s["body"],
    ))
    story.append(Paragraph("Drawbacks:", s["body"]))
    story.append(Paragraph("- Final answer is still one class for the input segment.", s["bullet"]))
    story.append(Paragraph("- No cleaning step for noisy window predictions.", s["bullet"]))
    story.append(Paragraph("- Continuous emotion timeline is not the aim of the paper.", s["bullet"]))

    story.append(Paragraph(
        "Paper 5. MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in "
        "Conversations [5]",
        s["h2"],
    ))
    story.append(Paragraph(
        "This paper gives the MELD dataset. Dialogues have many speakers and each utterance "
        "has an emotion label. Emotion can change from one turn to the next, so this data "
        "fits conversation work.",
        s["body"],
    ))
    story.append(Paragraph("Drawbacks:", s["body"]))
    story.append(Paragraph("- Label is for the full utterance, not every small part inside it.", s["bullet"]))
    story.append(Paragraph("- The dataset itself does not give timeline or transition steps.", s["bullet"]))
    story.append(Paragraph("- Many baselines also use text; our work uses audio for the timeline.", s["bullet"]))

    story.append(Paragraph(
        "From these papers we can see that most work stops at one label for a segment. Our "
        "project takes window-wise predictions and turns them into a continuous timeline with "
        "stable parts and transitions.",
        s["body"],
    ))

    story.append(Paragraph("References", s["h1"]))
    for r in [
        "[1] A Review on Speech Emotion Recognition: A Survey, Recent Advances, Challenges, "
        "and the Influence of Noise. ScienceDirect review paper.",
        "[2] Speech Emotion Recognition Approaches: A Systematic Review. ScienceDirect "
        "systematic review paper.",
        "[3] Speech Emotion Recognition Based on Multi-Feature Speed Rate and Rhythmic "
        "Information. ScienceDirect research paper.",
        "[4] Speech Emotion Recognition Based on Multi-Dimensional Feature Extraction and "
        "Multi-Scale Feature Fusion. ScienceDirect research paper.",
        "[5] S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria and R. Mihalcea, "
        "MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations, "
        "Proceedings of ACL, pp. 527-536, 2019.",
        "[6] S. Y. Chen, C. C. Hsu, C. C. Kuo and L. W. Ku, EmotionLines: An Emotion Corpus "
        "of Multi-Party Conversations, arXiv:1802.08379, 2018.",
    ]:
        story.append(Paragraph(r, s["ref"]))

    doc.build(story)
    return path


def draw_diagram(img: Path):
    fig, ax = plt.subplots(figsize=(7.0, 10.0), facecolor="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14.6)
    ax.axis("off")
    ax.set_facecolor("white")

    boxes = [
        (5, 13.85, "Speech input", False),
        (5, 12.7, "Pre-processing\n(resample, normalize)", False),
        (5, 11.55, "Feature extraction\n(Mel spectrogram / MFCC)", False),
        (5, 10.4, "CNN + Bi-LSTM", False),
        (
            5, 9.0,
            "[Contribution]\nWindow-wise emotion detection\n"
            "(overlapping windows)",
            True,
        ),
        (5, 7.45, "[Contribution]\nSmoothing weak / jumping predictions", True),
        (5, 6.05, "[Contribution]\nFinding emotion change points", True),
        (5, 4.65, "[Contribution]\nGrouping into stable emotion regions", True),
        (5, 3.25, "[Contribution]\nMarking transition intervals", True),
        (5, 1.8, "[Contribution]\nContinuous emotion timeline output", True),
        (5, 0.55, "Final result: emotion over time", False),
    ]

    drawn = []
    for x, y, text, contrib in boxes:
        w = 6.4
        h = 1.05 if contrib else 0.88
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle="square,pad=0.02",
                linewidth=1.8 if contrib else 1.0,
                edgecolor="black",
                facecolor="white",
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=8.2, family="serif", color="black")
        drawn.append((x, y, h))

    for i in range(len(drawn) - 1):
        x1, y1, h1 = drawn[i]
        x2, y2, h2 = drawn[i + 1]
        ax.annotate(
            "",
            xy=(x2, y2 + h2 / 2 + 0.015),
            xytext=(x1, y1 - h1 / 2 - 0.015),
            arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
        )

    ax.text(
        5, 14.45,
        "[Contribution] boxes = our new work in this project",
        ha="center", va="center", fontsize=8.3, family="serif", color="black",
    )
    fig.savefig(img, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pdf2():
    img = OUT / "block_diagram.png"
    draw_diagram(img)
    path = OUT / "02_Block_Diagram.pdf"
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=1.4 * cm, bottomMargin=1.3 * cm,
    )
    s = S()
    story = []
    head(story, s, "Block Diagram")

    story.append(Paragraph(
        "The diagram below shows our full method. The top boxes are normal SER steps "
        "(pre-processing, features, CNN + Bi-LSTM). The boxes tagged [Contribution] are "
        "our part. First we do window-wise emotion detection. Then we smooth weak jumps, "
        "find real change points, make stable regions, mark transitions, and print a "
        "continuous emotion timeline.",
        s["body"],
    ))
    story.append(Spacer(1, 3))
    story.append(Image(str(img), width=15.2 * cm, height=19.0 * cm))
    story.append(Paragraph("Fig. 1 Block diagram of the proposed system", s["cap"]))

    story.append(Paragraph(
        "About the [Contribution] boxes: window-wise detection means we slide a short window "
        "(example 2 sec with 1 sec hop) over the speech and predict emotion for each window. "
        "Smoothing fixes sudden low-confidence flips. Change points and grouping join nearby "
        "same labels into one stable region. When emotion really changes, a short Transition "
        "interval is put between the old and new emotion. The last contribution box prints "
        "the final timeline for the full speech. Mel features, CNN and Bi-LSTM are known "
        "methods and we use them as support, not as our claim.",
        s["body"],
    ))

    doc.build(story)
    return path


def pdf3():
    path = OUT / "03_Algorithms_and_Dataset.pdf"
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=1.9 * cm, rightMargin=1.9 * cm,
        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
    )
    s = S()
    story = []
    head(story, s, "Algorithms and Dataset")

    story.append(Paragraph("1. Algorithms / Pseudocode", s["h1"]))
    story.append(Paragraph(
        "Below is simple pseudocode for every block in the diagram.",
        s["body"],
    ))

    modules = [
        ("Pre-processing",
         """PREPROCESS(audio):
    x = resample(audio, 16kHz)
    x = normalize(x)
    return x"""),
        ("Feature extraction (Mel / MFCC)",
         """EXTRACT_FEATURES(window):
    mel = mel_spectrogram(window)
    mel = log(mel)
    mel = normalize(mel)
    return mel"""),
        ("CNN + Bi-LSTM",
         """CLASSIFY(mel):
    h = CNN(mel)
    h = BiLSTM(h)
    scores = softmax(Dense(h))
    label = argmax(scores)
    conf = max(scores)
    return label, conf"""),
        ("Window-wise emotion detection [Contribution]",
         """WINDOW_WISE_DETECT(audio):
    for t = 0, 1, 2, ... until audio ends:
        w = audio from t to t+2 sec
        feats = EXTRACT_FEATURES(w)
        label[t], conf[t] = CLASSIFY(feats)
    return all (t, label, conf)"""),
        ("Smoothing weak / jumping predictions [Contribution]",
         """SMOOTH(labels, conf):
    for each window i:
        if labels[i] differs from both neighbours
           and conf[i] is low:
            labels[i] = majority of nearby windows
        else:
            labels[i] = majority of nearby windows
    return labels"""),
        ("Finding emotion change points [Contribution]",
         """FIND_CHANGES(labels):
    changes = empty
    for i = 2 to N:
        if labels[i] != labels[i-1]:
            changes.append(i)
    return changes"""),
        ("Grouping into stable emotion regions [Contribution]",
         """GROUP_STABLE(times, labels):
    start = times[1]
    cur = labels[1]
    regions = empty
    for i = 2 to N:
        if labels[i] != cur:
            regions.append(Stable(start, times[i], cur))
            start = times[i]
            cur = labels[i]
    regions.append(Stable(start, end_time, cur))
    return regions"""),
        ("Marking transition intervals [Contribution]",
         """MARK_TRANSITIONS(regions):
    timeline = empty
    for each next pair of stable regions A then B:
        timeline.add(A)
        timeline.add(Transition from A.emotion to B.emotion)
    timeline.add(last region)
    return timeline"""),
        ("Continuous emotion timeline output [Contribution]",
         """MAKE_TIMELINE(timeline):
    for each item:
        print start-end : emotion (stable)
           or start-end : Transition (old -> new)
    return final timeline"""),
        ("Checking with dataset labels",
         """EVALUATE(window_preds, meld_labels):
    for each utterance with gold label g:
        take window predictions inside that utterance
        p = majority vote
        compare p with g
    report Accuracy and F1"""),
    ]
    for title, code in modules:
        story.append(Paragraph(title, s["h2"]))
        story.append(Preformatted(code, s["code"]))

    story.append(Paragraph("2. Dataset", s["h1"]))
    story.append(Paragraph(
        "We use MELD (Multimodal EmotionLines Dataset). It has Friends TV show dialogues "
        "with emotion labels and can be downloaded without special college permission.",
        s["body"],
    ))
    story.append(Paragraph(
        "Download links:",
        s["body"],
    ))
    story.append(Paragraph(
        "1) Official page: https://affective-meld.github.io/",
        s["bullet"],
    ))
    story.append(Paragraph(
        "2) Raw data: https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz",
        s["bullet"],
    ))
    story.append(Paragraph(
        "3) Alternate raw link: https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz",
        s["bullet"],
    ))
    story.append(Paragraph(
        "4) Labels and readme: https://github.com/declare-lab/MELD",
        s["bullet"],
    ))
    story.append(Paragraph(
        "MELD has about 1433 dialogues and more than 13000 utterances. Each utterance has "
        "audio, text, speaker name and one emotion. The emotion classes are anger, disgust, "
        "fear, joy, neutral, sadness and surprise. Sentiment labels (positive, negative, "
        "neutral) are also given. Train has about 9989 utterances, development about 1109 "
        "and test about 2610.",
        s["body"],
    ))
    story.append(Paragraph(
        "In our system we use the audio part for prediction. The MELD emotion labels are "
        "used to train the model and to check accuracy. The continuous timeline itself is "
        "made by our contribution steps. Because labels are for whole utterances, we compare "
        "the majority of window predictions inside an utterance with its gold label.",
        s["body"],
    ))
    story.append(Paragraph(
        "Reference: S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria and R. Mihalcea, "
        "MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations, "
        "Proceedings of ACL, pp. 527-536, 2019.",
        s["ref"],
    ))

    doc.build(story)
    return path


def main():
    print(pdf1())
    print(pdf2())
    print(pdf3())


if __name__ == "__main__":
    main()
