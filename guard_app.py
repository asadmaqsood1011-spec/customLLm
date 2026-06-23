"""Live demo for HalluGuard: paste a source and an LLM answer, get a verdict.

    python guard_app.py        # http://127.0.0.1:7860

Shows the hallucination score, a supported/unsupported call, and which words in
the answer never appear in the source (a cheap, honest explanation of the call).
"""

import gradio as gr

from guard import Guard

guard = Guard.load()

EXAMPLES = [
    ["The Eiffel Tower is in Paris, France and was completed in 1889.",
     "The Eiffel Tower was completed in 1889."],
    ["The Eiffel Tower is in Paris, France and was completed in 1889.",
     "The Eiffel Tower is located in Berlin, Germany."],
    ["Python is a programming language created by Guido van Rossum in 1991.",
     "Python was created by Guido van Rossum and first released in 1991."],
]


def run(source, answer):
    if not source.strip() or not answer.strip():
        return "Enter both a source and an answer.", ""
    r = guard.check(source, answer)
    pct = r["score"] * 100
    head = (f"### {'✅ Supported' if r['supported'] else '⚠️ Unsupported'}  "
            f"(hallucination score {pct:.1f}%)")
    if r["unsupported_words"]:
        words = ", ".join(f"`{w}`" for w in r["unsupported_words"])
        detail = f"Answer words not found in the source: {words}"
    else:
        detail = "Every content word in the answer appears in the source."
    return head, detail


with gr.Blocks(title="HalluGuard") as demo:
    gr.Markdown(
        "# HalluGuard\n"
        "A 12M-parameter faithfulness detector built from scratch. It flags when "
        "an answer is not supported by its source. Runs locally in a couple of "
        "milliseconds, no API call.")
    source = gr.Textbox(label="Source (what the model was given)", lines=5)
    answer = gr.Textbox(label="Answer (what the model produced)", lines=3)
    btn = gr.Button("Check faithfulness", variant="primary")
    verdict = gr.Markdown()
    detail = gr.Markdown()
    btn.click(run, [source, answer], [verdict, detail])
    gr.Examples(EXAMPLES, [source, answer])


if __name__ == "__main__":
    demo.launch()
