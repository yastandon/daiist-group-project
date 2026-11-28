"""
The app: an App tab and a Monitoring tab. `uv run python app.py`

Keep the logic in pipeline.py (or your own modules); this file only wires it to
the interface and the logs. Keep `serve()` and the `/run` endpoint: check.py
calls `serve()` locally and `/run` on your deployed app, with one input from
eval/cases.csv (text, or a file path).
"""

import os
import time

import gradio as gr

import env
import logbook
import monitor
import pipeline

env.load_env()
LOG_TARGET = logbook.start()

TITLE = "Damage Claim Triage"
DESCRIPTION = (
    "For claims agents: upload the customer's photo of the damaged parcel and their message, "
    "and get a drafted decision (approve / deny / escalate) with a reply. Uncertain or "
    "high-value claims are sent to a person."
)


def serve(raw_input: str, claim_value_eur: float | None = None) -> dict:
    """Run one request through the pipeline, log it, and return the logged record.

    `raw_input` is one string: a photo path, `photo path | claim text`, or text only
    (the convention of pipeline.py and eval/cases.csv). check.py calls this directly.
    """
    started = time.perf_counter()
    result = pipeline.run(raw_input, claim_value_eur=claim_value_eur)
    return logbook.log_request(result | {"latency_s": round(time.perf_counter() - started, 3)})


# --- App tab -----------------------------------------------------------------

def _present(record: dict) -> tuple:
    a, b = record["outputs"]["component_a"], record["outputs"]["component_b"]
    status = "⚠️ **Sent to a person for review.**" if record["review"] else "✅ Handled automatically."
    decision = f"**Decision: {b['decision'].upper()}**  \n{b.get('reason', '')}"
    return (
        a,
        decision,
        b["reply"],
        f"{status}  Latency {record['latency_s']} s, cost €{record['cost_eur']:.4f}.",
        record["id"],
    )


def on_run(photo, claim, value):
    if photo is None and not str(claim or "").strip():
        raise gr.Error("Upload a photo of the parcel (and ideally the customer's message) first.")
    combined = f"{photo} | {claim or ''}" if photo else str(claim)
    return _present(serve(combined, claim_value_eur=float(value) if value else None))


def on_run_api(raw_input):
    """The /run endpoint check.py calls, with one input string from eval/cases.csv."""
    if raw_input is None or not str(raw_input).strip():
        raise gr.Error("Give me an input first.")
    return serve(str(raw_input))


def on_feedback(request_id, value):
    if not request_id:
        gr.Warning("Run a request first.")
        return
    logbook.log_feedback(request_id, value)
    gr.Info("Thanks for the feedback.")


# --- Monitoring tab ----------------------------------------------------------

def on_refresh():
    logs = logbook.load()
    reference = monitor.load_reference()
    fired = monitor.alerts(logs, reference) if not logs.empty else []
    if reference is None:
        alert_text = "No reference set yet: run `uv run python main.py reference` and commit `monitoring/`."
    elif fired:
        alert_text = "### 🚨 Alerts\n" + "\n".join(f"- {message}" for message in fired)
    else:
        alert_text = "### ✅ No alerts"
    if reference is not None and len(logs) >= monitor.MIN_WINDOW:
        drift_table = monitor.drift(reference, monitor.live_features(logs))
    else:
        drift_table = monitor.drift(reference, reference.head(0)) if reference is not None else None
    return alert_text, monitor.summary(logs), monitor.quality(logs), drift_table


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}\n{DESCRIPTION}")
    if missing := env.missing():
        gr.Markdown(f"⚠️ **Missing environment variables:** {', '.join(missing)}. See SETUP.md.")

    with gr.Tab("App"):
        request_id = gr.State("")
        with gr.Row():
            with gr.Column():
                photo = gr.Image(label="Photo of the parcel", type="filepath", height=300)
                claim = gr.Textbox(label="Customer's message", lines=3,
                                   placeholder="e.g. The box arrived crushed and the mug inside is broken.")
                value = gr.Number(label="Order value (€, optional)", value=None, precision=2)
                run_button = gr.Button("Assess claim", variant="primary")
            with gr.Column():
                out_decision = gr.Markdown(label="Decision")
                out_reply = gr.Textbox(label="Reply to the customer", lines=5)
                status = gr.Markdown()
                out_a = gr.JSON(label="What the vision model saw")
                with gr.Row():
                    up = gr.Button("👍")
                    down = gr.Button("👎")
        run_button.click(on_run, [photo, claim, value],
                         [out_a, out_decision, out_reply, status, request_id], api_name=False)
        up.click(lambda rid: on_feedback(rid, +1), request_id, None, api_name=False)
        down.click(lambda rid: on_feedback(rid, -1), request_id, None, api_name=False)

        # The /run endpoint for check.py: one input string, the whole logged record back.
        # The row is invisible but the endpoint is real.
        with gr.Row(visible=False):
            api_in = gr.Textbox()
            api_out = gr.JSON()
            api_button = gr.Button()
            api_button.click(on_run_api, api_in, api_out, api_name="run")

    with gr.Tab("Monitoring"):
        refresh = gr.Button("Refresh")
        alert_box = gr.Markdown()
        summary_table = gr.Dataframe(label="Summary", interactive=False)
        quality_plot = gr.LinePlot(x="time", y="value", color="signal", label="Quality over time", height=300)
        drift_table = gr.Dataframe(label=f"Drift: PSI of the last {monitor.WINDOW} requests against the reference set", interactive=False)
        gr.Markdown(f"<small>{LOG_TARGET}</small>")
        outputs = [alert_box, summary_table, quality_plot, drift_table]
        refresh.click(on_refresh, None, outputs, api_name=False)
        demo.load(on_refresh, None, outputs, api_name=False)


if __name__ == "__main__":
    # Render gives the port in $PORT; locally it is 7860.
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
