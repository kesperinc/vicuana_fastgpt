import argparse
from collections import defaultdict
import datetime
import json
import os
import time

import gradio as gr
import requests

from chatserver.conversation import (default_conversation, conv_templates,
    SeparatorStyle)
from chatserver.constants import LOGDIR
from chatserver.utils import build_logger, server_error_msg
from chatserver.serve.gradio_patch import Chatbot as grChatbot


logger = build_logger("gradio_web_server", "gradio_web_server.log")

upvote_msg = "👍  Upvote the last response"
downvote_msg = "👎  Downvote the last response"

priority = {
}

code_highlight_css = (
"""
#chatbot .hll { background-color: #ffffcc }
#chatbot .c { color: #408080; font-style: italic }
#chatbot .err { border: 1px solid #FF0000 }
#chatbot .k { color: #008000; font-weight: bold }
#chatbot .o { color: #666666 }
#chatbot .ch { color: #408080; font-style: italic }
#chatbot .cm { color: #408080; font-style: italic }
#chatbot .cp { color: #BC7A00 }
#chatbot .cpf { color: #408080; font-style: italic }
#chatbot .c1 { color: #408080; font-style: italic }
#chatbot .cs { color: #408080; font-style: italic }
#chatbot .gd { color: #A00000 }
#chatbot .ge { font-style: italic }
#chatbot .gr { color: #FF0000 }
#chatbot .gh { color: #000080; font-weight: bold }
#chatbot .gi { color: #00A000 }
#chatbot .go { color: #888888 }
#chatbot .gp { color: #000080; font-weight: bold }
#chatbot .gs { font-weight: bold }
#chatbot .gu { color: #800080; font-weight: bold }
#chatbot .gt { color: #0044DD }
#chatbot .kc { color: #008000; font-weight: bold }
#chatbot .kd { color: #008000; font-weight: bold }
#chatbot .kn { color: #008000; font-weight: bold }
#chatbot .kp { color: #008000 }
#chatbot .kr { color: #008000; font-weight: bold }
#chatbot .kt { color: #B00040 }
#chatbot .m { color: #666666 }
#chatbot .s { color: #BA2121 }
#chatbot .na { color: #7D9029 }
#chatbot .nb { color: #008000 }
#chatbot .nc { color: #0000FF; font-weight: bold }
#chatbot .no { color: #880000 }
#chatbot .nd { color: #AA22FF }
#chatbot .ni { color: #999999; font-weight: bold }
#chatbot .ne { color: #D2413A; font-weight: bold }
#chatbot .nf { color: #0000FF }
#chatbot .nl { color: #A0A000 }
#chatbot .nn { color: #0000FF; font-weight: bold }
#chatbot .nt { color: #008000; font-weight: bold }
#chatbot .nv { color: #19177C }
#chatbot .ow { color: #AA22FF; font-weight: bold }
#chatbot .w { color: #bbbbbb }
#chatbot .mb { color: #666666 }
#chatbot .mf { color: #666666 }
#chatbot .mh { color: #666666 }
#chatbot .mi { color: #666666 }
#chatbot .mo { color: #666666 }
#chatbot .sa { color: #BA2121 }
#chatbot .sb { color: #BA2121 }
#chatbot .sc { color: #BA2121 }
#chatbot .dl { color: #BA2121 }
#chatbot .sd { color: #BA2121; font-style: italic }
#chatbot .s2 { color: #BA2121 }
#chatbot .se { color: #BB6622; font-weight: bold }
#chatbot .sh { color: #BA2121 }
#chatbot .si { color: #BB6688; font-weight: bold }
#chatbot .sx { color: #008000 }
#chatbot .sr { color: #BB6688 }
#chatbot .s1 { color: #BA2121 }
#chatbot .ss { color: #19177C }
#chatbot .bp { color: #008000 }
#chatbot .fm { color: #0000FF }
#chatbot .vc { color: #19177C }
#chatbot .vg { color: #19177C }
#chatbot .vi { color: #19177C }
#chatbot .vm { color: #19177C }
#chatbot .il { color: #666666 }
""")
#.highlight  { background: #f8f8f8; }

def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(args.controller_url + "/refresh_status")
    assert ret.status_code == 200
    ret = requests.post(args.controller_url + "/list_models")
    models = ret.json()["models"]
    models.sort(key=lambda x: priority.get(x, x))
    logger.info(f"Models: {models}")
    return models


def load_demo(request: gr.Request):
    logger.info(f"load demo: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    return (state,
            gr.Dropdown.update(
               choices=models,
               visible=True,
               value=models[0] if len(models) > 0 else ""),
            gr.Chatbot.update(visible=True),
            gr.Textbox.update(visible=True),
            gr.Button.update(visible=True),
            gr.Button.update(visible=True),
            gr.Button.update(visible=True),
            gr.Button.update(visible=True),
            gr.Button.update(visible=True),
            gr.Accordion.update(visible=True))


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    logger.info(f"vote_type: {vote_type}")
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, upvote_btn, downvote_btn, model_selector,
                         request: gr.Request):
    if len(state.messages) == state.offset:
        return upvote_btn, downvote_msg, ""
    if upvote_btn == "done":
        return "done", "done", ""
    vote_last_response(state, "upvote", model_selector, request)
    return "done", "done", ""


def downvote_last_response(state, upvote_btn, downvote_btn, model_selector,
                           request: gr.Request):
    if len(state.messages) == state.offset:
        return upvote_btn, downvote_msg, ""
    if upvote_btn == "done":
        return "done", "done", ""
    vote_last_response(state, "downvote", model_selector, request)
    return "done", "done", ""


def regenerate(state):
    if len(state.messages) == state.offset:
        # skip empty "Regenerate"
        return state, state.to_gradio_chatbot(), "", upvote_msg, downvote_msg

    state.messages[-1][-1] = None
    return state, state.to_gradio_chatbot(), "", upvote_msg, downvote_msg


def clear_history():
    state = default_conversation.copy()
    return state, state.to_gradio_chatbot(), ""


def add_text(state, text, request: gr.Request):
    text = text[:1536]  # Hard cut-off
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    return state, state.to_gradio_chatbot(), "", upvote_msg, downvote_msg


def broadcast_input(*args):
    print("LEN OF ARGS:", len(args))
    states = list(args[:-1])
    text = args[-1]

    text = text[:1536]  # Hard cut-off
    chatbots = []
    upvote = []
    downvote = []
    for i in range(len(states)):
        states[i].append_message(states[i].roles[0], text)
        states[i].append_message(states[i].roles[1], None)
        chatbots.append(states[i].to_gradio_chatbot())
        upvote.append(upvote_msg)
        downvote.append(downvote_msg)
    return states + chatbots + [""] + upvote + downvote


def http_bot(state, model_selector, temperature, max_new_tokens, request: gr.Request):
    start_tstamp = time.time()

    if len(state.messages) == state.offset:
        # Skip empty "Regenerate"
        yield state, state.to_gradio_chatbot()
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        if "bair-chat" in model_selector: # Hardcode the condition
            template_name = "bair_v1"
        else:
            template_name = "v1"
        new_state = conv_templates[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = args.controller_url
    ret = requests.post(controller_url + "/get_worker_address",
            json={"model_name": model_selector})
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_selector}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield state, state.to_gradio_chatbot()
        return

    # Construct prompt
    prompt = state.get_prompt()

    # Make requests
    headers = {"User-Agent": "Client"}
    pload = {
        "prompt": prompt,
        "temperature": float(temperature),
        "max_new_tokens": int(max_new_tokens),
        "stop": state.sep if state.sep_style == SeparatorStyle.SINGLE else state.sep2,
    }
    logger.info(f"==== request ====\n{pload}")
    response = requests.post(worker_addr + "/generate_stream",
        headers=headers, json=pload, stream=True)

    # Stream output
    for chunk in response.iter_lines(chunk_size=8192, decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode("utf-8"))
            if data["error_code"] == 0:
                output = data["text"][len(prompt) + 2:]
                state.messages[-1][-1] = output
                yield state, state.to_gradio_chatbot()
            else:
                output = data["text"]
                state.messages[-1][-1] = output
                yield state, state.to_gradio_chatbot()

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_selector,
            "start": round(start_tstamp, 4),
            "finish": round(start_tstamp, 4),
            "state": state.dict(),
            "ip": request.client.host if request is not None else None,
        }
        fout.write(json.dumps(data) + "\n")


def http_bot_all(*args):
    max_num_model = (len(args) - 2) // 2
    states = list(args[: max_num_model])
    model_selectors = list(args[max_num_model : max_num_model * 2])
    temperature = args[max_num_model * 2]
    max_new_tokens = args[max_num_model * 2 + 1]
    assert max_num_model * 2 + 2 == len(args)

    gen = []
    for i in range(max_num_model):
        gen.append(http_bot(states[i], model_selectors[i], temperature, max_new_tokens, None))

    chatbots = [None] * max_num_model
    while True:
        flag = False
        for i in range(max_num_model):
            print("!" * 20, i)
            try:
                states[i], chatbots[i] = next(gen[i])
                flag = True
            except StopIteration:
                pass
        yield states + chatbots
        if not flag: return


def build_demo():
    css = (
        """
        """ + code_highlight_css)

    with gr.Blocks(title="Chat Server", theme=gr.themes.Soft(), css=css) as demo:
        notice = gr.Markdown(
            """
            # Chat server\n
            ### Terms of Use\n
            By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It does not provide safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research.\n
            ### Choose a model to chat with
            """
        )
        max_num_model = 2
        state = [gr.State(), gr.State()]
        model_selector = [None] * max_num_model
        chatbot = [None] * max_num_model
        human_input = [None] * max_num_model
        upvote_btn = [None] * max_num_model
        downvote_btn = [None] * max_num_model
        regenerate_btn = [None] * max_num_model
        clear_btn = [None] * max_num_model
        broadcast_btn = [None] * max_num_model

        with gr.Row():
            with gr.Column():
                model_selector[0] = gr.Dropdown(
                    interactive=True,
                    show_label=False)
            with gr.Column():
                model_selector[1] = gr.Dropdown(
                    interactive=True,
                    show_label=False,
                    placeholder="Add a model to compare with")

        with gr.Row():
            for i in range(max_num_model):
                with gr.Column():
                    chatbot[i] = grChatbot(visible=False).style(height=550)

        with gr.Row():
            for i in range(max_num_model):
                with gr.Column(scale=4, min_width=0):
                    human_input[i] = gr.Textbox(show_label=False,
                        placeholder="Enter text and press ENTER", visible=False)
                with gr.Column(scale=1, min_width=0):
                    broadcast_btn[i] = gr.Button(value="broadcast", visible=False)

        with gr.Row():
            for i in range(max_num_model):
                with gr.Column(scale=1, min_width=0):
                    upvote_btn[i] = gr.Button(value=upvote_msg, visible=False)
                with gr.Column(scale=1, min_width=0):
                    downvote_btn[i] = gr.Button(value=downvote_msg, visible=False)
                with gr.Column(scale=1, min_width=0):
                    regenerate_btn[i] = gr.Button(value="Regenerate", visible=False)
                with gr.Column(scale=1, min_width=0):
                    clear_btn[i] = gr.Button(value="Clear history", visible=False)

        with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
            temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Temperature",)
            max_output_tokens = gr.Slider(minimum=0, maximum=1024, value=512, step=64, interactive=True, label="Max output tokens",)

        for i in range(max_num_model):
            upvote_btn[i].click(upvote_last_response,
                [state[i], upvote_btn[i], downvote_btn[i], model_selector[i]],
                [upvote_btn[i], downvote_btn[i], human_input[i]])
            downvote_btn[i].click(downvote_last_response,
                [state[i], upvote_btn[i], downvote_btn[i], model_selector[i]],
                [upvote_btn[i], downvote_btn[i], human_input[i]])
            regenerate_btn[i].click(regenerate, state[i],
                [state[i], chatbot[i], human_input[i], upvote_btn[i], downvote_btn[i]]).then(
                http_bot, [state[i], model_selector[i], temperature, max_output_tokens],
                [state[i], chatbot[i]])
            clear_btn[i].click(clear_history, None, [state[i], chatbot[i], human_input[i]])

            human_input[i].submit(add_text, [state[i], human_input[i]],
                [state[i], chatbot[i], human_input[i], upvote_btn[i], downvote_btn[i]]).then(
                http_bot, [state[i], model_selector[i], temperature, max_output_tokens],
                [state[i], chatbot[i]])

            broadcast_btn[i].click(broadcast_input, state + [human_input[i]],
                state + chatbot + [human_input[i]] + 
                upvote_btn + downvote_btn).then(
                http_bot_all,
                state + model_selector + [temperature, max_output_tokens],
                state + chatbot)
    
            demo.load(load_demo, None, 
                [state[i], model_selector[i], chatbot[i], human_input[i], broadcast_btn[i],
                 upvote_btn[i], downvote_btn[i], regenerate_btn[i], clear_btn[i],
                 parameter_row])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=2)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_demo()
    demo.queue(concurrency_count=args.concurrency_count, status_update_rate=10).launch(
        server_name=args.host, server_port=args.port, show_api=False, share=args.share)
