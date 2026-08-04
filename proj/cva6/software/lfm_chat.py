#!/usr/bin/env python3

import argparse
import os
import pickle
import re
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_MODEL = Path("/root/models/LFM2.5-350M-Q4_0.gguf")
DEFAULT_TOKENIZER = Path("/usr/share/lfm/tokenizer.pickle")
DEFAULT_DECODER = Path("/usr/bin/lfm")
PRETOKEN = re.compile(r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|(?:[^\r\n\w]|_)?[^\W\d_]+|\d{1,3}| ?(?:[^\s\w]|_)+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", re.UNICODE)
SPECIAL = re.compile(r"(<\|[^|\r\n]*\|>)")


def byte_encoder():
    values = list(range(ord("!"), ord("~") + 1))
    values += list(range(ord("¡"), ord("¬") + 1))
    values += list(range(ord("®"), ord("ÿ") + 1))
    chars = values[:]
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(256 + extra)
            extra += 1
    return {value: chr(char) for value, char in zip(values, chars)}


def load_tokenizer(path):
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    tokens = payload["tokens"]
    token_to_id = {token: token_id for token_id, token in enumerate(tokens)}
    merge_rank = {}
    for rank, merge in enumerate(payload["merges"]):
        left, right = merge.split(" ", 1)
        merge_rank[(left, right)] = rank
    return tokens, token_to_id, merge_rank


def pretokenize(text):
    offset = 0
    for match in PRETOKEN.finditer(text):
        if match.start() != offset:
            yield text[offset:match.start()]
        yield match.group(0)
        offset = match.end()
    if offset != len(text):
        yield text[offset:]


def bpe(encoded, token_to_id, merge_rank):
    direct = token_to_id.get(encoded)
    if direct is not None:
        return [direct]
    symbols = list(encoded)
    while len(symbols) > 1:
        best_rank = None
        best_index = None
        for index in range(len(symbols) - 1):
            rank = merge_rank.get((symbols[index], symbols[index + 1]))
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best_index = index
        if best_index is None:
            break
        symbols[best_index:best_index + 2] = [symbols[best_index] + symbols[best_index + 1]]
    try:
        return [token_to_id[symbol] for symbol in symbols]
    except KeyError as error:
        raise ValueError(f"tokenizer has no token for {error.args[0]!r}") from error


def tokenize(text, token_to_id, merge_rank):
    encoder = byte_encoder()
    output = []
    for fragment in SPECIAL.split(text):
        if not fragment:
            continue
        special_id = token_to_id.get(fragment) if fragment.startswith("<|") else None
        if special_id is not None:
            output.append(special_id)
            continue
        for piece in pretokenize(fragment):
            encoded = "".join(encoder[value] for value in piece.encode("utf-8"))
            output.extend(bpe(encoded, token_to_id, merge_rank))
    return output


def byte_decoder():
    return {character: value for value, character in byte_encoder().items()}


def decode_piece(piece, decoder):
    if piece.startswith("<|") and piece.endswith("|>"):
        return b""
    return bytes(decoder.get(character, ord("?")) for character in piece)


def render_prompt(messages):
    text = "<|startoftext|>"
    for role, content in messages:
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return text + "<|im_start|>assistant\n"


def run_chat(args, tokens, token_to_id, merge_rank):
    process = subprocess.Popen(
        [str(args.decoder), str(args.model), "--server", "--threads", str(args.threads)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("cannot open decoder pipes")
    if process.stdout.readline().strip() != "READY":
        raise RuntimeError("decoder did not become ready")
    decoder = byte_decoder()
    messages = []
    print(f"LFM2.5-350M standalone chat ({args.threads} worker). Commands: /clear, /quit")
    print("FPGA-accelerated generation; measured prompt and decode rates are printed after each reply.")
    try:
        while True:
            user = input("\nyou> ").strip()
            if user == "/quit":
                break
            if user == "/clear":
                messages.clear()
                print("history cleared")
                continue
            if not user:
                continue
            messages.append(("user", user))
            ids = tokenize(render_prompt(messages), token_to_id, merge_rank)
            if len(ids) + args.max_tokens >= 256:
                messages.pop()
                print("context limit reached; use /clear")
                continue
            request = f"RUN {args.max_tokens} {len(ids)} " + " ".join(map(str, ids)) + "\n"
            process.stdin.write(request)
            process.stdin.flush()
            assistant = bytearray()
            eval_ns = []
            prompt_tokens = 0
            prompt_ns = 0
            print("lfm> ", end="", flush=True)
            while True:
                line = process.stdout.readline().strip()
                if line == "DONE":
                    break
                kind, *fields = line.split()
                if kind == "PROMPT":
                    prompt_tokens, prompt_ns = map(int, fields)
                elif kind == "TOKEN":
                    token = int(fields[0])
                    if token != 7:
                        data = decode_piece(tokens[token], decoder)
                        assistant.extend(data)
                        print(data.decode("utf-8", errors="replace"), end="", flush=True)
                elif kind == "EVAL":
                    eval_ns.append(int(fields[0]))
            print()
            messages.append(("assistant", assistant.decode("utf-8", errors="replace")))
            prefill_tps = prompt_tokens * 1e9 / prompt_ns if prompt_ns else 0.0
            decode_tps = len(eval_ns) * 1e9 / sum(eval_ns) if eval_ns else 0.0
            print(f"[{prefill_tps:.3f} prompt tok/s, {decode_tps:.3f} decode tok/s]")
    finally:
        if process.poll() is None:
            process.stdin.write("QUIT\n")
            process.stdin.flush()
            process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description="Chat with the standalone LFM2.5-350M decoder")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--threads", type=int, default=max(1, min(os.cpu_count() or 1, 1)))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    start = time.monotonic()
    tokens, token_to_id, merge_rank = load_tokenizer(args.tokenizer)
    if args.self_test:
        prompt = "<|startoftext|><|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        expected = [1, 6, 6423, 708, 36309, 7, 708, 6, 64015, 708]
        actual = tokenize(prompt, token_to_id, merge_rank)
        if actual != expected:
            raise SystemExit(f"tokenizer mismatch\nexpected: {expected}\nactual:   {actual}")
        print(f"tokenizer self-test passed in {time.monotonic() - start:.3f} seconds")
        return
    run_chat(args, tokens, token_to_id, merge_rank)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("", file=sys.stderr)
