#!/bin/sh
""":"
show_progress=1
for argument do
    case "$argument" in
        -h|--help)
            cat <<'EOF'
usage: bonsai-chat [-h] [--model MODEL] [--tokenizer TOKENIZER]
                   [--decoder DECODER] [--max-tokens MAX_TOKENS]
                   [--threads THREADS] [--no-progress] [--self-test]

Chat with the standalone Bonsai-1.7B decoder (bonsai-lite)

options:
  -h, --help            show this help message and exit
  --model MODEL
  --tokenizer TOKENIZER
  --decoder DECODER
  --max-tokens MAX_TOKENS
  --threads THREADS
  --no-progress         disable the startup progress bar
  --self-test
EOF
            exit 0
            ;;
        --no-progress)
            show_progress=0
            ;;
    esac
done
if [ "$show_progress" -eq 1 ] && [ -t 1 ]; then
    printf '\r[=                           ] Starting runtime'
fi
exec python3 "$0" "$@"
":"""

import argparse
import os
import pickle
import re
import select
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_MODEL = Path("/tmp/m.gguf")
FALLBACK_MODEL = Path("/root/models/Bonsai-1.7B.gguf")
DEFAULT_TOKENIZER = Path("/usr/share/bonsai/tokenizer.pickle")
DEFAULT_DECODER = Path("/usr/bin/bonsai-lite")
# Python-re approximation of the qwen2 pretokenizer: \p{L} -> [^\W\d_],
# \p{N} -> \d (single digits), with the case-insensitive contractions
# expanded into character classes.
PRETOKEN = re.compile(r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|(?:[^\r\n\w]|_)?[^\W\d_]+|\d| ?(?:[^\s\w]|_)+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+", re.UNICODE)
SPECIAL = re.compile(r"(<\|[^|\r\n]*\|>|</?think>)")
EOS_TOKENS = {151643, 151645}
THINK_TOKENS = {151667, 151668}
MAX_CONTEXT = 256


class StartupProgress:
    def __init__(self, enabled=True, width=28):
        self.enabled = enabled
        self.width = width
        self.last_length = 0
        self.last_line = None
        self.pulse_step = 0

    def _write(self, line):
        if not self.enabled:
            return
        if line == self.last_line:
            return
        padding = " " * max(0, self.last_length - len(line))
        print(f"\r{line}{padding}", end="", flush=True)
        self.last_length = len(line)
        self.last_line = line

    def update(self, fraction, label):
        filled = min(self.width, max(0, int(fraction * self.width)))
        bar = "=" * filled + " " * (self.width - filled)
        self._write(f"[{bar}] {label}")

    def pulse(self, label):
        block_width = 5
        travel = self.width - block_width
        cycle = max(1, travel * 2)
        position = self.pulse_step % cycle
        if position > travel:
            position = cycle - position
        bar = " " * position + "=" * block_width + " " * (self.width - position - block_width)
        self._write(f"[{bar}] {label}")
        self.pulse_step += 1

    def finish(self, label="Model ready"):
        if not self.enabled:
            return
        self.update(1.0, label)
        print()


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


def load_tokenizer(path, progress=None):
    if progress is not None:
        progress.update(0.0, "Reading tokenizer")
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    tokens = payload["tokens"]
    token_to_id = payload.get("token_to_id")
    merge_rank = payload.get("merge_rank")
    if token_to_id is None or merge_rank is None:
        # Keep old SD-card tokenizer images usable.
        token_count = len(tokens)
        token_to_id = {}
        for token_id, token in enumerate(tokens):
            token_to_id[token] = token_id
            if progress is not None and token_id % 8192 == 0:
                progress.update(0.05 + 0.20 * token_id / token_count, "Indexing tokenizer tokens")
        merge_rank = {}
        merges = payload["merges"]
        merge_count = len(merges)
        for rank, merge in enumerate(merges):
            left, right = merge.split(" ", 1)
            merge_rank[(left, right)] = rank
            if progress is not None and rank % 8192 == 0:
                progress.update(0.25 + 0.20 * rank / merge_count, "Indexing tokenizer merges")
    if progress is not None:
        progress.update(0.45, "Tokenizer ready")
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
        special_id = token_to_id.get(fragment) if fragment.startswith("<") else None
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
    # Qwen3 ChatML: no BOS token (tokenizer.ggml.add_bos_token = false). The
    # empty think block prefills non-thinking mode so replies start
    # immediately instead of spending generated tokens inside <think>.
    text = ""
    for role, content in messages:
        if role == "assistant":
            content = f"<think>\n\n</think>\n\n{content}"
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return text + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def wait_for_decoder(process, progress):
    if process.stdout is None:
        raise RuntimeError("cannot open decoder output")
    start = time.monotonic()
    while True:
        readable, _, _ = select.select([process.stdout], [], [], 0.1)
        if readable:
            line = process.stdout.readline()
            if line == "":
                status = process.poll()
                raise RuntimeError(f"decoder exited before becoming ready (status {status})")
            if line.split() == ["READY"]:
                progress.finish()
                return
            raise RuntimeError(f"unexpected decoder startup message: {line.strip()}")
        elapsed = time.monotonic() - start
        progress.pulse(f"Loading model ({elapsed:.1f}s)")


def run_chat(args, tokens, token_to_id, merge_rank, progress):
    progress.update(0.50, "Starting model decoder")
    process = subprocess.Popen(
        [str(args.decoder), "--model", str(args.model), "--server",
         "--threads", str(args.threads)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("cannot open decoder pipes")
    wait_for_decoder(process, progress)
    decoder = byte_decoder()
    messages = []
    noun = "worker" if args.threads == 1 else "workers"
    print(f"Bonsai-1.7B standalone chat ({args.threads} {noun}). Commands: /clear, /quit")
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
            if len(ids) + args.max_tokens >= MAX_CONTEXT:
                messages.pop()
                print("context limit reached; use /clear")
                continue
            request = f"RUN {args.max_tokens} {len(ids)} " + " ".join(map(str, ids)) + "\n"
            process.stdin.write(request)
            process.stdin.flush()
            assistant = bytearray()
            eval_ns = []
            prompt_tokens = 0
            cached_tokens = 0
            prompt_ns = 0
            print("bonsai> ", end="", flush=True)
            while True:
                line = process.stdout.readline().strip()
                if line == "DONE":
                    break
                kind, *fields = line.split()
                if kind == "PROMPT":
                    if len(fields) == 2:
                        prompt_tokens, prompt_ns = map(int, fields)
                    else:
                        prompt_tokens, cached_tokens, prompt_ns = map(int, fields)
                elif kind == "TOKEN":
                    token = int(fields[0])
                    if token not in EOS_TOKENS and token not in THINK_TOKENS:
                        data = decode_piece(tokens[token], decoder)
                        assistant.extend(data)
                        print(data.decode("utf-8", errors="replace"), end="", flush=True)
                elif kind == "EVAL":
                    eval_ns.append(int(fields[0]))
            print()
            messages.append(("assistant", assistant.decode("utf-8", errors="replace")))
            prefill_tps = prompt_tokens * 1e9 / prompt_ns if prompt_ns else 0.0
            decode_tps = len(eval_ns) * 1e9 / sum(eval_ns) if eval_ns else 0.0
            cache_note = f", {cached_tokens} prompt tokens cached" if cached_tokens else ""
            print(f"[{prefill_tps:.3f} prompt tok/s{cache_note}, {decode_tps:.3f} decode tok/s]")
    finally:
        if process.poll() is None:
            process.stdin.write("QUIT\n")
            process.stdin.flush()
            process.wait(timeout=5)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="bonsai-chat", description="Chat with the standalone Bonsai-1.7B decoder")
    default_model = DEFAULT_MODEL if DEFAULT_MODEL.exists() or not FALLBACK_MODEL.exists() else FALLBACK_MODEL
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-progress", action="store_true", help="disable the startup progress bar")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    # Parse first so --help and malformed commands never touch the tokenizer,
    # model mapping, or decoder process.
    args = parse_args(argv)
    progress = StartupProgress(enabled=not args.no_progress and sys.stdout.isatty())
    start = time.monotonic()
    tokens, token_to_id, merge_rank = load_tokenizer(args.tokenizer, progress)
    if args.self_test:
        expected = [12522, 5193, 264, 882]
        actual = tokenize("Once upon a time", token_to_id, merge_rank)
        if actual != expected:
            raise SystemExit(f"tokenizer mismatch\nexpected: {expected}\nactual:   {actual}")
        prompt = render_prompt([("user", "Hello")])
        chat = tokenize(prompt, token_to_id, merge_rank)
        expected_chat = [151644, 872, 198, 9707, 151645, 198, 151644, 77091,
                         198, 151667, 271, 151668, 271]
        if chat != expected_chat:
            raise SystemExit(f"chat tokenizer mismatch\nexpected: {expected_chat}\nactual:   {chat}")
        continued = tokenize(
            render_prompt([
                ("user", "Hello"),
                ("assistant", "Hi there"),
                ("user", "Again"),
            ]),
            token_to_id,
            merge_rank,
        )
        if continued[:len(chat)] != chat:
            raise SystemExit("continued chat does not preserve the cached token prefix")
        progress.finish("Tokenizer ready")
        print(f"tokenizer self-test passed in {time.monotonic() - start:.3f} seconds")
        return
    run_chat(args, tokens, token_to_id, merge_rank, progress)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("", file=sys.stderr)
