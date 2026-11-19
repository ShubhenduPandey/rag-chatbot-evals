"""Local Markdown semantic retrieval. Only embedding requests leave this machine."""

from dataclasses import dataclass
from contextlib import closing
from hashlib import sha256
from pathlib import Path
import json
import math
import re
import sqlite3
import threading

BASE_DIR = Path(__file__).resolve().parent
CONTEXT_DIR = BASE_DIR / "knowledge_base"
CACHE_PATH = BASE_DIR / ".rag_cache" / "index.sqlite3"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 4
MIN_SIMILARITY = 0.35   # cosine score, not a probability; calibrate with evaluate_rag.py
MAX_CONTEXT_CHARS = 10000
CHUNK_CHARS = 1400
FOLLOW_UP = re.compile(
    r"\b(that|those|these|it|them|same|another|instead|again|previous|last|"
    r"more detail|example|what about|how about)\b", re.I)


@dataclass(frozen=True)
class Chunk:
    source: str
    heading: str
    part: int
    text: str

    @property
    def key(self):
        return sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Match:
    chunk: Chunk
    score: float


def split_markdown(text, source):
    """Keep each heading's ancestry and body; ignore heading markers inside fences.

    Oversized sections split at line boundaries, reopening fences in continuations.
    Bounds (1400 body + 400 heading chars) keep UTF-8 bytes below the model's
    per-input token limit even for non-English text, without another tokenizer.
    """
    sections, headings, body = [], [], []
    fence = None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^\s{0,3}(`{3,}|~{3,})(.*)$", line)
        heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading and fence is None:
            if ''.join(body).strip():
                sections.append((list(headings), ''.join(body)))
            body = []
            level, title = len(heading[1]), heading[2]
            headings = [(n, t) for n, t in headings if n < level]
            headings.append((level, title))
        else:
            body.append(line)
        if marker:
            if fence is None:
                fence = marker[1]
            elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
    if ''.join(body).strip():
        sections.append((list(headings), ''.join(body)))

    chunks = []
    for ancestry, content in sections:
        title = " > ".join(t for _, t in ancestry) or Path(source).stem
        title = title[-400:]
        prefix = title + "\n\n"
        current, active, part = '', None, 1
        for line in content.splitlines(keepends=True):
            # Pathological single lines are bounded too; never discard their tail.
            for start in range(0, len(line), CHUNK_CHARS):
                piece = line[start:start + CHUNK_CHARS]
                if current and len(current) + len(piece) > CHUNK_CHARS:
                    suffix = '\n' + active[0] if active else ''
                    chunks.append(Chunk(source, title, part, prefix + current.rstrip() + suffix))
                    part += 1
                    current = active[0] + active[1] + '\n' if active else ''
                current += piece
                marker = re.match(r"^\s{0,3}(`{3,}|~{3,})(.*)$", piece)
                if marker:
                    if active is None:
                        active = (marker[1], marker[2])
                    elif marker[1][0] == active[0][0] and len(marker[1]) >= len(active[0]) and not marker[2].strip():
                        active = None
        if current.strip():
            suffix = '\n' + active[0] if active else ''
            chunks.append(Chunk(source, title, part, prefix + current.rstrip() + suffix))
    return chunks


def retrieval_query(custom, history):
    """Use the latest human turn; attach context only for referential follow-ups."""
    history = list(history)
    if custom:
        question = custom.strip()
        if history and history[-1].removeprefix('Me: (typed) ').strip() == question:
            history = history[:-1]
    else:
        humans = [i for i, line in enumerate(history) if line.startswith(('Me:', 'Them:'))]
        if not humans:
            return ''
        last = humans[-1]
        question = history[last].split(':', 1)[1].removeprefix(' (typed)').strip()
        history = history[:last]
    if FOLLOW_UP.search(question):
        context = '\n'.join(history[-4:])[-1200:]
        if context:
            question = f"Current question: {question}\nRecent conversation for reference: {context}"
    # At most 6000 UTF-8 bytes, so safely below 8192 tokens for any language.
    return question.encode('utf-8')[:6000].decode('utf-8', errors='ignore')


def normalized(vector):
    values = [float(x) for x in vector]
    norm = math.sqrt(sum(x * x for x in values))
    if not values or not math.isfinite(norm) or norm == 0:
        raise ValueError('Invalid embedding vector')
    return [x / norm for x in values]


class SemanticIndex:
    def __init__(self, client, context_dir=CONTEXT_DIR, cache_path=CACHE_PATH,
                 model=EMBEDDING_MODEL):
        self.client = client
        self.context_dir = Path(context_dir)
        self.cache_path = Path(cache_path)
        self.model = model
        self.lock = threading.RLock()
        self.chunks, self.vectors = [], {}
        self.last_stats = {}

    def _embed(self, texts):
        result = self.client.embeddings.create(model=self.model, input=texts,
                                                encoding_format='float')
        ordered = sorted(result.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(texts))):
            raise ValueError('Incomplete embedding response')
        return [normalized(item.embedding) for item in ordered]

    def refresh(self):
        """Rescan files; reuse content-addressed vectors and prune deleted chunks."""
        with self.lock:
            self.context_dir.mkdir(parents=True, exist_ok=True)
            chunks = []
            for path in sorted(self.context_dir.rglob('*')):
                if path.is_file() and path.suffix.lower() == '.md' and not path.is_symlink():
                    if not path.resolve().is_relative_to(self.context_dir.resolve()):
                        continue
                    chunks.extend(split_markdown(path.read_text(encoding='utf-8-sig'),
                                                 path.relative_to(self.context_dir).as_posix()))
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.cache_path)) as db:
                db.execute('CREATE TABLE IF NOT EXISTS embeddings '
                           '(model TEXT, hash TEXT, vector TEXT, PRIMARY KEY(model, hash))')
                vectors = {}
                for key, raw in db.execute('SELECT hash, vector FROM embeddings WHERE model=?', (self.model,)):
                    try:
                        vectors[key] = normalized(json.loads(raw))
                    except (TypeError, ValueError):
                        pass  # repair a damaged entry by embedding it again
                unique = {chunk.key: chunk for chunk in chunks}
                missing = [chunk for key, chunk in unique.items() if key not in vectors]
                for start in range(0, len(missing), 32):
                    batch = missing[start:start + 32]
                    embeddings = self._embed([chunk.text for chunk in batch])
                    for chunk, vector in zip(batch, embeddings):
                        vectors[chunk.key] = vector
                        db.execute('INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)',
                                   (self.model, chunk.key, json.dumps(vector)))
                    db.commit()  # completed batches survive a later request failure
                for key in vectors.keys() - unique.keys():
                    db.execute('DELETE FROM embeddings WHERE model=? AND hash=?', (self.model, key))
                db.commit()
            active_vectors = {key: vectors[key] for key in unique}
            if len({len(v) for v in active_vectors.values()}) > 1:
                raise ValueError('Embedding dimensions differ; rebuild the local cache')
            self.chunks, self.vectors = chunks, active_vectors
            self.last_stats = dict(chunks=len(chunks), embedded=len(missing),
                                   reused=len(unique) - len(missing))
            return self.last_stats

    def search(self, question, top_k=TOP_K, min_similarity=MIN_SIMILARITY):
        with self.lock:
            self.refresh()
            if not question.strip() or not self.chunks:
                return []
            query = self._embed([question.encode('utf-8')[:6000].decode('utf-8', errors='ignore')])[0]
            matches, seen = [], set()
            for chunk in self.chunks:
                vector = self.vectors[chunk.key]
                if len(query) != len(vector):
                    raise ValueError('Query and index dimensions differ')
                score = sum(a * b for a, b in zip(query, vector))  # normalized cosine
                if score >= min_similarity and chunk.key not in seen:
                    matches.append(Match(chunk, score))
                    seen.add(chunk.key)
            return sorted(matches, key=lambda hit: hit.score, reverse=True)[:top_k]


def format_context(matches, max_chars=MAX_CONTEXT_CHARS):
    """Include only whole selected chunks within the answer context budget."""
    blocks, used = [], 0
    for hit in matches:
        block = f"[Source: {hit.chunk.source}; part {hit.chunk.part}]\n{hit.chunk.text}"
        if used + len(block) + 2 <= max_chars:
            blocks.append(block)
            used += len(block) + 2
    return '\n\n'.join(blocks)
