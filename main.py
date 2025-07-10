from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from flashtext import KeywordProcessor
from symspellpy import SymSpell, Verbosity
import string
from fastapi.middleware.cors import CORSMiddleware

from typo_list import CBC_TYPO_LIST
from index_terms import INDEX_TERMS
from normalization_dict import CBC_NORMALIZATION_DICT
from protected_words import PROTECTED_WORDS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allows specific origins to make requests
    allow_credentials=True,      # Supports cookies and authorization headers
    allow_methods=["*"],         # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],         # Allows all headers
)

# ————— SymSpell setup (constant time corrections) —
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)
sym_spell.load_bigram_dictionary("frequency_bigramdictionary_en_243_342.txt", term_index=0, count_index=2)

# ————— FlashText for phrase-level mappings —————
phrase_map = {
    **{k.lower(): v for k, v in CBC_NORMALIZATION_DICT.items()},
    **{t["name"].lower(): t["name"] for t in INDEX_TERMS},
}
keyword_processor = KeywordProcessor(case_sensitive=False)
for phrase, rep in phrase_map.items():
    keyword_processor.add_keyword(phrase, rep)

def preprocess_phrases(text: str) -> str:
    return keyword_processor.replace_keywords(text)

# ————— Precompute sets for O(1) checks —————
PROTECTED_SET = set(w.lower() for w in PROTECTED_WORDS)
DOMAIN_SET    = set(w.lower() for w in CBC_TYPO_LIST if len(w) > 2)

# ————— Request / Response models —————
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    normalizedQuery: str
    suggestions: List[str]
    phrasePassMatches: List[str]
    firstPassMatches: List[str]
    fallbackMatches: List[str]

# ————— Static suggestions —————
STATIC_SUGGESTIONS = [f"Suggestion {i}" for i in range(1, 11)]

# ————— Phase 1: exact matching —————
def first_pass(tokens: List[str]) -> List[Optional[str]]:
    out: List[Optional[str]] = []
    for tok in tokens:
        core = tok.strip(string.punctuation)
        low  = core.lower()
        if low in PROTECTED_SET or low in DOMAIN_SET:
            out.append(core)
        else:
            out.append(None)
    return out

# ————— Phase 2: fallback correction —————
def correct_token(token: str) -> str:
    suggestions = sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=2)
    return suggestions[0].term if suggestions else token

@app.post("/query", response_model=QueryResponse)
async def correct_and_suggest(body: QueryRequest):
    raw = body.query.strip()
    if not raw:
        raise HTTPException(400, "`query` must be non-empty")

    # Phase A: phrase-level normalization & extraction
    raw_tokens = raw.split()
    pre = preprocess_phrases(raw)
    pre_tokens = pre.split()
    # extract original matched keywords
    phrase_pass_matches = keyword_processor.extract_keywords(raw)

    # Phase B: first-pass token filtering
    first = first_pass(pre_tokens)
    first_pass_matches = [fp for fp in first if fp is not None]

    # Phase C: fallback via SymSpell
    normalized_tokens: List[str] = []
    fallback_matches: List[str] = []
    for orig, fp in zip(pre_tokens, first):
        if fp is not None:
            normalized_tokens.append(fp)
        else:
            core = orig.strip(string.punctuation)
            corr = correct_token(core)
            if corr.lower() != core.lower():
                fallback_matches.append(corr)
            punct = orig[len(core):]
            normalized_tokens.append(corr + punct)

    normalized = " ".join(normalized_tokens)

    return QueryResponse(
        normalizedQuery=normalized,
        suggestions=STATIC_SUGGESTIONS,
        phrasePassMatches=phrase_pass_matches,
        firstPassMatches=first_pass_matches,
        fallbackMatches=fallback_matches
    )
