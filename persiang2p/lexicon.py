"""
persiang2p.lexicon
Fast O(1) pronunciation dictionary for Persian G2P.
Loads KaamelDict (116K+ words) + modern AI/computing loanwords.
Provides instantaneous pronunciation lookup for standard vocabulary.
"""

import os
import pickle

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

class Lexicon:
    def __init__(self, data_path: str = None, cache: bool = True):
        self.entries = {}
        if data_path is None:
            train_tsv = os.path.join(DEFAULT_DATA_DIR, "persian_g2p_train_kaamel.tsv")
            dev_tsv = os.path.join(DEFAULT_DATA_DIR, "persian_g2p_dev_kaamel.tsv")
            sources = [train_tsv, dev_tsv]
        else:
            sources = [data_path]

        self._load_sources(sources, cache)

    def _load_sources(self, file_paths, cache: bool):
        cache_path = os.path.join(DEFAULT_DATA_DIR, "lexicon_cache.pkl")
        
        # Try loading from fast binary cache
        if cache and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    self.entries = pickle.load(f)
                    return
            except Exception:
                pass

        # Load from TSV files
        for p in file_paths:
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        word, ipa = parts[0].strip(), parts[1].strip()
                        if word and ipa and word not in self.entries:
                            self.entries[word] = ipa

        # Save binary cache for instant cold starts in subsequent calls
        if cache and self.entries:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(self.entries, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception:
                pass

    def get(self, word: str) -> str:
        """Looks up word in lexicon. Returns IPA string or None."""
        return self.entries.get(word, None)

    def __contains__(self, word: str) -> bool:
        return word in self.entries

    def __len__(self) -> int:
        return len(self.entries)
