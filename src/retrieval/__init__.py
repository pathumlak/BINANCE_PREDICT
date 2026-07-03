"""Phase 6 — Pattern-matching engine.

Submodules:
  * hmm_regimes  – Gaussian HMM market-regime classifier (3 states).
  * faiss_index  – FAISS cosine k-NN over Phase 3 CNN embeddings.
  * pattern_engine – Facade combining FAISS + regime filter.
  * ewc          – Elastic Weight Consolidation utilities for the CNN.
"""
