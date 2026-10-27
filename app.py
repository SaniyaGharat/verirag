"""
VeriRAG Web Application & API Server
====================================
Serves the Grade One Verification Studio UI and REST API.
Endpoints:
- POST /api/verify          : End-to-end verification of legal text
- POST /api/search          : Search across ILDC FAISS & BM25 retrieval index
- GET  /api/benchmark       : Benchmark dataset metadata & sample triples
- GET  /api/evaluation      : Consolidated experimental evaluation results (Exp A-D)
- GET  /api/health          : System health & index status
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, Any
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pipeline import VeriRAG
from retriever.search import Retriever, reciprocal_rank_fusion

app = Flask(__name__, static_folder="ui", static_url_path="")
CORS(app)

# Global singleton pipeline instance
rag_engine: VeriRAG = None


def get_engine() -> VeriRAG:
    global rag_engine
    if rag_engine is None:
        print("[App] Initializing VeriRAG engine...")
        rag_engine = VeriRAG()
        print("[App] Engine ready.")
    return rag_engine


@app.route("/")
def index():
    return send_from_directory("ui", "index.html")


@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory("ui", path)


@app.route("/api/health", methods=["GET"])
def health():
    engine = get_engine()
    return jsonify({
        "status": "healthy",
        "faiss_entries": engine.retriever.index.ntotal if hasattr(engine.retriever, "index") else 0,
        "bm25_active": engine.retriever.bm25 is not None,
        "embedding_model": engine.retriever.model_name,
        "verifier_model": engine.verifier.verifier.model_name,
        "legal_checks_enabled": engine.verifier.enable_legal_checks
    })


@app.route("/api/verify", methods=["POST"])
def verify_text():
    data = request.get_json(force=True) or {}
    text = data.get("text", "").strip()
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 5))
    decompose_method = data.get("decompose_method", "rule_based")
    source_doc_id = data.get("source_doc_id")

    if not text:
        return jsonify({"error": "Field 'text' is required"}), 400

    engine = get_engine()
    t0 = time.time()
    try:
        result = engine.verify(
            text=text,
            query=query,
            top_k=top_k,
            decompose_method=decompose_method,
            source_doc_id=source_doc_id
        )
        result["execution_time_sec"] = round(time.time() - t0, 3)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search", methods=["POST"])
def search_corpus():
    data = request.get_json(force=True) or {}
    query = data.get("query", "").strip()
    top_k = int(data.get("top_k", 5))
    mode = data.get("mode", "dense")  # dense, sparse, hybrid

    if not query:
        return jsonify({"error": "Field 'query' is required"}), 400

    engine = get_engine()
    t0 = time.time()
    try:
        if mode == "dense":
            results = engine.retriever.search(query=query, top_k=top_k)
        elif mode == "sparse":
            results = engine.retriever.bm25_search(query=query, top_k=top_k)
        elif mode == "hybrid":
            dense_results = engine.retriever.search(query=query, top_k=top_k)
            sparse_results = engine.retriever.bm25_search(query=query, top_k=top_k)
            results = reciprocal_rank_fusion(dense_results, sparse_results, top_k)
        else:
            return jsonify({"error": "Field 'mode' must be dense, sparse, or hybrid"}), 400

        return jsonify({
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "total_matches": len(results),
            "latency_ms": round((time.time() - t0) * 1000, 2),
            "passages": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/benchmark", methods=["GET"])
def get_benchmark_data():
    bench_dir = PROJECT_ROOT / "data" / "benchmark"
    summary_file = bench_dir / "summary.json"
    test_file = bench_dir / "test.jsonl"

    summary = {}
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            summary = json.load(f)

    samples = []
    if test_file.exists():
        with open(test_file, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= 30:  # send first 30 test triples
                    break
                if line.strip():
                    samples.append(json.loads(line))

    return jsonify({
        "summary": summary,
        "sample_triples": samples
    })


@app.route("/api/evaluation", methods=["GET"])
def get_evaluation_data():
    results_dir = PROJECT_ROOT / "eval" / "results"
    
    def read_json_if_exists(fname):
        p = results_dir / fname
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    report_md = ""
    rep_path = results_dir / "EVALUATION_REPORT.md"
    if rep_path.exists():
        with open(rep_path, "r", encoding="utf-8") as f:
            report_md = f.read()

    return jsonify({
        "experiment_a": read_json_if_exists("experiment_a_baselines.json"),
        "experiment_b": read_json_if_exists("experiment_b_legal_checks.json"),
        "experiment_c": read_json_if_exists("experiment_c_ablations.json"),
        "experiment_d": read_json_if_exists("experiment_d_robustness.json"),
        "report_markdown": report_md
    })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VeriRAG Web Application Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    args = parser.parse_args()

    # Pre-warm engine
    get_engine()

    print(f"\n[VeriRAG] Server running at: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
