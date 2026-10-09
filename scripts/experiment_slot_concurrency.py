"""Autonomous Empirical Slot Concurrency Calibration Experiment for QUANTA.

Benchmarks Unsloth Studio (:8888) with unsloth/Qwen3.5-4B-MTP-GGUF:Q5_K_M on NVIDIA GeForce RTX 3070.
Evaluates parallel slot allocations (4, 8, 12, 16 slots with f16 and q8_0 KV cache) under concurrent load.
Measures:
1. Aggregate Generation Throughput (tok/s) across C in {1, 4, 8, 16} concurrent requests.
2. Latency profile (Mean, Min, Max, P95) and queue contention delay.
3. Context capacity per slot (n_ctx / slots) vs QUANTA subgraph context requirements.
4. VRAM allocation and safety headroom on 8GB physical GPU.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "http://127.0.0.1:8888"
API_URL = f"{BASE_URL}/v1"
MODEL_ID = "unsloth/Qwen3.5-4B-MTP-GGUF"
MODEL_VARIANT = "Q5_K_M"

SAMPLE_SHORT_PROMPT = (
    "Provide a concise summary in exactly two bullet points explaining how continuous batching "
    "amortizes memory bandwidth overhead in large language model inference."
)

SAMPLE_MEDIUM_PROMPT_PREAMBLE = (
    "The following is an extracted neuro-symbolic knowledge subgraph from the QUANTA PageTable:\n"
    + "\n".join([
        f"[Node {i:04d}] Entity: Concept_{i} | Valency: (AGENT: Process_{i%5}, PATIENT: Item_{i%7}) | "
        f"Temporal: [t_start={100+i}, t_end={110+i}] | Epistemic: DIRECT_OBSERVATION (P=0.98)"
        for i in range(40)
    ])
    + "\n\nQuestion: Based on the knowledge graph above, describe the temporal relationship between Process_0 and Item_0."
)


def get_gpu_telemetry() -> Dict[str, Any]:
    """Queries nvidia-smi for current VRAM usage."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5.0)
        parts = [p.strip() for p in res.stdout.strip().splitlines()[0].split(",")]
        return {
            "name": parts[0],
            "used_mb": float(parts[1]),
            "total_mb": float(parts[2]),
            "free_mb": float(parts[3]),
            "util_pct": float(parts[4]),
        }
    except Exception as e:
        return {"error": str(e), "used_mb": 0.0, "total_mb": 8192.0}


def load_model_configuration(
    slots: int,
    cache_type_kv: Optional[str] = None,
    timeout: float = 45.0,
) -> Dict[str, Any]:
    """Hot-loads model in Unsloth Studio with specified slots and KV cache type."""
    payload = {
        "model_path": MODEL_ID,
        "gguf_variant": MODEL_VARIANT,
        "gpu_memory_mode": "auto",
        "gpu_layers": -1,
        "n_parallel": slots,
    }
    if cache_type_kv:
        payload["cache_type_kv"] = cache_type_kv

    print(f"\n[*] Requesting load: slots={slots}, kv={cache_type_kv or 'default (f16)'}...")
    t0 = time.time()
    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{BASE_URL}/api/inference/load", json=payload)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to load model: HTTP {r.status_code} - {r.text}")

        # Poll status until loaded
        while time.time() - t0 < timeout:
            time.sleep(1.0)
            st_r = client.get(f"{BASE_URL}/api/inference/status")
            if st_r.status_code == 200:
                st = st_r.json()
                if MODEL_ID in st.get("loaded", []):
                    break

        props = client.get(f"{BASE_URL}/props").json()
        status = client.get(f"{BASE_URL}/api/inference/status").json()

    n_ctx = props.get("default_generation_settings", {}).get("n_ctx", 0)
    actual_slots = props.get("total_slots", 0)
    offloaded = status.get("offloaded_layers", 0)
    total_layers = status.get("offload_total_layers", 0)
    telemetry = get_gpu_telemetry()

    ctx_per_slot = n_ctx // actual_slots if actual_slots > 0 else 0

    print(f"    [+] Loaded: actual_slots={actual_slots}, n_ctx={n_ctx}, ctx/slot={ctx_per_slot}, GPU offload={offloaded}/{total_layers}")
    print(f"    [+] VRAM: {telemetry.get('used_mb'):.0f} / {telemetry.get('total_mb'):.0f} MB (Headroom: {telemetry.get('free_mb'):.0f} MB)")

    return {
        "requested_slots": slots,
        "actual_slots": actual_slots,
        "cache_type_kv": cache_type_kv or "f16",
        "n_ctx": n_ctx,
        "ctx_per_slot": ctx_per_slot,
        "offloaded_layers": offloaded,
        "total_layers": total_layers,
        "all_on_gpu": offloaded == total_layers and total_layers > 0,
        "idle_vram_mb": telemetry.get("used_mb", 0.0),
        "free_vram_mb": telemetry.get("free_mb", 0.0),
    }


async def _dispatch_single_chat(
    client: httpx.AsyncClient,
    prompt: str,
    max_tokens: int = 64,
    request_id: int = 0,
) -> Dict[str, Any]:
    """Sends a single chat completion request and tracks latency and throughput."""
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are a concise expert technical assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(f"{API_URL}/chat/completions", json=payload, timeout=60.0)
        dt = time.perf_counter() - t0
        if resp.status_code == 200:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            comp_tokens = usage.get("completion_tokens", len(content.split()))
            prompt_tokens = usage.get("prompt_tokens", 0)
            return {
                "request_id": request_id,
                "success": True,
                "status_code": 200,
                "latency_s": dt,
                "completion_tokens": comp_tokens,
                "prompt_tokens": prompt_tokens,
                "content": content,
            }
        else:
            return {
                "request_id": request_id,
                "success": False,
                "status_code": resp.status_code,
                "error": resp.text[:200],
                "latency_s": dt,
                "completion_tokens": 0,
            }
    except Exception as exc:
        dt = time.perf_counter() - t0
        return {
            "request_id": request_id,
            "success": False,
            "error": str(exc),
            "latency_s": dt,
            "completion_tokens": 0,
        }


async def run_concurrent_batch(
    concurrency: int,
    prompt: str,
    max_tokens: int = 64,
) -> Dict[str, Any]:
    """Runs a batch of N simultaneous requests concurrently and computes aggregate statistics."""
    limits = httpx.Limits(max_keepalive_connections=concurrency + 4, max_connections=concurrency + 8)
    async with httpx.AsyncClient(limits=limits, timeout=90.0) as client:
        # Pre-record GPU
        gpu_start = get_gpu_telemetry()
        t_batch_start = time.perf_counter()

        tasks = [
            _dispatch_single_chat(client, prompt, max_tokens=max_tokens, request_id=i)
            for i in range(concurrency)
        ]
        results = await asyncio.gather(*tasks)

        t_batch_wall = time.perf_counter() - t_batch_start
        gpu_peak = get_gpu_telemetry()

    successful = [r for r in results if r.get("success")]
    latencies = [r["latency_s"] for r in successful]
    total_tokens = sum(r.get("completion_tokens", 0) for r in successful)
    agg_throughput = total_tokens / t_batch_wall if t_batch_wall > 0 else 0.0

    mean_lat = sum(latencies) / len(latencies) if latencies else 0.0
    min_lat = min(latencies) if latencies else 0.0
    max_lat = max(latencies) if latencies else 0.0
    sorted_lat = sorted(latencies)
    p95_idx = int(0.95 * len(sorted_lat)) if sorted_lat else 0
    p95_lat = sorted_lat[p95_idx] if sorted_lat else 0.0

    return {
        "concurrency": concurrency,
        "total_requests": concurrency,
        "success_count": len(successful),
        "success_rate": len(successful) / concurrency if concurrency > 0 else 0.0,
        "batch_wall_time_s": t_batch_wall,
        "total_completion_tokens": total_tokens,
        "aggregate_throughput_tps": agg_throughput,
        "latency_mean_s": mean_lat,
        "latency_min_s": min_lat,
        "latency_max_s": max_lat,
        "latency_p95_s": p95_lat,
        "peak_vram_mb": max(gpu_start.get("used_mb", 0), gpu_peak.get("used_mb", 0)),
    }


async def test_medium_context_support() -> Dict[str, Any]:
    """Tests if the loaded model can process a realistic ~1,200 token QUANTA subgraph prompt."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        res = await _dispatch_single_chat(client, SAMPLE_MEDIUM_PROMPT_PREAMBLE, max_tokens=50, request_id=999)
        return {
            "success": res.get("success", False),
            "prompt_tokens": res.get("prompt_tokens", 0),
            "latency_s": res.get("latency_s", 0.0),
            "error": res.get("error", None),
        }


async def run_full_experiment() -> Dict[str, Any]:
    """Executes the full slot calibration experiment across configurations."""
    print("=" * 80)
    print("    QUANTA PARALLEL SLOT CONCURRENCY CALIBRATION BENCHMARK")
    print(f"    Hardware: NVIDIA GeForce RTX 3070 (8GB VRAM)")
    print(f"    Model: {MODEL_ID} ({MODEL_VARIANT})")
    print("=" * 80)

    # Configurations to evaluate
    configurations = [
        {"slots": 4, "kv": None, "label": "4 Slots (f16 KV) [Baseline]"},
        {"slots": 8, "kv": None, "label": "8 Slots (f16 KV)"},
        {"slots": 12, "kv": None, "label": "12 Slots (f16 KV)"},
        {"slots": 16, "kv": None, "label": "16 Slots (f16 KV)"},
        {"slots": 8, "kv": "q8_0", "label": "8 Slots (q8_0 KV)"},
        {"slots": 16, "kv": "q8_0", "label": "16 Slots (q8_0 KV)"},
    ]

    concurrency_levels = [1, 4, 8, 16]
    experiment_results: List[Dict[str, Any]] = []

    for cfg in configurations:
        slots = cfg["slots"]
        kv = cfg["kv"]
        label = cfg["label"]

        print(f"\n=======================================================")
        print(f" EVALUATING CONFIGURATION: {label}")
        print(f"=======================================================")

        try:
            load_meta = load_model_configuration(slots=slots, cache_type_kv=kv)
        except Exception as exc:
            print(f"[!] FAILED TO LOAD CONFIGURATION: {exc}")
            continue

        if not load_meta.get("all_on_gpu"):
            print(f"[!] WARNING: Layers spilled to CPU ({load_meta.get('offloaded_layers')}/{load_meta.get('total_layers')}). Skipping per strict GPU policy.")
            continue

        # Test medium context capacity
        print(f"\n  [*] Testing medium context (~1,200 tokens QUANTA subgraph)...")
        med_ctx_result = await test_medium_context_support()
        print(f"      Result: {'PASS' if med_ctx_result['success'] else 'FAIL'} | Prompt tokens: {med_ctx_result.get('prompt_tokens')} | Latency: {med_ctx_result.get('latency_s'):.2f}s")

        # Concurrency sweeps
        bench_sweeps: List[Dict[str, Any]] = []
        for c in concurrency_levels:
            print(f"  [*] Benchmarking Concurrency C={c}...")
            # Warmup short sleep
            await asyncio.sleep(0.5)
            sweep_res = await run_concurrent_batch(concurrency=c, prompt=SAMPLE_SHORT_PROMPT, max_tokens=48)
            bench_sweeps.append(sweep_res)
            print(
                f"      C={c:2d} -> Batch Wall: {sweep_res['batch_wall_time_s']:.2f}s | "
                f"Throughput: {sweep_res['aggregate_throughput_tps']:5.1f} tok/s | "
                f"Mean Lat: {sweep_res['latency_mean_s']:.2f}s | "
                f"P95: {sweep_res['latency_p95_s']:.2f}s | "
                f"VRAM Peak: {sweep_res['peak_vram_mb']:.0f} MB"
            )

        config_report = {
            "label": label,
            "load_metadata": load_meta,
            "medium_context_test": med_ctx_result,
            "concurrency_sweeps": bench_sweeps,
        }
        experiment_results.append(config_report)

    # Save artifact
    output_dir = REPO_ROOT / "output"
    output_dir.mkdir(exist_ok=True, parents=True)
    report_path = output_dir / "slot_concurrency_calibration.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)

    # Print summary comparative table
    print("\n\n" + "=" * 90)
    print("                     EMPIRICAL SLOT CALIBRATION COMPARISON TABLE")
    print("=" * 90)
    print(f"{'Configuration':<26} | {'Ctx/Slot':<8} | {'VRAM (MB)':<10} | {'C=1 (tps)':<9} | {'C=4 (tps)':<9} | {'C=8 (tps)':<9} | {'C=16 (tps)':<10} | {'Med Ctx':<7}")
    print("-" * 90)

    for item in experiment_results:
        lbl = item["label"]
        meta = item["load_metadata"]
        sweeps = {s["concurrency"]: s for s in item["concurrency_sweeps"]}
        c1 = sweeps.get(1, {}).get("aggregate_throughput_tps", 0.0)
        c4 = sweeps.get(4, {}).get("aggregate_throughput_tps", 0.0)
        c8 = sweeps.get(8, {}).get("aggregate_throughput_tps", 0.0)
        c16 = sweeps.get(16, {}).get("aggregate_throughput_tps", 0.0)
        med_ok = "PASS" if item["medium_context_test"].get("success") else "FAIL"

        print(
            f"{lbl:<26} | {meta['ctx_per_slot']:<8} | {meta['idle_vram_mb']:<5.0f}/8192 | "
            f"{c1:7.1f}   | {c4:7.1f}   | {c8:7.1f}   | {c16:8.1f}   | {med_ok:<7}"
        )
    print("=" * 90)

    return {"results": experiment_results, "report_path": str(report_path)}


if __name__ == "__main__":
    asyncio.run(run_full_experiment())
