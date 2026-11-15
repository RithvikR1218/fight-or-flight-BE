"""Search eval report: pass rate, failures, top results and a latency table.

    cd backend && python -m tests.search_eval [--semantic PATH.npz] [--fusion or|rrf] [-v] [-q QUERY ...]
"""
import argparse
import os
import time
import tracemalloc

from .cases import CASES
from .harness import Runner, build_index, latency, load_rows, percentile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semantic", metavar="NPZ", default=os.getenv("SEARCH_EVAL_VECTORS"))
    ap.add_argument("--lexical", action="store_true", help="ignore --semantic / SEARCH_EVAL_VECTORS")
    ap.add_argument("--fusion", default="or", choices=("or", "rrf"))
    ap.add_argument("-v", "--verbose", action="store_true", help="print the top 3 of every query")
    ap.add_argument("-q", "--query", nargs="*", help="only these queries")
    ap.add_argument("--latency", action="store_true")
    ap.add_argument("--mem", action="store_true", help="measure index memory with tracemalloc (slow build)")
    args = ap.parse_args()

    t = time.perf_counter()
    load_rows()
    print(f"rows loaded from data files in {time.perf_counter() - t:.1f}s")
    path = None if args.lexical else args.semantic
    if args.mem:      # tracemalloc slows the build ~8x; only for the memory number
        tracemalloc.start()
    idx = build_index(path)
    mem_idx = tracemalloc.get_traced_memory()[0] if args.mem else 0
    tracemalloc.stop()
    encoder = None
    if path:
        from app.services.search_semantic import QueryEncoder

        encoder = QueryEncoder(mode="on")
    runner = Runner(idx, encoder=encoder, fusion=args.fusion)
    print(f"index: {idx.N} places, built in {idx.build_seconds:.1f}s, ~{mem_idx / 1e6:.0f} MB "
          f"(numpy arrays {idx.nbytes() / 1e6:.0f} MB), semantic={'on (' + str(idx.semantic_model) + ')' if path else 'off'}"
          f", fusion={args.fusion}")

    cases = [c for c in CASES if not args.query or c.q in args.query]
    passed, failed = 0, []
    for c in cases:
        res, fails = runner.check(c)
        if fails:
            failed.append((c, fails, res))
        else:
            passed += 1
        if args.verbose or args.query:
            mark = "ok  " if not fails else "FAIL"
            top = "; ".join(f"{r['name'][:38]} ({r.get('distance_m', 0) / 1000:.2f} km, {r['match']})" for r in res[:3])
            print(f"{mark} {c.q!r:24} {top}")
    tuned = [c for c in cases if not c.held_out]
    held = [c for c in cases if c.held_out]
    bad = {id(c) for c, _, _ in failed}
    pt = sum(id(c) not in bad for c in tuned)
    ph = sum(id(c) not in bad for c in held)
    print(f"\npassed {passed}/{len(cases)} ({100 * passed / max(len(cases), 1):.0f}%): "
          f"judged {pt}/{len(tuned)}, held-out paraphrases {ph}/{len(held)}")
    for c, fails, res in failed:
        print(f"  FAIL {c.q!r}: " + "; ".join(fails))

    if args.latency:
        s_ms, _ = latency(idx, cases, None)
        print(f"\nlatency (index search incl. cold query parsing, {len(s_ms)} runs):"
              f" p50 {percentile(s_ms, 50):.1f} ms, p95 {percentile(s_ms, 95):.1f} ms, max {max(s_ms):.1f} ms")
        if encoder is not None:
            s2, e_ms = latency(idx, cases, encoder)
            print(f"with semantic: search p50 {percentile(s2, 50):.1f} ms, p95 {percentile(s2, 95):.1f} ms; "
                  f"query embed (uncached) p50 {percentile(e_ms, 50):.1f} ms, p95 {percentile(e_ms, 95):.1f} ms "
                  f"({len(e_ms)} of {len(cases)} queries embedded)")


if __name__ == "__main__":
    main()
