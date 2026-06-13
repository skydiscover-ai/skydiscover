"""Real-world trace generators.

Downloads oracleGeneral binary traces from libCacheSim's S3, decompresses
with zstd, and converts to our flat uint64_t format.  Key IDs are remapped
to compact 0..N-1 in order of first appearance.

Sources:
  metakv        - Meta Cachelib KV (Berg OSDI'20): 82M unique, 1.64B requests
  twitter       - Twitter Twemcache (Yang OSDI'20): 54 clusters, 10% sampled
  wikimedia     - Wikimedia CDN 2019: ~56M objects, 2.9B requests (var sizes)
  tencent_photo - Tencent QQPhoto CDN ICS'18: ~1B unique objects, heavy tail
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np

from .base import (
    DEFAULT_OUTDIR,
    TraceGenerator,
    TraceMetadata,
    human_count,
)

S3_BASE = "https://s3.amazonaws.com/cache-datasets/cache_dataset_oracleGeneral"

SOURCES = {
    # Meta Cachelib KV (Berg OSDI'20). Default is the 1.68 GB dev sample so
    # `generate.py metakv` stays CI-friendly. For paper-grade LTM runs pick
    # the 202312 aggregate via `--file`:
    #   meta_kvcache_traces_1.oracleGeneral.bin.zst      (1.68 GB, ~17M keys — default)
    #   202206_kv_traces_all.csv.oracleGeneral.zst       (8.03 GB, ~50M keys)
    #   202210_kv_traces_all_sort.csv.oracleGeneral.zst  (6.97 GB, ~50M keys)
    #   202312_kv_traces_all.csv.oracleGeneral.zst       (42.98 GB, ~82M keys — matches Berg OSDI'20)
    #   202401_kv_traces_all_sort.csv.oracleGeneral.zst  (7.33 GB, ~50M keys)
    "metakv": (
        "2022_metaKV/meta_kvcache_traces_1.oracleGeneral.bin.zst",
        "metakv",
        "Meta Cachelib KV (Berg OSDI'20)",
    ),
    "twitter": (
        "2020_twitter/cluster{cluster}.oracleGeneral.sample10.zst",
        "twitter{cluster}",
        "Twitter Twemcache (Yang OSDI'20), cluster {cluster}",
    ),
    # Wikimedia CDN 2019. Variable object sizes — useful for size-aware
    # eviction stories. Alternates (use `--file`):
    #   wiki_2019u.oracleGeneral.zst  (27.33 GB — default, full day)
    #   wiki_2019t.oracleGeneral.zst  ( 2.04 GB — smaller sample)
    #   wiki_2016u.oracleGeneral.zst  (24.28 GB — older snapshot)
    "wikimedia": (
        "2019_wiki/wiki_2019u.oracleGeneral.zst",
        "wikimedia",
        "Wikimedia CDN (2019) -- ~56M objects, 2.9B requests, variable sizes",
    ),
    # Tencent QQPhoto CDN ICS'18. ~1B unique objects — huge. Pair with
    # `--max-load N` to keep it tractable.
    #   tencent_photo1.oracleGeneral.zst  (37.27 GB — default)
    #   tencent_photo2.oracleGeneral.zst  (34.87 GB)
    "tencent_photo": (
        "2018_tencentPhoto/tencent_photo1.oracleGeneral.zst",
        "tencent_photo",
        "Tencent QQPhoto CDN (ICS'18) -- ~1B unique, heavy read-skew",
    ),
}

RECORD_BYTES = 24  # u32 ts | u64 obj_id | u32 size | i64 next_vtime


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _download(url: str, dest: str) -> None:
    """Download with progress.  Idempotent if file already exists."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            remote_size = int(r.headers.get("Content-Length", "0"))
    except urllib.error.URLError as e:
        print(f"ERROR: HEAD {url}: {e}", file=sys.stderr)
        sys.exit(1)

    if os.path.exists(dest) and os.path.getsize(dest) == remote_size:
        print(f"  cached: {dest} ({_human_bytes(remote_size)})")
        return

    print(f"  downloading {url}")
    print(f"    -> {dest} ({_human_bytes(remote_size)})")
    t0 = time.monotonic()
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as out:
        done = 0
        last_print = t0
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            now = time.monotonic()
            if now - last_print > 5.0:
                pct = 100.0 * done / remote_size if remote_size else 0.0
                mbs = done / 1e6 / (now - t0 + 1e-9)
                print(f"    {_human_bytes(done)} / {_human_bytes(remote_size)} "
                      f"({pct:.1f}%, {mbs:.1f} MB/s)")
                last_print = now
    os.rename(tmp, dest)
    dt = time.monotonic() - t0
    print(f"    done in {dt:.1f}s ({remote_size / 1e6 / dt:.1f} MB/s)")


def _convert_oracle_general(zst_path: str, load_path: str, run_path: str,
                            max_run: int | None = None,
                            max_load: int | None = None) -> tuple[int, int]:
    """Stream oracleGeneral trace -> load + run binary files.

    `max_load` caps the number of unique compact IDs minted.  Once the cap
    is reached, further first-sightings of obj_ids are dropped from BOTH
    streams (they have no compact id, so run accesses can't reference
    them).  Makes huge traces (~1B unique keys) tractable.
    """
    try:
        proc = subprocess.Popen(
            ["zstd", "-dcq", zst_path],
            stdout=subprocess.PIPE, bufsize=64 * 1024 * 1024,
        )
    except FileNotFoundError:
        print("ERROR: `zstd` not found. Install: sudo apt install zstd", file=sys.stderr)
        sys.exit(1)

    assert proc.stdout is not None
    id_map: dict[int, int] = {}
    next_id = 0
    n_read = n_load = n_run = 0

    BUF = 256 * 1024
    load_buf = np.empty(BUF, dtype=np.uint64)
    run_buf = np.empty(BUF, dtype=np.uint64)
    load_i = run_i = 0

    t0 = time.monotonic()
    last_print = t0
    chunk_records = 1 << 18
    chunk_bytes = chunk_records * RECORD_BYTES
    unpack = struct.Struct("<Q").unpack_from

    with open(load_path, "wb") as fl, open(run_path, "wb") as fr:
        leftover = b""
        while True:
            data = proc.stdout.read(chunk_bytes)
            if not data and not leftover:
                break
            data = leftover + data
            usable = len(data) - (len(data) % RECORD_BYTES)
            leftover = data[usable:]
            data = data[:usable]

            for off in range(0, usable, RECORD_BYTES):
                obj_id = unpack(data, off + 4)[0]
                n_read += 1

                compact = id_map.get(obj_id)
                if compact is None:
                    # --max-load cap: once the unique-key budget is reached,
                    # new obj_ids get dropped (no compact id, no run entry).
                    if max_load is not None and n_load >= max_load:
                        continue
                    compact = next_id
                    id_map[obj_id] = next_id
                    next_id += 1
                    load_buf[load_i] = compact
                    load_i += 1
                    if load_i == BUF:
                        load_buf.tofile(fl)
                        load_i = 0
                    n_load += 1

                run_buf[run_i] = compact
                run_i += 1
                if run_i == BUF:
                    run_buf.tofile(fr)
                    run_i = 0
                n_run += 1

                if max_run is not None and n_run >= max_run:
                    break

            now = time.monotonic()
            if now - last_print > 10.0:
                rate = n_read / 1e6 / (now - t0 + 1e-9)
                print(f"    {n_read / 1e6:.1f}M records ({n_load / 1e6:.1f}M unique) "
                      f"[{rate:.1f}M rec/s]")
                last_print = now

            if max_run is not None and n_run >= max_run:
                break

        if load_i:
            load_buf[:load_i].tofile(fl)
        if run_i:
            run_buf[:run_i].tofile(fr)

    proc.stdout.close()
    proc.wait()
    return n_load, n_run


def _resolve_key(source: str, file_override: str | None) -> tuple[str, str]:
    """Return (s3_key, desc) honoring an optional --file override."""
    key_tmpl, _name, desc = SOURCES[source]
    if not file_override:
        return key_tmpl, desc
    # Accept either a bare basename (we prepend the source's dir) or a full
    # `2022_metaKV/...` key.  `cache_dataset_...` keys are taken verbatim.
    if file_override.startswith("cache_dataset_") or "/" in file_override:
        key = file_override
    else:
        key = f"{os.path.dirname(key_tmpl)}/{file_override}"
    return key, f"{desc} [{file_override}]"


def _run_real_source(source: str, name: str, outdir: str,
                     max_run: int | None, max_load: int | None,
                     file_override: str | None,
                     generator_classname: str) -> TraceMetadata:
    """Shared driver for every oracleGeneral source (metakv/wikimedia/tencent_photo)."""
    s3_key, desc = _resolve_key(source, file_override)
    url = f"{S3_BASE}/{s3_key}"
    cache_dir = os.path.join(outdir, "_cache")
    zst_local = os.path.join(cache_dir, os.path.basename(s3_key))

    print(f"\n=== {desc} ===")
    os.makedirs(outdir, exist_ok=True)
    _download(url, zst_local)

    tmp_load = os.path.join(outdir, f"load_{name}_raw.dat.tmp")
    tmp_run = os.path.join(outdir, f"run_{name}_raw.dat.tmp")
    print(f"  Converting -> {outdir}/{{load,run}}_{name}_*.dat")
    n_load, n_run = _convert_oracle_general(
        zst_local, tmp_load, tmp_run,
        max_run=max_run, max_load=max_load)

    lcs, rcs = human_count(n_load), human_count(n_run)
    load_path = os.path.join(outdir, f"load_{name}_{lcs}_raw.dat")
    run_path = os.path.join(outdir, f"run_{name}_{lcs}_{rcs}_raw.dat")
    os.replace(tmp_load, load_path)
    os.replace(tmp_run, run_path)

    print(f"  load: {n_load:>12,} unique keys -> {load_path}")
    print(f"  run : {n_run:>12,} accesses    -> {run_path}")

    return TraceMetadata(
        name=f"{name}_{lcs}", category="real",
        generator=generator_classname, description=desc,
        load_file=os.path.abspath(load_path),
        run_file=os.path.abspath(run_path),
        load_count=n_load, run_count=n_run,
        params={"source": source, "max_run": max_run,
                "max_load": max_load, "file": file_override},
    )


class MetaKVGenerator(TraceGenerator):
    name = "metakv"
    category = "real"
    description = "Meta Cachelib KV trace (Berg OSDI'20)"

    def add_args(self, parser) -> None:
        parser.add_argument("--max-run", type=int, default=None,
                            help="Cap run-stream length (for quick dev loops)")
        parser.add_argument("--max-load", type=int, default=None,
                            help="Cap number of unique keys admitted to the load file")
        parser.add_argument("--file", dest="file_override", default=None,
                            help="Override the default S3 basename. Use "
                                 "'202312_kv_traces_all.csv.oracleGeneral.zst' for "
                                 "the 42 GB paper-grade file (matches Berg OSDI'20 "
                                 "82M/1.64B figures). Default is a 1.68 GB dev sample.")
        parser.add_argument("--outdir", default=DEFAULT_OUTDIR)

    def generate(self, outdir: str, args) -> TraceMetadata:
        meta = _run_real_source(
            "metakv", "metakv", outdir,
            max_run=args.max_run, max_load=args.max_load,
            file_override=args.file_override,
            generator_classname="MetaKVGenerator")
        meta.save(outdir)
        return meta


class WikimediaGenerator(TraceGenerator):
    name = "wikimedia"
    category = "real"
    description = "Wikimedia CDN trace (2019 release)"

    def add_args(self, parser) -> None:
        parser.add_argument("--max-run", type=int, default=None)
        parser.add_argument("--max-load", type=int, default=None,
                            help="Cap number of unique keys admitted to the load file")
        parser.add_argument("--file", dest="file_override", default=None,
                            help="Override the default S3 basename (e.g. "
                                 "'wiki_2019t.oracleGeneral.zst' for the 2 GB sample)")
        parser.add_argument("--outdir", default=DEFAULT_OUTDIR)

    def generate(self, outdir: str, args) -> TraceMetadata:
        meta = _run_real_source(
            "wikimedia", "wikimedia", outdir,
            max_run=args.max_run, max_load=args.max_load,
            file_override=args.file_override,
            generator_classname="WikimediaGenerator")
        meta.save(outdir)
        return meta


class TencentPhotoGenerator(TraceGenerator):
    name = "tencent_photo"
    category = "real"
    description = "Tencent QQPhoto CDN trace (ICS'18) -- pair with --max-load"

    def add_args(self, parser) -> None:
        parser.add_argument("--max-run", type=int, default=None)
        parser.add_argument("--max-load", type=int, default=None,
                            help="Cap number of unique keys admitted to the load "
                                 "file. ~1B unique keys native -- strongly "
                                 "recommended to set this.")
        parser.add_argument("--file", dest="file_override", default=None,
                            help="Override the default S3 basename (e.g. "
                                 "'tencent_photo2.oracleGeneral.zst')")
        parser.add_argument("--outdir", default=DEFAULT_OUTDIR)

    def generate(self, outdir: str, args) -> TraceMetadata:
        meta = _run_real_source(
            "tencent_photo", "tencent_photo", outdir,
            max_run=args.max_run, max_load=args.max_load,
            file_override=args.file_override,
            generator_classname="TencentPhotoGenerator")
        meta.save(outdir)
        return meta


class TwitterGenerator(TraceGenerator):
    name = "twitter"
    category = "real"
    description = "Twitter Twemcache trace (Yang OSDI'20)"

    def add_args(self, parser) -> None:
        parser.add_argument("--cluster", type=int, default=18,
                            help="Twitter cluster number (default: 18)")
        parser.add_argument("--max-run", type=int, default=None)
        parser.add_argument("--outdir", default=DEFAULT_OUTDIR)

    def generate(self, outdir: str, args) -> TraceMetadata:
        cluster = args.cluster
        key_tmpl, name_tmpl, desc_tmpl = SOURCES["twitter"]
        s3_key = key_tmpl.format(cluster=cluster)
        name = name_tmpl.format(cluster=cluster)
        desc = desc_tmpl.format(cluster=cluster)

        url = f"{S3_BASE}/{s3_key}"
        cache_dir = os.path.join(outdir, "_cache")
        zst_local = os.path.join(cache_dir, os.path.basename(s3_key))

        print(f"\n=== {desc} ===")
        os.makedirs(outdir, exist_ok=True)
        _download(url, zst_local)

        tmp_load = os.path.join(outdir, f"load_{name}_raw.dat.tmp")
        tmp_run = os.path.join(outdir, f"run_{name}_raw.dat.tmp")
        print(f"  Converting -> {outdir}/{{load,run}}_{name}_*.dat")
        n_load, n_run = _convert_oracle_general(zst_local, tmp_load, tmp_run, args.max_run)

        lcs, rcs = human_count(n_load), human_count(n_run)
        load_path = os.path.join(outdir, f"load_{name}_{lcs}_raw.dat")
        run_path = os.path.join(outdir, f"run_{name}_{lcs}_{rcs}_raw.dat")
        os.replace(tmp_load, load_path)
        os.replace(tmp_run, run_path)

        print(f"  load: {n_load:>12,} unique keys -> {load_path}")
        print(f"  run : {n_run:>12,} accesses    -> {run_path}")

        meta = TraceMetadata(
            name=f"{name}_{lcs}", category="real",
            generator="TwitterGenerator", description=desc,
            load_file=os.path.abspath(load_path),
            run_file=os.path.abspath(run_path),
            load_count=n_load, run_count=n_run,
            params={"source": "twitter", "cluster": cluster, "max_run": args.max_run},
        )
        meta.save(outdir)
        return meta
