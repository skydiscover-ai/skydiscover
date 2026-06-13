"""
bench/worker_pool.py — GCP VM worker pool for parallel benchmark evaluation.

Creates and manages a pool of identical VMs that run benchmarks and
correctness tests in parallel. Workers auto-delete after idle timeout.

Used by orchestrator.py when --parallel-eval is active.
"""

import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


def _log(msg: str) -> None:
    print(f"[pool] {msg}", file=sys.stderr, flush=True)


def _gcp_metadata(path: str) -> str:
    """Fetch GCP instance metadata. Returns '' if unavailable."""
    try:
        r = subprocess.run(
            ["curl", "-sf", "-H", "Metadata-Flavor: Google",
             f"http://metadata.google.internal/computeMetadata/v1/{path}"],
            capture_output=True, text=True, timeout=3,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _gcloud(*args, project: str = "", zone: str = "",
            capture: bool = True, timeout: int = 120) -> subprocess.CompletedProcess:
    cmd = ["gcloud"] + list(args)
    if project:
        cmd += [f"--project={project}"]
    if zone:
        cmd += [f"--zone={zone}"]
    return subprocess.run(
        cmd, capture_output=capture, text=True, timeout=timeout,
    )


@dataclass
class PoolConfig:
    """Worker pool configuration. Auto-detects from environment."""
    project: str = ""
    zone: str = ""
    machine_type: str = ""
    idle_timeout_min: int = 30
    local_ssd_count: int = 8
    boot_disk_size: int = 50
    prefix: str = "skykv-eval"
    network: str = ""
    subnet: str = ""

    @classmethod
    def auto(cls, machine_type: str = "", num_workers: int = 0) -> "PoolConfig":
        cfg = cls()

        # Project
        cfg.project = (
            os.environ.get("POOL_PROJECT")
            or _gcp_metadata("project/project-id")
            or subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        )

        # Zone
        raw_zone = _gcp_metadata("instance/zone")
        cfg.zone = (
            os.environ.get("POOL_ZONE")
            or (raw_zone.split("/")[-1] if raw_zone else "")
            or "us-central1-a"
        )

        # Machine type
        cfg.machine_type = machine_type  # set by caller based on memory needs

        # Idle timeout
        cfg.idle_timeout_min = int(os.environ.get("POOL_IDLE_TIMEOUT_MIN", "30"))

        # Network/subnet — detect from existing instance in project
        cfg.network = os.environ.get("POOL_NETWORK", "")
        cfg.subnet = os.environ.get("POOL_SUBNET", "")
        if not cfg.network:
            ref = _gcloud(
                "compute", "instances", "list",
                f"--zones={cfg.zone}", "--limit=1", "--format=value(name)",
                project=cfg.project,
            )
            ref_name = ref.stdout.strip()
            if ref_name:
                desc = _gcloud(
                    "compute", "instances", "describe", ref_name,
                    "--format=value(networkInterfaces[0].network,"
                    "networkInterfaces[0].subnetwork)",
                    project=cfg.project, zone=cfg.zone,
                )
                parts = desc.stdout.strip().split("\n")
                if len(parts) >= 1 and parts[0]:
                    cfg.network = parts[0].split("/")[-1]
                if len(parts) >= 2 and parts[1]:
                    cfg.subnet = parts[1].split("/")[-1]

        return cfg


class WorkerPool:
    """Manages a pool of GCP VMs for parallel eval."""

    def __init__(self, cfg: PoolConfig):
        self.cfg = cfg
        self.workers: list[str] = []  # VM names
        self.ready = False

    def init(self, count: int) -> bool:
        """Ensure at least `count` workers exist and are ready."""
        cfg = self.cfg
        if not shutil.which("gcloud"):
            _log("ERROR: gcloud CLI not found")
            return False
        if not cfg.project:
            _log("ERROR: could not determine GCP project")
            return False

        # Discover existing workers
        _log(f"Checking for existing workers (prefix={cfg.prefix})...")
        r = _gcloud(
            "compute", "instances", "list",
            f"--filter=name~'^{cfg.prefix}-[0-9]+$' AND status=RUNNING",
            "--format=value(name)",
            project=cfg.project,
        )
        existing = [n for n in r.stdout.strip().splitlines() if n]
        if existing:
            self.workers = existing
            _log(f"Found {len(existing)} existing worker(s): {' '.join(existing)}")

        # Create more if needed
        need = count - len(self.workers)
        if need > 0:
            _log(f"Creating {need} new worker(s)...")
            setup_script = str(
                Path(__file__).resolve().parent.parent / "scripts" / "worker_setup.sh"
            )
            if not os.path.isfile(setup_script):
                _log(f"ERROR: {setup_script} not found")
                return False

            ssd_flags = " ".join(
                ["--local-ssd=interface=NVME"] * cfg.local_ssd_count
            )
            net_flags = ""
            if cfg.network:
                net_flags += f" --network={cfg.network}"
            if cfg.subnet:
                net_flags += f" --subnet={cfg.subnet}"

            for i in range(need):
                idx = len(self.workers) + 1
                name = f"{cfg.prefix}-{idx:02d}"
                _log(f"Creating {name} ({cfg.machine_type}, "
                     f"{cfg.local_ssd_count} local SSDs, "
                     f"network={cfg.network or 'default'})...")
                try:
                    r = _gcloud(
                        "compute", "instances", "create", name,
                        f"--machine-type={cfg.machine_type}",
                        *ssd_flags.split(),
                        *net_flags.split(),
                        "--image-family=ubuntu-2404-lts-amd64",
                        "--image-project=ubuntu-os-cloud",
                        f"--boot-disk-size={cfg.boot_disk_size}GB",
                        "--boot-disk-type=pd-ssd",
                        f"--metadata=idle-timeout-min={cfg.idle_timeout_min}",
                        f"--metadata-from-file=startup-script={setup_script}",
                        "--scopes=compute-rw,storage-ro",
                        "--quiet",
                        project=cfg.project, zone=cfg.zone,
                        timeout=300,  # VM creation with 8 SSDs can take >2 min
                    )
                    if r.returncode == 0:
                        self.workers.append(name)
                    else:
                        _log(f"WARNING: Failed to create {name}: "
                             f"{r.stderr.strip().splitlines()[-1] if r.stderr else 'unknown'}")
                except subprocess.TimeoutExpired:
                    _log(f"WARNING: Timed out creating {name} (300s)")

        if not self.workers:
            _log("ERROR: No workers available")
            return False

        # Wait for ready
        _log("Waiting for workers to be ready (SSH + trace data)...")
        ready_count = 0
        for name in self.workers:
            for attempt in range(60):  # 5 min
                if self._ssh(name, "test -f /tmp/skykv_worker_ready && "
                             "test -f /mnt/ssd/ycsb_data/load_zipf_250M_raw.dat"):
                    _log(f"  {name}: ready")
                    ready_count += 1
                    break
                time.sleep(5)
            else:
                _log(f"  WARNING: {name} not ready after 5 minutes")

        if ready_count == 0:
            _log("ERROR: No workers became ready")
            return False

        self.ready = True
        _log(f"{ready_count} worker(s) ready")
        return True

    def ship(self, build_dir: str) -> None:
        """Copy binaries + worker_bench.sh to all workers."""
        bench_script = str(
            Path(__file__).resolve().parent.parent / "scripts" / "worker_bench.sh"
        )
        files = [
            (os.path.join(build_dir, "kvstore_bench"), "/tmp/skykv_eval/kvstore_bench"),
            (os.path.join(build_dir, "consistency_test"), "/tmp/skykv_eval/consistency_test"),
            (bench_script, "/tmp/skykv_eval/worker_bench.sh"),
        ]
        _log(f"Shipping binaries to {len(self.workers)} worker(s)...")

        def ship_one(name: str) -> bool:
            self._ssh(name, "mkdir -p /tmp/skykv_eval")
            for src, dst in files:
                if os.path.isfile(src):
                    self._scp(name, src, dst)
            self._ssh(name, "chmod +x /tmp/skykv_eval/*")
            return True

        with ThreadPoolExecutor(max_workers=len(self.workers)) as pool:
            futures = {pool.submit(ship_one, w): w for w in self.workers}
            failed = 0
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception:
                    failed += 1
            if failed:
                _log(f"WARNING: Ship failed for {failed} worker(s)")
        _log("Ship complete")

    def fan_out(self, jobs: list[tuple[str, str]]) -> list[int]:
        """Run commands in parallel on workers, collecting stdout to local files.

        Args:
            jobs: list of (remote_command, local_output_file) tuples.

        Returns:
            list of exit codes, one per job.
        """
        n_jobs = len(jobs)
        n_workers = len(self.workers)
        _log(f"Fan out: {n_jobs} job(s) across {n_workers} worker(s)")

        exit_codes = [1] * n_jobs

        # Process in batches of n_workers
        for batch_start in range(0, n_jobs, n_workers):
            batch = jobs[batch_start:batch_start + n_workers]

            def run_one(idx_cmd_out: tuple[int, str, str, str]) -> tuple[int, int]:
                idx, name, cmd, out_file = idx_cmd_out
                self._ssh(name, "touch /tmp/skykv_heartbeat")
                rc = self._ssh_capture(name, cmd, out_file)
                return idx, rc

            work = []
            for i, (cmd, out_file) in enumerate(batch):
                worker = self.workers[i % n_workers]
                work.append((batch_start + i, worker, cmd, out_file))

            with ThreadPoolExecutor(max_workers=len(batch)) as pool:
                for idx, rc in pool.map(run_one, work):
                    exit_codes[idx] = rc

        _log("Fan out complete")
        return exit_codes

    def heartbeat(self) -> None:
        """Touch heartbeat on all workers to prevent idle deletion."""
        for name in self.workers:
            self._ssh(name, "touch /tmp/skykv_heartbeat")

    def teardown(self) -> None:
        """Delete all worker VMs."""
        if not self.workers:
            return
        _log(f"Deleting {len(self.workers)} worker(s)...")
        for name in self.workers:
            _gcloud("compute", "instances", "delete", name, "--quiet",
                    project=self.cfg.project, zone=self.cfg.zone)
        self.workers.clear()
        self.ready = False

    def has_traces(self, load_file: str, run_file: str) -> bool:
        """Check if workers have the needed trace files."""
        if not self.workers:
            return False
        return self._ssh(self.workers[0],
                         f"test -f '{load_file}' && test -f '{run_file}'")

    # ── SSH/SCP helpers ──────────────────────────────────────────────────

    def _ssh(self, name: str, cmd: str) -> bool:
        """Run command on worker, return True if exit 0."""
        r = _gcloud(
            "compute", "ssh", name,
            "--ssh-flag=-o StrictHostKeyChecking=no",
            "--ssh-flag=-o UserKnownHostsFile=/dev/null",
            "--ssh-flag=-o ConnectTimeout=10",
            "--ssh-flag=-o LogLevel=ERROR",
            "--quiet",
            f"--command={cmd}",
            project=self.cfg.project, zone=self.cfg.zone,
            timeout=30,
        )
        return r.returncode == 0

    def _ssh_capture(self, name: str, cmd: str, output_file: str) -> int:
        """Run command on worker, capture stdout+stderr to local file."""
        with open(output_file, "w") as f:
            try:
                r = subprocess.run(
                    ["gcloud", "compute", "ssh", name,
                     "--ssh-flag=-o StrictHostKeyChecking=no",
                     "--ssh-flag=-o UserKnownHostsFile=/dev/null",
                     "--ssh-flag=-o ConnectTimeout=30",
                     "--ssh-flag=-o ServerAliveInterval=30",
                     "--ssh-flag=-o ServerAliveCountMax=35",
                     "--ssh-flag=-o LogLevel=ERROR",
                     "--quiet",
                     f"--project={self.cfg.project}",
                     f"--zone={self.cfg.zone}",
                     f"--command={cmd}"],
                    stdout=f, stderr=subprocess.STDOUT,
                    timeout=960,
                )
                return r.returncode
            except subprocess.TimeoutExpired:
                return 124

    def _scp(self, name: str, src: str, dst: str) -> bool:
        r = _gcloud(
            "compute", "scp", src, f"{name}:{dst}",
            "--scp-flag=-o StrictHostKeyChecking=no",
            "--scp-flag=-o UserKnownHostsFile=/dev/null",
            "--quiet",
            project=self.cfg.project, zone=self.cfg.zone,
            timeout=60,
        )
        return r.returncode == 0
