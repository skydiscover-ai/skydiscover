import json
import logging
import os
import signal
import sys
import time
import uuid
from typing import Optional

from skydiscover.config import Config, build_output_dir, load_config
from skydiscover.search.base_database import Program
from skydiscover.search.default_discovery_controller import (
    DiscoveryController,
    DiscoveryControllerInput,
)
from skydiscover.search.registry import create_database, get_program
from skydiscover.search.route import get_discovery_controller
from skydiscover.search.utils.logging_utils import setup_search_logging
from skydiscover.utils.code_utils import extract_solution_language
from skydiscover.utils.metrics import format_metrics, get_score

logger = logging.getLogger(__name__)


class Runner:
    """Top-level entry point for a discovery run.

    Loads config, creates the database and discovery controller, runs the
    search loop, and saves checkpoints + best program.

    Args:
        initial_program_path: path to the starting solution file.
        evaluation_file: path to the user's evaluator script (must define evaluate()).
        config_path: optional YAML config file (ignored if config is provided).
        config: optional pre-built Config object (takes priority over config_path).
        output_dir: where to write logs, checkpoints, and best program.
            Auto-generated from search type + problem name if omitted.
    """

    def __init__(
        self,
        evaluation_file: str,
        initial_program_path: Optional[str] = None,
        config_path: Optional[str] = None,
        config: Optional[Config] = None,
        output_dir: Optional[str] = None,
        evaluator_env_vars: Optional[dict[str, str]] = None,
    ):
        self.config = config if config is not None else load_config(config_path)
        self.name = self.config.search.type
        self.output_dir = output_dir or build_output_dir(
            self.name, initial_program_path or "scratch"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_logging()

        # Load the initial program (can be optional)
        self.initial_program_path = initial_program_path
        self.initial_program_solution = (
            self._load_initial_program() if initial_program_path else None
        )
        if self.initial_program_solution and not self.config.language:
            self.config.language = extract_solution_language(self.initial_program_solution)
        if not self.config.language:
            self.config.language = "python"

        # Set the file extension
        ext = os.path.splitext(initial_program_path)[1] if initial_program_path else ".py"
        ext = ext or ".py"
        self.file_extension = ext if ext.startswith(".") else f".{ext}"
        if self.config.file_suffix == ".py":
            self.config.file_suffix = self.file_extension

        # Create the database
        self.database = create_database(self.config.search.type, self.config.search.database)
        self.database.language = self.config.language or "python"
        self.evaluation_file = evaluation_file
        self.evaluator_env_vars = dict(evaluator_env_vars or {})

        # Initialize the discovery controller
        self.discovery_controller: Optional[DiscoveryController] = None

        logger.info(f"Runner ready: search={self.name}, program={self.initial_program_path}")

    @property
    def initial_score(self) -> Optional[float]:
        """Score of the seed program, or None if unavailable."""
        if not self.database or not self.database.programs or not self.initial_program_solution:
            return None

        seed_solution = self.initial_program_solution
        seed_prog = None
        for prog in self.database.programs.values():
            if prog.solution == seed_solution:
                seed_prog = prog
                break
        if seed_prog is None:
            for prog in self.database.programs.values():
                if prog.iteration_found == 0:
                    seed_prog = prog
                    break

        if seed_prog and seed_prog.metrics:
            return get_score(seed_prog.metrics)
        return None

    async def run(
        self,
        iterations: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Optional[Program]:
        """Entrypoint for the discovery process.

        Args:
            iterations: max iterations (uses config.max_iterations if None).
            checkpoint_path: resume from this checkpoint directory if provided.

        Returns:
            Best Program found, or None if no valid programs were produced.
        """
        max_iterations = iterations if iterations is not None else self.config.max_iterations

        start_iteration = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            start_iteration = self.database.last_iteration + 1
            logger.info(f"Resuming from iteration {start_iteration}")
        else:
            start_iteration = self.database.last_iteration

        # Create the discovery controller input
        controller_input = DiscoveryControllerInput(
            config=self.config,
            evaluation_file=self.evaluation_file,
            database=self.database,
            file_suffix=self.config.file_suffix,
            output_dir=self.output_dir,
            evaluator_env_vars=self.evaluator_env_vars,
        )

        # Get the discovery controller
        self.discovery_controller = get_discovery_controller(controller_input)

        # Add initial program to database if not resuming
        should_add_initial = (
            start_iteration == 0
            and len(self.database.programs) == 0
            and self.initial_program_solution is not None
        )

        if should_add_initial:
            await self._add_initial_program(start_iteration)
        else:
            logger.info(
                f"Resuming from iteration {start_iteration} with {len(self.database.programs)} programs"
            )

        # Start the monitor
        monitor_server = None
        try:
            monitor_server = self._start_monitor(max_iterations)
            self._setup_human_feedback(monitor_server)
            self._setup_monitor_summary(monitor_server)
            self._push_existing_to_monitor()
            self._install_signal_handlers()

            discovery_start = start_iteration + 1 if should_add_initial else start_iteration
            self.database.log_status()

            def checkpoint_cb(iteration: int) -> None:
                self._sync_database()
                self._save_checkpoint(iteration)

            # MAIN LOOP: Run the discovery
            await self.discovery_controller.run_discovery(
                discovery_start,
                max_iterations,
                checkpoint_callback=checkpoint_cb,
            )

            self._sync_database()
            final_iteration = discovery_start + max_iterations - 1
            if final_iteration > 0:
                self._save_checkpoint(final_iteration)

            # Re-evaluate best program in test mode (authoritative score).
            best = self._get_best_program()
            if best:
                try:
                    test_result = await self.discovery_controller.evaluator.evaluate_program(
                        best.solution, best.id, mode="test"
                    )
                    for k, v in test_result.metrics.items():
                        best.metrics[f"test_{k}"] = v
                    logger.info(
                        f"Test evaluation for best program: {format_metrics(test_result.metrics)}"
                    )
                    # Persist test metrics to disk so they survive the run.
                    self._save_best_program(best)
                except Exception as e:
                    logger.warning(f"Test-mode re-evaluation failed: {e}")

        finally:
            # Stop the monitor
            early_stopped = (
                self.discovery_controller is not None
                and self.discovery_controller.early_stopping_triggered
            )
            if self.discovery_controller is not None:
                self.discovery_controller.close()
            self.discovery_controller = None

            if monitor_server:
                try:
                    reason = "early_stopping" if early_stopped else "completed"
                    monitor_server.push_event({"type": "discovery_complete", "reason": reason})
                    time.sleep(0.5)
                    monitor_server.stop()
                except Exception:
                    logger.debug("Failed to stop monitor server", exc_info=True)

        # Get the best program
        best_program = self._get_best_program()
        if best_program:
            status = "early stopping" if early_stopped else "completed"
            logger.info(f"Discovery {status}. Best: {format_metrics(best_program.metrics)}")
            self._save_best_program(best_program)
            return best_program

        logger.warning("No valid programs found")
        return None

    # ------------------------------------------------------------------
    # Initial program
    # ------------------------------------------------------------------

    async def _add_initial_program(self, start_iteration: int) -> None:
        logger.info("Adding initial program to database")
        program_id = str(uuid.uuid4())

        initial_image_path = None
        if self.config.language == "image":
            logger.info("Generating initial image from seed text...")
            img_dir = os.path.join(self.output_dir, "generated_images")
            try:
                result = await self.discovery_controller.llms.generate(
                    system_message="Generate an image based on the following description. Also provide brief reasoning about your creative choices.",
                    messages=[{"role": "user", "content": self.initial_program_solution}],
                    image_output=True,
                    output_dir=img_dir,
                    program_id=program_id,
                )
                initial_image_path = result.image_path
                logger.info(f"Initial image: {initial_image_path}")
            except Exception as e:
                logger.warning(f"Failed to generate initial image: {e}")

        eval_input = (
            initial_image_path
            if self.config.language == "image" and initial_image_path
            else self.initial_program_solution
        )
        eval_result = await self.discovery_controller.evaluator.evaluate_program(
            eval_input, program_id
        )
        metrics = eval_result.metrics

        if not initial_image_path and isinstance(metrics.get("image_path"), str):
            initial_image_path = metrics.pop("image_path")

        program = get_program(
            self.config, self.initial_program_solution, program_id, metrics, start_iteration
        )
        program.artifacts = eval_result.artifacts

        if initial_image_path:
            program.metadata = program.metadata or {}
            program.metadata["image_path"] = initial_image_path

        self.database.add(program)
        try:
            self.database.initial_program_id = program.id
            self.database.initial_program_score = get_score(program.metrics or {})
        except Exception as e:
            logger.warning(f"Failed to set initial program score: {e}")

    # ------------------------------------------------------------------
    # Monitor and feedback setup
    # ------------------------------------------------------------------

    def _start_monitor(self, max_iterations: int):
        if not self.config.monitor.enabled:
            return None
        try:
            from skydiscover.extras.monitor import MonitorServer, create_monitor_callback

            server = MonitorServer(
                host=self.config.monitor.host,
                port=self.config.monitor.port,
                max_solution_length=self.config.monitor.max_solution_length,
            )
            server.set_config_summary(f"{self.name} | max_iter={max_iterations}")
            server.start()

            callback = create_monitor_callback(server, self.database, time.time())
            self.discovery_controller.monitor_callback = callback

            url = f"http://localhost:{server.port}/"
            print(f"\n  Live monitor: {url}\n", flush=True)
            logger.info(f"Live monitor: {url}")
            return server
        except Exception as e:
            logger.warning(f"Failed to start monitor: {e}")
            return None

    def _setup_human_feedback(self, monitor_server) -> None:
        if not (self.config.human_feedback_enabled or monitor_server):
            return
        try:
            from skydiscover.context_builder import HumanFeedbackReader

            path = self.config.human_feedback_file or os.path.join(
                self.output_dir, "human_feedback.md"
            )
            mode = getattr(self.config, "human_feedback_mode", "append")
            reader = HumanFeedbackReader(path, mode=mode)
            self.discovery_controller.feedback_reader = reader
            if monitor_server:
                monitor_server.set_feedback_reader(reader)
            logger.info(f"Human feedback: {path}")
        except Exception as e:
            logger.warning(f"Failed to set up human feedback: {e}")

    def _setup_monitor_summary(self, monitor_server) -> None:
        if not (monitor_server and self.config.monitor.summary_model):
            return
        try:
            monitor_server.configure_summary(
                model=self.config.monitor.summary_model,
                api_key=self.config.monitor.summary_api_key or "",
                api_base=self.config.monitor.summary_api_base,
                top_k=self.config.monitor.summary_top_k,
                interval=self.config.monitor.summary_interval,
            )
        except Exception as e:
            logger.warning(f"Failed to configure AI summary: {e}")

    def _push_existing_to_monitor(self) -> None:
        if not (self.discovery_controller.monitor_callback and self.database.programs):
            return
        for prog in self.database.programs.values():
            try:
                self.discovery_controller.monitor_callback(
                    prog, getattr(prog, "iteration_found", 0)
                )
            except Exception:
                logger.debug("Monitor callback failed for program %s", prog.id, exc_info=True)
        logger.info(f"Pushed {len(self.database.programs)} existing program(s) to monitor")

    def _install_signal_handlers(self) -> None:
        def on_signal(signum, frame):
            logger.info(f"Signal {signum} received, shutting down...")
            if self.discovery_controller is not None:
                self.discovery_controller.request_shutdown()

            def force_exit(signum, frame):
                sys.exit(128 + signum)

            # After the first termination signal, ensure subsequent SIGINT/SIGTERM
            # cause an immediate exit instead of re-running the soft handler.
            signal.signal(signal.SIGINT, force_exit)
            signal.signal(signal.SIGTERM, force_exit)

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)

    # ------------------------------------------------------------------
    # Checkpointing and saving
    # ------------------------------------------------------------------

    def _sync_database(self) -> None:
        """Ensure we have the controller's latest database"""
        db = getattr(self.discovery_controller, "database", None)
        if db is not None and db is not self.database:
            self.database = db

    def _setup_logging(self) -> None:
        log_dir = self.config.log_dir or os.path.join(self.output_dir, "logs")
        setup_search_logging(log_level=self.config.log_level, log_dir=log_dir, name=self.name)

    def _load_initial_program(self) -> str:
        with open(self.initial_program_path, "r") as f:
            return f.read()

    def _save_checkpoint(self, iteration: int) -> None:
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}")
        os.makedirs(checkpoint_path, exist_ok=True)

        self.database.save(checkpoint_path, iteration)

        best = self._get_best_program()
        if best:
            with open(
                os.path.join(checkpoint_path, f"best_program{self.file_extension}"), "w"
            ) as f:
                f.write(best.solution)
            with open(os.path.join(checkpoint_path, "best_program_info.json"), "w") as f:
                from skydiscover.search.utils.checkpoint_manager import SafeJSONEncoder

                json.dump(
                    {
                        "id": best.id,
                        "generation": best.generation,
                        "iteration": best.iteration_found,
                        "current_iteration": iteration,
                        "metrics": best.metrics,
                        "language": best.language,
                        "timestamp": best.timestamp,
                        "saved_at": time.time(),
                    },
                    f,
                    indent=2,
                    cls=SafeJSONEncoder,
                )
            logger.info(f"Checkpoint {iteration}: best={format_metrics(best.metrics)}")

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.database.load(checkpoint_path)
        logger.info(f"Loaded checkpoint (iteration {self.database.last_iteration})")

    def _get_best_program(self) -> Optional[Program]:
        if self.database.best_program_id:
            prog = self.database.get(self.database.best_program_id)
            if prog:
                return prog
        return self.database.get_best_program()

    def _save_best_program(self, program: Program) -> None:
        best_dir = os.path.join(self.output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)

        code_path = os.path.join(best_dir, f"best_program{self.file_extension}")
        with open(code_path, "w") as f:
            f.write(program.solution)

        info_path = os.path.join(best_dir, "best_program_info.json")
        with open(info_path, "w") as f:
            from skydiscover.search.utils.checkpoint_manager import SafeJSONEncoder

            json.dump(
                {
                    "id": program.id,
                    "generation": program.generation,
                    "iteration": program.iteration_found,
                    "timestamp": program.timestamp,
                    "parent_id": program.parent_id,
                    "metrics": program.metrics,
                    "language": program.language,
                    "saved_at": time.time(),
                },
                f,
                indent=2,
                cls=SafeJSONEncoder,
            )

        if self.config.language == "image" and program.metadata:
            img = program.metadata.get("image_path")
            if img and os.path.exists(img):
                import shutil

                shutil.copy2(img, os.path.join(best_dir, "best_image" + os.path.splitext(img)[1]))

        logger.info(f"Best program saved to {best_dir}")
