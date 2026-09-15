import inspect
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

from skydiscover.optimize.config import DatabaseConfig
from skydiscover.optimize.search.base_database import Program, ProgramDatabase
from skydiscover.optimize.search.utils.discovery_utils import load_database_from_file


def _error_response(error_message: str) -> Dict[str, Any]:
    """Create a standardized error response dictionary."""
    return {"validity": 0, "error": error_message, "combined_score": None}


def _verify_metrics_preserved(
    original_metrics: Dict[str, Any],
    stored_metrics: Dict[str, Any],
    operation: str,
    program_id: str,
) -> str:
    """
    Verify that all metrics from original_metrics are present in stored_metrics
    with the same values. Returns error message if verification fails, empty string if OK.
    """
    original_keys = set(original_metrics.keys())

    for key in original_keys:
        if key not in stored_metrics:
            return (
                f"Metric '{key}' was deleted from program '{program_id}' during {operation}. "
                f"Original metrics: {list(original_keys)}, stored metrics: {list(stored_metrics.keys())}"
            )

        original_value = original_metrics[key]
        stored_value = stored_metrics[key]

        if isinstance(original_value, (int, float)) and isinstance(stored_value, (int, float)):
            if abs(float(original_value) - float(stored_value)) > 1e-10:
                return (
                    f"Metric '{key}' value was modified in program '{program_id}' during {operation}: "
                    f"original={original_value}, stored={stored_value}. "
                    f"Metric values must remain unchanged."
                )
        elif original_value != stored_value:
            return (
                f"Metric '{key}' value was modified in program '{program_id}' during {operation}: "
                f"original={original_value!r}, stored={stored_value!r}. "
                f"Metric values must remain unchanged."
            )

    return ""


def evaluate(program_path: str, fast_mode: bool = False) -> Dict[str, Any]:
    """
    Validate a search algorithm by testing its database implementation.

    Checks:
    1. Structural: Class name (EvolvedProgramDatabase), inheritance, sample() signature
    2. add() operations: Metric preservation for single and multiple programs
    3. sample() return type: Tuple[Dict[str, Program], Dict[str, List[Program]]]
    4. sample() metrics preservation: Returned and stored programs maintain all original metrics
    5. Edge cases: Programs with error strings in metrics
    6. Migration compatibility: Can handle base Program instances (not just EvolvedProgram)

    Returns:
        If valid: {"validity": 1, "combined_score": 0.0}
        If invalid: {"validity": 0, "error": error_message, "combined_score": None}
    """
    NUM_PROGRAMS_TO_ADD = 5 if fast_mode else 10
    NUM_SAMPLE_ITERATIONS = 2 if fast_mode else 10
    NUM_ERROR_SAMPLE_ITERATIONS = 2 if fast_mode else 5
    NUM_MIGRATION_PROGRAMS = 3 if fast_mode else 5
    NUM_MIGRATION_SAMPLES = 1 if fast_mode else 3

    try:
        try:
            database_class, program_class = load_database_from_file(program_path)
        except Exception as e:
            return _error_response(f"Failed to load database class: {str(e)}")

        try:
            if database_class.__name__ != "EvolvedProgramDatabase":
                return _error_response(
                    f"Database class must be named 'EvolvedProgramDatabase', "
                    f"got '{database_class.__name__}'"
                )

            if not issubclass(database_class, ProgramDatabase):
                return _error_response("EvolvedProgramDatabase must inherit from ProgramDatabase")

            if not issubclass(program_class, Program):
                return _error_response(
                    "EvolvedProgram must inherit from Program (base_database.Program)"
                )

            sample_sig = inspect.signature(database_class.sample)
            params = list(sample_sig.parameters.values())
            if len(params) < 2 or params[1].name != "num_context_programs":
                return _error_response(
                    "sample() must have signature sample(self, num_context_programs: Optional[int] = 4, **kwargs). "
                    "Expected second parameter named 'num_context_programs'."
                )

            return_annotation = sample_sig.return_annotation
            if return_annotation != inspect.Signature.empty:
                import typing

                if hasattr(typing, "get_args") and hasattr(typing, "get_origin"):
                    origin = typing.get_origin(return_annotation)
                    args = typing.get_args(return_annotation)
                    if origin is tuple or (hasattr(typing, "Tuple") and origin is typing.Tuple):
                        if len(args) != 2:
                            return _error_response(
                                f"sample() return type must be Tuple[Dict[str, Program], Dict[str, List[Program]]] "
                                f"(2 elements), got {return_annotation} with {len(args)} elements"
                            )
        except Exception as e:
            return _error_response(f"Failed structural checks for EvolvedProgramDatabase: {str(e)}")

        # Ensure DIVERGE_LABEL and REFINE_LABEL exist for evolved databases that use them in __init__
        # (coevolve_controller assigns real values via _assign_labels_to_db; evaluator uses empty defaults)
        if not hasattr(database_class, "DIVERGE_LABEL"):
            database_class.DIVERGE_LABEL = ""
        if not hasattr(database_class, "REFINE_LABEL"):
            database_class.REFINE_LABEL = ""

        try:
            test_config = DatabaseConfig()
            db = database_class("test_db", test_config)
        except Exception as e:
            return _error_response(f"Failed to initialize database: {str(e)}")

        all_original_metrics = {}

        # Test: Add one program and verify
        try:
            test_program = program_class(
                id="test_program_1",
                solution="def test(): return 1",
                language="python",
                metrics={"score": 0.5, "combined_score": 0.5},
            )
            test_program.iteration_found = 0
            original_metrics = test_program.metrics.copy()
            all_original_metrics["test_program_1"] = original_metrics.copy()

            program_id = db.add(test_program, iteration=0)
            if program_id != "test_program_1":
                return _error_response(f"add() returned unexpected ID: {program_id}")

            db.initial_program_id = program_id

            stored_program = db.get("test_program_1")
            if stored_program is None:
                return _error_response("Program not found in database after add()")

            error_msg = _verify_metrics_preserved(
                original_metrics, stored_program.metrics, "add()", "test_program_1"
            )
            if error_msg:
                return _error_response(error_msg)

            error_msg = _verify_metrics_preserved(
                original_metrics, test_program.metrics, "add() (original object)", "test_program_1"
            )
            if error_msg:
                return _error_response(error_msg)
        except Exception as e:
            return _error_response(f"Failed to add or verify program: {str(e)}")

        # Test: Add programs and sample from them
        try:
            for i in range(NUM_PROGRAMS_TO_ADD):
                program = program_class(
                    id=f"program_{i}",
                    solution=f"def func_{i}(): return {i}",
                    language="python",
                    metrics={"score": float(i) / 10.0, "combined_score": float(i) / 10.0},
                )
                program.iteration_found = i
                original_metrics = program.metrics.copy()
                all_original_metrics[f"program_{i}"] = original_metrics.copy()
                db.add(program, iteration=i)

                if not fast_mode or i == NUM_PROGRAMS_TO_ADD - 1:
                    stored_program = db.get(f"program_{i}")
                    if stored_program is None:
                        return _error_response(
                            f"Program 'program_{i}' not found in database after add()"
                        )

                    error_msg = _verify_metrics_preserved(
                        all_original_metrics[f"program_{i}"],
                        stored_program.metrics,
                        "add()",
                        f"program_{i}",
                    )
                    if error_msg:
                        return _error_response(error_msg)

            if len(db.programs) < NUM_PROGRAMS_TO_ADD:
                return _error_response(
                    f"Expected at least {NUM_PROGRAMS_TO_ADD} programs, found {len(db.programs)}"
                )

            for sample_iter in range(NUM_SAMPLE_ITERATIONS):
                sample_result = db.sample(num_context_programs=4)

                if not isinstance(sample_result, tuple):
                    return _error_response(
                        f"sample() must return a tuple, got {type(sample_result)} on iteration {sample_iter}"
                    )
                if len(sample_result) != 2:
                    return _error_response(
                        f"sample() must return a tuple of 2 elements (parent_dict, context_programs_dict), got {len(sample_result)} elements on iteration {sample_iter}"
                    )

                parent_dict, context_programs_dict = sample_result

                if not isinstance(parent_dict, dict):
                    return _error_response(
                        f"sample() first element must be a Dict[str, Program], got {type(parent_dict)} on iteration {sample_iter}"
                    )

                if len(parent_dict) != 1:
                    return _error_response(
                        f"sample() must return exactly one parent program in parent_dict, "
                        f"got {len(parent_dict)} parents with keys {list(parent_dict.keys())} on iteration {sample_iter}"
                    )

                parent_label = list(parent_dict.keys())[0]
                parent = parent_dict[parent_label]

                if parent is None:
                    return _error_response(
                        f"sample() returned None for parent on iteration {sample_iter}"
                    )
                if not isinstance(parent, Program):
                    return _error_response(
                        f"sample() parent_dict value must be a Program, got {type(parent)} on iteration {sample_iter}"
                    )
                if parent.id not in db.programs:
                    return _error_response(f"Sampled parent (id={parent.id}) not found in database")

                if not isinstance(context_programs_dict, dict):
                    return _error_response(
                        f"sample() second element must be a Dict[str, List[Program]], got {type(context_programs_dict)} on iteration {sample_iter}"
                    )

                all_context_programs = []
                for label, prog_list in context_programs_dict.items():
                    if not isinstance(prog_list, list):
                        return _error_response(
                            f"sample() context_programs_dict['{label}'] must be a list, got {type(prog_list)} on iteration {sample_iter}"
                        )
                    for prog in prog_list:
                        if not isinstance(prog, Program):
                            return _error_response(
                                f"sample() context_programs_dict['{label}'] contains non-Program object: {type(prog)} on iteration {sample_iter}"
                            )
                    all_context_programs.extend(prog_list)

                requested_num = 4
                max_possible = min(requested_num, len(db.programs))
                if len(all_context_programs) > max_possible:
                    return _error_response(
                        f"sample() returned {len(all_context_programs)} total context programs, which exceeds the maximum possible "
                        f"({max_possible} given {len(db.programs)} programs in database) "
                        f"for num_context_programs={requested_num} on iteration {sample_iter}."
                    )

                stored_parent = db.get(parent.id)
                if stored_parent is None:
                    return _error_response(
                        f"Sampled parent (id={parent.id}) not found in database via get()"
                    )

                if parent.id in all_original_metrics:
                    original_parent_metrics = all_original_metrics[parent.id]
                else:
                    return _error_response(
                        f"Program '{parent.id}' not found in original_metrics tracking. "
                        f"This indicates a bug in the test - all programs should have their original metrics tracked."
                    )

                error_msg = _verify_metrics_preserved(
                    original_parent_metrics, parent.metrics, "sample() (parent returned)", parent.id
                )
                if error_msg:
                    return _error_response(error_msg)

                error_msg = _verify_metrics_preserved(
                    original_parent_metrics,
                    stored_parent.metrics,
                    "sample() (parent stored)",
                    parent.id,
                )
                if error_msg:
                    return _error_response(error_msg)

                for context_prog in all_context_programs:
                    if context_prog.id not in db.programs:
                        return _error_response(
                            f"Sampled context program (id={context_prog.id}) not found in database"
                        )

                    stored_context = db.get(context_prog.id)
                    if stored_context is None:
                        return _error_response(
                            f"Sampled context program (id={context_prog.id}) not found in database via get()"
                        )

                    if context_prog.id in all_original_metrics:
                        original_context_metrics = all_original_metrics[context_prog.id]
                    else:
                        return _error_response(
                            f"Program '{context_prog.id}' not found in original_metrics tracking. "
                            f"This indicates a bug in the test - all programs should have their original metrics tracked."
                        )

                    error_msg = _verify_metrics_preserved(
                        original_context_metrics,
                        context_prog.metrics,
                        "sample() (context program returned)",
                        context_prog.id,
                    )
                    if error_msg:
                        return _error_response(error_msg)

                    error_msg = _verify_metrics_preserved(
                        original_context_metrics,
                        stored_context.metrics,
                        "sample() (context program stored)",
                        context_prog.id,
                    )
                    if error_msg:
                        return _error_response(error_msg)
        except Exception as e:
            return _error_response(f"Failed to add 10 programs and sample: {str(e)}")

        # Test: Handle programs with error strings in metrics
        try:
            error_program = program_class(
                id="error_program",
                solution="def error_func(): return 0",
                language="python",
                metrics={
                    "combined_score": 0.0,
                    "error": "Stage 1 error: cannot access local variable 'r' where it is not associated with a value",
                    "runs_successfully": 0.0,
                },
            )
            error_program.iteration_found = 10
            original_error_metrics = error_program.metrics.copy()
            all_original_metrics["error_program"] = original_error_metrics.copy()
            db.add(error_program, iteration=10)

            stored_error_program = db.get("error_program")
            if stored_error_program is None:
                return _error_response("Program with error string in metrics not found after add()")

            error_msg = _verify_metrics_preserved(
                original_error_metrics, stored_error_program.metrics, "add()", "error_program"
            )
            if error_msg:
                return _error_response(error_msg)

            for sample_iter in range(NUM_ERROR_SAMPLE_ITERATIONS):
                sample_result = db.sample(num_context_programs=3)

                if not isinstance(sample_result, tuple):
                    return _error_response(
                        f"sample() must return a tuple when testing error strings, got {type(sample_result)} on iteration {sample_iter}"
                    )
                if len(sample_result) != 2:
                    return _error_response(
                        f"sample() must return a tuple of 2 elements when testing error strings, got {len(sample_result)} elements on iteration {sample_iter}"
                    )

                parent_dict, context_programs_dict = sample_result

                if not isinstance(parent_dict, dict):
                    return _error_response(
                        f"sample() first element must be a Dict[str, Program] when testing error strings, got {type(parent_dict)} on iteration {sample_iter}"
                    )

                if len(parent_dict) != 1:
                    return _error_response(
                        f"sample() must return exactly one parent program when testing error strings, "
                        f"got {len(parent_dict)} parents with keys {list(parent_dict.keys())} on iteration {sample_iter}"
                    )

                parent_label = list(parent_dict.keys())[0]
                parent = parent_dict[parent_label]

                if parent is None:
                    return _error_response(
                        f"sample() returned None for parent when testing error strings (iteration {sample_iter})"
                    )
                if not isinstance(parent, Program):
                    return _error_response(
                        f"sample() parent_dict value must be a Program when testing error strings, got {type(parent)} on iteration {sample_iter}"
                    )

                if not isinstance(context_programs_dict, dict):
                    return _error_response(
                        f"sample() second element must be a Dict[str, List[Program]] when testing error strings, got {type(context_programs_dict)} on iteration {sample_iter}"
                    )

                all_context_programs = []
                for label, prog_list in context_programs_dict.items():
                    if not isinstance(prog_list, list):
                        return _error_response(
                            f"sample() context_programs_dict['{label}'] must be a list when testing error strings, got {type(prog_list)} on iteration {sample_iter}"
                        )
                    for prog in prog_list:
                        if not isinstance(prog, Program):
                            return _error_response(
                                f"sample() context_programs_dict['{label}'] contains non-Program object when testing error strings: {type(prog)} on iteration {sample_iter}"
                            )
                    all_context_programs.extend(prog_list)

                requested_num = 3
                max_possible = min(requested_num, len(db.programs))
                if len(all_context_programs) > max_possible:
                    return _error_response(
                        f"sample() returned {len(all_context_programs)} total context programs when testing error strings, which exceeds the maximum possible "
                        f"({max_possible} given {len(db.programs)} programs in database) "
                        f"for num_context_programs={requested_num} on iteration {sample_iter}."
                    )

                stored_parent = db.get(parent.id)
                if stored_parent is None:
                    return _error_response(
                        f"Parent (id={parent.id}) not found in database when testing error strings"
                    )

                if parent.id in all_original_metrics:
                    original_parent_metrics = all_original_metrics[parent.id]
                else:
                    return _error_response(
                        f"Program '{parent.id}' not found in original_metrics tracking when testing error strings."
                    )

                error_msg = _verify_metrics_preserved(
                    original_parent_metrics,
                    parent.metrics,
                    "sample() (parent with error strings)",
                    parent.id,
                )
                if error_msg:
                    return _error_response(error_msg)

                error_msg = _verify_metrics_preserved(
                    original_parent_metrics,
                    stored_parent.metrics,
                    "sample() (parent stored with error strings)",
                    parent.id,
                )
                if error_msg:
                    return _error_response(error_msg)

                for context_prog in all_context_programs:
                    stored_context = db.get(context_prog.id)
                    if stored_context is None:
                        return _error_response(
                            f"Other context program (id={context_prog.id}) not found in database when testing error strings"
                        )

                    if context_prog.id in all_original_metrics:
                        original_context_metrics = all_original_metrics[context_prog.id]
                    else:
                        return _error_response(
                            f"Program '{context_prog.id}' not found in original_metrics tracking when testing error strings."
                        )

                    error_msg = _verify_metrics_preserved(
                        original_context_metrics,
                        context_prog.metrics,
                        "sample() (context program with error strings)",
                        context_prog.id,
                    )
                    if error_msg:
                        return _error_response(error_msg)

                    error_msg = _verify_metrics_preserved(
                        original_context_metrics,
                        stored_context.metrics,
                        "sample() (context program stored with error strings)",
                        context_prog.id,
                    )
                    if error_msg:
                        return _error_response(error_msg)
        except Exception as e:
            return _error_response(
                f"Failed to handle programs with error strings in metrics: {str(e)}"
            )

        # Test: Verify existing metrics are not modified by add()
        try:
            metrics_test_program = program_class(
                id="metrics_test_program",
                solution="def test_metrics(): return 42",
                language="python",
                metrics={
                    "combined_score": 0.85,
                    "correlation": 0.92,
                    "success_rate": 1.0,
                    "error": "No error",
                    "execution_time": 0.123,
                },
            )
            metrics_test_program.iteration_found = 11

            original_metrics = metrics_test_program.metrics.copy()
            all_original_metrics["metrics_test_program"] = original_metrics.copy()

            db.add(metrics_test_program, iteration=11)

            stored_metrics_program = db.get("metrics_test_program")
            if stored_metrics_program is None:
                return _error_response("Metrics test program not found in database after add()")

            error_msg = _verify_metrics_preserved(
                original_metrics, stored_metrics_program.metrics, "add()", "metrics_test_program"
            )
            if error_msg:
                return _error_response(error_msg)

            error_msg = _verify_metrics_preserved(
                original_metrics,
                metrics_test_program.metrics,
                "add() (original object)",
                "metrics_test_program",
            )
            if error_msg:
                return _error_response(error_msg)
        except Exception as e:
            return _error_response(f"Failed to verify metrics immutability: {str(e)}")

        # Test: Migration scenario - add base Program instances (not EvolvedProgram)
        try:
            migration_test_db = database_class("migration_test_db", test_config)

            for i in range(NUM_MIGRATION_PROGRAMS):
                base_program = Program(
                    id=f"migrated_program_{i}",
                    solution=f"def migrated_func_{i}(): return {i * 10}",
                    language="python",
                    metrics={"combined_score": float(i) / 5.0, "score": float(i) / 5.0},
                )
                base_program.iteration_found = i
                migration_test_db.add(base_program, iteration=i)

            migration_test_db.initial_program_id = "migrated_program_0"

            if len(migration_test_db.programs) < NUM_MIGRATION_PROGRAMS:
                return _error_response(
                    f"Migration test: Expected at least {NUM_MIGRATION_PROGRAMS} programs after adding base Program instances, "
                    f"found {len(migration_test_db.programs)}"
                )

            for sample_iter in range(NUM_MIGRATION_SAMPLES):
                try:
                    sample_result = migration_test_db.sample(num_context_programs=3)
                except ValueError as e:
                    if "No candidates to sample" in str(e):
                        return _error_response(
                            f"Migration test FAILED: sample() raised 'No candidates to sample' when database "
                            f"contains {len(migration_test_db.programs)} programs. This typically happens when "
                            f"the implementation uses 'isinstance(p, EvolvedProgram)' to filter programs, "
                            f"which fails for migrated programs that are instances of the base Program class. "
                            f"Fix: Use 'self.programs.values()' directly without isinstance checks, or check "
                            f"against the base Program class instead."
                        )
                    raise

                if not isinstance(sample_result, tuple) or len(sample_result) != 2:
                    return _error_response(
                        f"Migration test: sample() must return a tuple of 2 elements, "
                        f"got {type(sample_result)} with {len(sample_result) if isinstance(sample_result, tuple) else 'N/A'} elements"
                    )

                parent_dict, context_programs_dict = sample_result

                if not isinstance(parent_dict, dict) or len(parent_dict) != 1:
                    return _error_response(
                        f"Migration test: sample() must return exactly one parent in parent_dict, "
                        f"got {len(parent_dict) if isinstance(parent_dict, dict) else type(parent_dict)}"
                    )

                parent = list(parent_dict.values())[0]
                if parent is None:
                    return _error_response(
                        f"Migration test: sample() returned None for parent on iteration {sample_iter}"
                    )
                if not isinstance(parent, Program):
                    return _error_response(
                        f"Migration test: sample() parent must be a Program, got {type(parent)}"
                    )

                if not isinstance(context_programs_dict, dict):
                    return _error_response(
                        f"Migration test: sample() second element must be a dict, got {type(context_programs_dict)}"
                    )

                for label, prog_list in context_programs_dict.items():
                    if not isinstance(prog_list, list):
                        return _error_response(
                            f"Migration test: context_programs_dict['{label}'] must be a list, got {type(prog_list)}"
                        )
                    for prog in prog_list:
                        if not isinstance(prog, Program):
                            return _error_response(
                                f"Migration test: context program must be a Program, got {type(prog)}"
                            )
        except Exception as e:
            if "No candidates to sample" in str(e):
                return _error_response(
                    f"Migration test FAILED: {str(e)}. "
                    f"The database cannot handle programs that are instances of the base Program class "
                    f"(not EvolvedProgram). This breaks evox migration. "
                    f"Fix: Remove isinstance(p, EvolvedProgram) checks in sampling logic."
                )
            return _error_response(f"Migration test failed: {str(e)}")

        # Test: Verify num_context_programs contract
        try:
            sample_sig = inspect.signature(db.sample)
            if "num_context_programs" in sample_sig.parameters:
                requested_num = 3
                sample_result = db.sample(num_context_programs=requested_num)

                if not isinstance(sample_result, tuple):
                    return _error_response(
                        f"sample() must return a tuple, got {type(sample_result)}"
                    )
                if len(sample_result) != 2:
                    return _error_response(
                        f"sample() must return a tuple of 2 elements (parent_dict, context_programs_dict), got {len(sample_result)} elements"
                    )

                parent_dict, context_programs_dict = sample_result

                if not isinstance(parent_dict, dict):
                    return _error_response(
                        f"sample() first element must be a Dict[str, Program], got {type(parent_dict)}"
                    )
                if len(parent_dict) != 1:
                    return _error_response(
                        f"sample() must return exactly one parent program in parent_dict, "
                        f"got {len(parent_dict)} parents with keys {list(parent_dict.keys())}"
                    )

                parent_label = list(parent_dict.keys())[0]
                parent = parent_dict[parent_label]
                if parent is None or not isinstance(parent, Program):
                    return _error_response(
                        f"sample() parent_dict value must be a Program, got {type(parent)}"
                    )

                if not isinstance(context_programs_dict, dict):
                    return _error_response(
                        f"sample() second element must be a Dict[str, List[Program]], got {type(context_programs_dict)}"
                    )

                all_context_programs = []
                for label, prog_list in context_programs_dict.items():
                    if not isinstance(prog_list, list):
                        return _error_response(
                            f"sample() context_programs_dict['{label}'] must be a list, got {type(prog_list)}"
                        )
                    for prog in prog_list:
                        if not isinstance(prog, Program):
                            return _error_response(
                                f"sample() context_programs_dict['{label}'] contains non-Program object: {type(prog)}"
                            )
                    all_context_programs.extend(prog_list)

                max_possible = min(requested_num, len(db.programs))
                if len(all_context_programs) > max_possible:
                    return _error_response(
                        f"sample() returned {len(all_context_programs)} total context programs, which exceeds the maximum possible "
                        f"({max_possible} given {len(db.programs)} programs in database) "
                        f"for num_context_programs={requested_num}."
                    )
        except Exception as e:
            return _error_response(
                f"Failed to verify num_context_programs contract in sample(): {str(e)}"
            )

        return {
            "validity": 1,
            "combined_score": 0.0,
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        return _error_response(f"Unexpected error: {str(e)}\n{error_trace}")


def evaluate_batch(
    program_paths: List[str],
    fast_mode: bool = True,
    max_workers: int = 4,
    timeout_per_file: float = 30.0,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate multiple database implementations in parallel."""
    results = {}

    def eval_with_timeout(path: str) -> Tuple[str, Dict[str, Any]]:
        try:
            result = evaluate(path, fast_mode=fast_mode)
            return path, result
        except Exception as e:
            return path, _error_response(f"Evaluation failed: {str(e)}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(eval_with_timeout, path): path for path in program_paths}

        for future in as_completed(future_to_path, timeout=timeout_per_file * len(program_paths)):
            try:
                path, result = future.result(timeout=timeout_per_file)
                results[path] = result
            except Exception as e:
                path = future_to_path[future]
                results[path] = _error_response(f"Evaluation timed out or failed: {str(e)}")

    return results


if __name__ == "__main__":
    """
    Usage: uv run skydiscover/search/evox/database/database_evaluator.py /path/to/database_impl.py
    """
    import json
    import sys
    from pathlib import Path

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    fast_mode = "--fast" in flags

    max_workers = 4
    for flag in flags:
        if flag.startswith("--workers="):
            try:
                max_workers = int(flag.split("=")[1])
            except ValueError:
                pass

    if not args:
        print(
            "Usage: uv run skydiscover/search/evox/database/database_evaluator.py <database.py> [--fast] [--workers=N]"
        )
        sys.exit(1)

    program_paths = [str(Path(p).resolve()) for p in args]

    if len(program_paths) == 1:
        result = evaluate(program_paths[0], fast_mode=fast_mode)
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        import time

        start = time.time()
        results = evaluate_batch(program_paths, fast_mode=fast_mode, max_workers=max_workers)
        elapsed = time.time() - start

        valid_count = sum(1 for r in results.values() if r.get("validity") == 1)
        print(
            f"Evaluated {len(results)} files in {elapsed:.2f}s ({valid_count}/{len(results)} valid)"
        )
        print(json.dumps(results, indent=2, sort_keys=True))
