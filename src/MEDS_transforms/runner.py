"""This script is a helper utility to run entire pipelines from a single script.

To do this effectively, this runner functionally takes a "meta configuration" file that contains:

  1. The path to the pipeline configuration file.
  2. Configuration details for how to run each stage of the pipeline, including mappings to the underlying
     stage scripts and Hydra launcher configurations for each stage to control parallelism, resources, etc.
"""

import argparse
import logging
import subprocess
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

from .configs import PipelineConfig
from .configs.utils import OmegaConfResolver

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

logger = logging.getLogger(__name__)


def get_parallelization_args(
    parallelization_cfg: dict | DictConfig | None, default_parallelization_cfg: dict | DictConfig
) -> list[str]:
    """Extracts the specific parallelization arguments given the default and stage-specific configurations.

    Args:
        parallelization_cfg: The stage-specific parallelization configuration.
        default_parallelization_cfg: The default parallelization configuration.

    Returns:
        A list of command-line arguments for parallelization.

    Examples:
        >>> get_parallelization_args({}, {})
        []
        >>> get_parallelization_args(None, {"n_workers": 4})
        []
        >>> get_parallelization_args({"launcher": "joblib"}, {})
        ['--multirun', 'worker="range(0,1)"', 'hydra/launcher=joblib']
        >>> get_parallelization_args({"n_workers": 2, "launcher_params": "foo"}, {})
        Traceback (most recent call last):
            ...
        ValueError: If launcher_params is provided, launcher must also be provided.
        >>> get_parallelization_args({"n_workers": 2}, {})
        ['--multirun', 'worker="range(0,2)"']
        >>> get_parallelization_args({"launcher": "slurm"}, {"n_workers": 3, "launcher": "joblib"})
        ['--multirun', 'worker="range(0,3)"', 'hydra/launcher=slurm']
        >>> get_parallelization_args(
        ...     {"n_workers": 2, "launcher": "joblib"},
        ...     {"n_workers": 5, "launcher_params": {"foo": "bar"}},
        ... )
        ['--multirun', 'worker="range(0,2)"', 'hydra/launcher=joblib', 'hydra.launcher.foo=bar']
        >>> get_parallelization_args(
        ...     {"n_workers": 5, "launcher_params": {"biz": "baz"}, "launcher": "slurm"}, {}
        ... )
        ['--multirun', 'worker="range(0,5)"', 'hydra/launcher=slurm', 'hydra.launcher.biz=baz']
    """

    print("\n[DEBUG:get_parallelization_args] called", flush=True)
    print(f"[DEBUG:get_parallelization_args] parallelization_cfg = {parallelization_cfg}", flush=True)
    print(
        f"[DEBUG:get_parallelization_args] default_parallelization_cfg = {default_parallelization_cfg}",
        flush=True,
    )

    if parallelization_cfg is None:
        print("[DEBUG:get_parallelization_args] parallelization_cfg is None -> return []", flush=True)
        return []

    if len(parallelization_cfg) == 0 and len(default_parallelization_cfg) == 0:
        print("[DEBUG:get_parallelization_args] both cfgs empty -> return []", flush=True)
        return []

    if "n_workers" in parallelization_cfg:
        n_workers = parallelization_cfg["n_workers"]
        print(f"[DEBUG:get_parallelization_args] n_workers from stage cfg = {n_workers}", flush=True)
    elif "n_workers" in default_parallelization_cfg:
        n_workers = default_parallelization_cfg["n_workers"]
        print(f"[DEBUG:get_parallelization_args] n_workers from default cfg = {n_workers}", flush=True)
    else:
        n_workers = 1
        print("[DEBUG:get_parallelization_args] n_workers defaulted to 1", flush=True)

    parallelization_args = [
        "--multirun",
        f'worker="range(0,{n_workers})"',
    ]
    print(f"[DEBUG:get_parallelization_args] initial parallelization_args = {parallelization_args}", flush=True)

    if "launcher" in parallelization_cfg:
        launcher = parallelization_cfg["launcher"]
        print(f"[DEBUG:get_parallelization_args] launcher from stage cfg = {launcher}", flush=True)
    elif "launcher" in default_parallelization_cfg:
        launcher = default_parallelization_cfg["launcher"]
        print(f"[DEBUG:get_parallelization_args] launcher from default cfg = {launcher}", flush=True)
    else:
        launcher = None
        print("[DEBUG:get_parallelization_args] no launcher configured", flush=True)

    if launcher is None:
        if "launcher_params" in parallelization_cfg:
            print("[DEBUG:get_parallelization_args] ERROR: launcher_params without launcher", flush=True)
            raise ValueError("If launcher_params is provided, launcher must also be provided.")

        print(f"[DEBUG:get_parallelization_args] returning = {parallelization_args}", flush=True)
        return parallelization_args

    parallelization_args.append(f"hydra/launcher={launcher}")
    print(f"[DEBUG:get_parallelization_args] after launcher append = {parallelization_args}", flush=True)

    if "launcher_params" in parallelization_cfg:
        launcher_params = parallelization_cfg["launcher_params"]
        print(f"[DEBUG:get_parallelization_args] launcher_params from stage cfg = {launcher_params}", flush=True)
    elif "launcher_params" in default_parallelization_cfg:
        launcher_params = default_parallelization_cfg["launcher_params"]
        print(
            f"[DEBUG:get_parallelization_args] launcher_params from default cfg = {launcher_params}",
            flush=True,
        )
    else:
        launcher_params = {}
        print("[DEBUG:get_parallelization_args] no launcher_params configured", flush=True)

    for k, v in launcher_params.items():
        arg = f"hydra.launcher.{k}={v}"
        parallelization_args.append(arg)
        print(f"[DEBUG:get_parallelization_args] appended launcher param -> {arg}", flush=True)

    print(f"[DEBUG:get_parallelization_args] final parallelization_args = {parallelization_args}", flush=True)
    return parallelization_args


def run_stage(
    pipeline_config_fp: str,
    stage_runners_cfg: dict | DictConfig,
    pipeline_cfg: PipelineConfig,
    stage_name: str,
    cfg_overrides: list[str] | None = None,
    default_parallelization_cfg: dict | DictConfig | None = None,
    do_profile: bool = False,
    runner_fn: callable = subprocess.run,  # For dependency injection
):
    """Runs a single stage of the pipeline.

    Args:
        pipeline_config_fp: Path to the pipeline configuration on disk.
        stage_runners_cfg: The dictionary of stage runner configurations.
        stage_name: The name of the stage to run.
        default_parallelization_cfg: Default parallelization configuration.
        do_profile: Whether to enable Hydra profiling for the stage.
        runner_fn: Function used to actually run the stage command. This is
            primarily for dependency injection in testing.

    Raises:
        ValueError: If the stage fails to run.

    Examples:
        >>> def fake_shell_succeed(cmd, shell, capture_output):
        ...     print(cmd)
        ...     return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=b"", stderr=b"")
        >>> def fake_shell_fail(cmd, shell, capture_output):
        ...     print(cmd)
        ...     return subprocess.CompletedProcess(args=cmd, returncode=1, stdout=b"", stderr=b"")
        >>> stage_runners = {
        ...     "reshard_to_split": {"_script": "not used"},
        ...     "fit_vocabulary_indices": {},
        ...     "reorder_measurements": {"script": "baz_script"},
        ... }
        >>> pipeline_cfg = PipelineConfig(
        ...     stages=[
        ...         "reshard_to_split",
        ...         {"fit_vocabulary_indices": {"_script": "foobar"}},
        ...         "reorder_measurements",
        ...     ],
        ... )
        >>> run_stage(
        ...     "pipeline_config.yaml",
        ...     stage_runners,
        ...     pipeline_cfg,
        ...     "reshard_to_split",
        ...     runner_fn=fake_shell_succeed,
        ... )
        MEDS_transform-stage pipeline_config.yaml reshard_to_split stage=reshard_to_split
        >>> run_stage(
        ...     "pipeline_config.yaml",
        ...     stage_runners,
        ...     pipeline_cfg,
        ...     "fit_vocabulary_indices",
        ...     runner_fn=fake_shell_succeed,
        ... )
        foobar stage=fit_vocabulary_indices
        >>> run_stage(
        ...     "pipeline_config.yaml",
        ...     stage_runners,
        ...     pipeline_cfg,
        ...     "reorder_measurements",
        ...     runner_fn=fake_shell_succeed,
        ... )
        baz_script stage=reorder_measurements
        >>> run_stage(
        ...     "pipeline_config.yaml",
        ...     stage_runners,
        ...     pipeline_cfg,
        ...     "reorder_measurements",
        ...     do_profile=True,
        ...     runner_fn=fake_shell_succeed,
        ... )
        baz_script stage=reorder_measurements
            ++hydra.callbacks.profiler._target_=hydra_profiler.profiler.ProfilerCallback
        >>> stage_runners["reorder_measurements"]["parallelize"] = {"n_workers": 2}
        >>> run_stage(
        ...     "pipeline_config.yaml",
        ...     stage_runners,
        ...     pipeline_cfg,
        ...     "reorder_measurements",
        ...     runner_fn=fake_shell_succeed,
        ... )
        baz_script --multirun stage=reorder_measurements worker="range(0,2)"
        >>> run_stage(
        ...     "pipeline_config.yaml",
        ...     stage_runners,
        ...     pipeline_cfg,
        ...     "reorder_measurements",
        ...     runner_fn=fake_shell_fail,
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Stage reorder_measurements failed via ...

        Improper configurations also raise errors:

        >>> bad_runners = {"reshard_to_split": {"_base_stage": "belongs in the stage"}}
        >>> bad_cfg = PipelineConfig(stages=["reshard_to_split"])
        >>> run_stage(
        ...     "pipeline_config.yaml",
        ...     bad_runners,
        ...     bad_cfg,
        ...     "reshard_to_split",
        ...     runner_fn=fake_shell_succeed,
        ... )
        Traceback (most recent call last):
            ...
        ValueError: Put _base_stage args is in your pipeline config
    """

    print("\n" + "=" * 100, flush=True)
    print(f"[DEBUG:run_stage] ENTER stage_name={stage_name}", flush=True)
    print(f"[DEBUG:run_stage] pipeline_config_fp = {pipeline_config_fp}", flush=True)
    print(f"[DEBUG:run_stage] cfg_overrides = {cfg_overrides}", flush=True)
    print(f"[DEBUG:run_stage] default_parallelization_cfg (raw) = {default_parallelization_cfg}", flush=True)
    print(f"[DEBUG:run_stage] do_profile = {do_profile}", flush=True)

    if default_parallelization_cfg is None:
        default_parallelization_cfg = {}
        print("[DEBUG:run_stage] default_parallelization_cfg was None -> replaced with {}", flush=True)

    stage_config = pipeline_cfg.parsed_stages_by_name[stage_name].config
    stage_runner_config = stage_runners_cfg.get(stage_name, {})

    print(f"[DEBUG:run_stage] stage_config = {stage_config}", flush=True)
    print(f"[DEBUG:run_stage] stage_runner_config = {stage_runner_config}", flush=True)

    script = None
    if "script" in stage_runner_config:
        script = stage_runner_config.get("script")
        print(f"[DEBUG:run_stage] script from stage_runner_config['script'] = {script}", flush=True)
    elif "_script" in stage_config:
        script = stage_config.get("_script")
        print(f"[DEBUG:run_stage] script from stage_config['_script'] = {script}", flush=True)
    elif "_base_stage" in stage_runner_config:
        print("[DEBUG:run_stage] ERROR: _base_stage found in stage_runner_config", flush=True)
        raise ValueError("Put _base_stage args is in your pipeline config")
    else:
        script = f"MEDS_transform-stage {pipeline_config_fp} {stage_name}"
        print(f"[DEBUG:run_stage] script defaulted to = {script}", flush=True)

    command_parts = [
        script,
        f"stage={stage_name}",
    ]
    print(f"[DEBUG:run_stage] initial command_parts = {command_parts}", flush=True)

    if cfg_overrides:
        command_parts.extend(cfg_overrides)
        print(f"[DEBUG:run_stage] command_parts after cfg_overrides = {command_parts}", flush=True)
    else:
        print("[DEBUG:run_stage] no cfg_overrides provided", flush=True)

    parallelization_args = get_parallelization_args(
        stage_runner_config.get("parallelize", {}), default_parallelization_cfg
    )
    print(f"[DEBUG:run_stage] parallelization_args = {parallelization_args}", flush=True)

    if parallelization_args:
        multirun = parallelization_args.pop(0)
        print(f"[DEBUG:run_stage] multirun token = {multirun}", flush=True)
        command_parts = [*command_parts[:1], multirun, *command_parts[1:], *parallelization_args]
        print(f"[DEBUG:run_stage] command_parts after parallelization = {command_parts}", flush=True)
    else:
        print("[DEBUG:run_stage] no parallelization args added", flush=True)

    if do_profile:
        profiler_arg = "++hydra.callbacks.profiler._target_=hydra_profiler.profiler.ProfilerCallback"
        command_parts.append(profiler_arg)
        print(f"[DEBUG:run_stage] profiling enabled -> appended {profiler_arg}", flush=True)
    else:
        print("[DEBUG:run_stage] profiling disabled", flush=True)

    full_cmd = " ".join(command_parts)
    logger.info(f"Running command: {full_cmd}")

    print("-" * 100, flush=True)
    print(f"[DEBUG:run_stage] FINAL COMMAND for stage '{stage_name}':", flush=True)
    print(full_cmd, flush=True)
    print("-" * 100, flush=True)

    # https://stackoverflow.com/questions/21953835/run-subprocess-and-print-output-to-logging

    print("[DEBUG:run_stage] launching subprocess now", flush=True)

    # Intentionally no capture_output=True so that stdout/stderr of substeps are visible live in the console.
    command_out = runner_fn(full_cmd, shell=True)

    print(f"[DEBUG:run_stage] subprocess finished with returncode = {command_out.returncode}", flush=True)

    # stdout/stderr are only available when the injected runner returns them (e.g. in tests).
    stdout = getattr(command_out, "stdout", None)
    stderr = getattr(command_out, "stderr", None)

    if stdout is not None:
        try:
            stdout_decoded = stdout.decode() if isinstance(stdout, (bytes, bytearray)) else str(stdout)
            logger.info(f"Command output:\n{stdout_decoded}")
            print(f"[DEBUG:run_stage] captured stdout:\n{stdout_decoded}", flush=True)
        except Exception as e:
            print(f"[DEBUG:run_stage] failed to decode stdout: {e}", flush=True)

    if stderr is not None:
        try:
            stderr_decoded = stderr.decode() if isinstance(stderr, (bytes, bytearray)) else str(stderr)
            logger.info(f"Command error:\n{stderr_decoded}")
            print(f"[DEBUG:run_stage] captured stderr:\n{stderr_decoded}", flush=True)
        except Exception as e:
            print(f"[DEBUG:run_stage] failed to decode stderr: {e}", flush=True)

    if command_out.returncode != 0:
        print(f"[DEBUG:run_stage] Stage {stage_name} FAILED", flush=True)
        raise ValueError(
            f"Stage {stage_name} failed via {full_cmd} with return code {command_out.returncode}."
        )

    print(f"[DEBUG:run_stage] Stage {stage_name} completed successfully", flush=True)


def main(argv: list[str] | None = None) -> int:
    """Run an entire pipeline based on command line arguments."""

    print("\n" + "#" * 100, flush=True)
    print("[DEBUG:main] ENTER main()", flush=True)
    print(f"[DEBUG:main] argv = {argv}", flush=True)

    parser = argparse.ArgumentParser(description="MEDS-Transforms Pipeline Runner")
    parser.add_argument(
        "pipeline_config_fp",
        help="Path to the pipeline configuration file, either as a raw path or with pkg:// syntax.",
    )
    parser.add_argument(
        "--stage_runner_fp",
        type=str,
        default=None,
        help="Path to the stage runner configuration file. If not provided, no specialized runner is used.",
    )
    parser.add_argument(
        "--do_profile",
        default=False,
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="Enable Hydra profiling for the stages.",
    )
    parser.add_argument(
        "--overrides", nargs="*", default=[], help="Additional overrides for the pipeline configuration."
    )
    args = parser.parse_args(argv)

    print(f"[DEBUG:main] parsed args = {args}", flush=True)

    pipeline_config = PipelineConfig.from_arg(args.pipeline_config_fp, args.overrides)
    print("[DEBUG:main] pipeline_config loaded", flush=True)
    print(f"[DEBUG:main] pipeline_config.additional_params = {pipeline_config.additional_params}", flush=True)

    if pipeline_config.additional_params is None or "output_dir" not in pipeline_config.additional_params:
        print("[DEBUG:main] ERROR: output_dir missing in pipeline config/additional_params", flush=True)
        raise ValueError("Pipeline configuration or override must specify an 'output_dir'")

    log_dir = Path(pipeline_config.additional_params["output_dir"]) / ".logs"
    print(f"[DEBUG:main] log_dir = {log_dir}", flush=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    print("[DEBUG:main] ensured log_dir exists", flush=True)

    logging.basicConfig(filename=(log_dir / "pipeline.log"), level=logging.INFO)
    print(f"[DEBUG:main] logging configured to {(log_dir / 'pipeline.log')}", flush=True)

    logging.info("Running MEDS-Transforms Pipeline Runner with the following arguments:")
    for arg_name, arg_value in vars(args).items():
        logging.info(f"  {arg_name}: {arg_value}")
        print(f"[DEBUG:main] arg {arg_name} = {arg_value}", flush=True)

    global_done_file = log_dir / "_all_stages.done"
    print(f"[DEBUG:main] global_done_file = {global_done_file}", flush=True)
    if global_done_file.exists():
        logger.info("All stages are already complete. Exiting.")
        print("[DEBUG:main] global_done_file exists -> exiting early", flush=True)
        return 0

    stages = [s.name for s in pipeline_config.parsed_stages]
    print(f"[DEBUG:main] stages = {stages}", flush=True)
    if not stages:
        print("[DEBUG:main] ERROR: no stages found", flush=True)
        raise ValueError("Pipeline configuration must specify at least one stage.")

    stage_runners_cfg = load_yaml_file(args.stage_runner_fp)
    print(f"[DEBUG:main] stage_runners_cfg = {stage_runners_cfg}", flush=True)

    if "parallelize" in stage_runners_cfg:
        logger.info("Parallelization configuration loaded from stage runner")
        default_parallelization_cfg = stage_runners_cfg["parallelize"]
        print("[DEBUG:main] parallelization configuration loaded from stage runner", flush=True)
    elif "parallelize" in pipeline_config.additional_params:
        logger.info("Parallelization configuration loaded from pipeline config")
        default_parallelization_cfg = pipeline_config.additional_params["parallelize"]
        print("[DEBUG:main] parallelization configuration loaded from pipeline config", flush=True)
    else:
        logger.info("No parallelization configuration provided.")
        default_parallelization_cfg = None
        print("[DEBUG:main] no parallelization configuration provided", flush=True)

    print(f"[DEBUG:main] effective default_parallelization_cfg = {default_parallelization_cfg}", flush=True)

    for stage in stages:
        done_file = log_dir / f"{stage}.done"
        print("\n" + "-" * 100, flush=True)
        print(f"[DEBUG:main] considering stage = {stage}", flush=True)
        print(f"[DEBUG:main] done_file = {done_file}", flush=True)

        if done_file.exists():
            logger.info(f"Skipping stage {stage} as it is already complete.")
            print(f"[DEBUG:main] skipping stage {stage} as done file exists", flush=True)
        else:
            logger.info(f"Running stage: {stage}")
            print(f"[DEBUG:main] running stage: {stage}", flush=True)
            run_stage(
                args.pipeline_config_fp,
                stage_runners_cfg,
                pipeline_config,
                stage,
                cfg_overrides=args.overrides,
                default_parallelization_cfg=default_parallelization_cfg,
                do_profile=args.do_profile,
            )
            done_file.touch()
            print(f"[DEBUG:main] touched done file for stage {stage}", flush=True)

    global_done_file.touch()
    print(f"[DEBUG:main] touched global done file {global_done_file}", flush=True)
    print("[DEBUG:main] EXIT main() -> return 0", flush=True)
    print("#" * 100 + "\n", flush=True)
    return 0


@OmegaConfResolver
def load_yaml_file(path: str | None) -> dict | DictConfig:
    """Loads a YAML file as an OmegaConf object.

    Args:
        path: The path to the YAML file.

    Returns:
        The OmegaConf object representing the YAML file, or None if no path is provided.

    Raises:
        FileNotFoundError: If the file does not exist.

    Examples:
        >>> load_yaml_file(None)
        {}
        >>> load_yaml_file("nonexistent_file.yaml")
        Traceback (most recent call last):
            ...
        FileNotFoundError: File nonexistent_file.yaml does not exist.
        >>> with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
        ...     _ = f.write(b"foo: bar")
        ...     f.flush()
        ...     load_yaml_file(f.name)
        {'foo': 'bar'}
        >>> with tempfile.NamedTemporaryFile(suffix=".yaml") as f:
        ...     cfg = OmegaConf.create({"foo": "bar"})
        ...     OmegaConf.save(cfg, f.name)
        ...     load_yaml_file(f.name)
        {'foo': 'bar'}
    """

    print(f"[DEBUG:load_yaml_file] called with path = {path}", flush=True)

    if not path:
        print("[DEBUG:load_yaml_file] no path provided -> return {}", flush=True)
        return {}

    path = Path(path)
    print(f"[DEBUG:load_yaml_file] resolved path = {path}", flush=True)
    if not path.exists():
        print(f"[DEBUG:load_yaml_file] ERROR: file does not exist -> {path}", flush=True)
        raise FileNotFoundError(f"File {path} does not exist.")

    try:
        cfg = OmegaConf.load(path)
        print(f"[DEBUG:load_yaml_file] loaded with OmegaConf -> {cfg}", flush=True)
        return cfg
    except Exception as e:  # pragma: no cover
        logger.warning(f"Failed to load {path} as an OmegaConf: {e}. Trying as a plain YAML file.")
        print(f"[DEBUG:load_yaml_file] OmegaConf load failed: {e}", flush=True)
        yaml_text = path.read_text()
        print(f"[DEBUG:load_yaml_file] raw YAML text:\n{yaml_text}", flush=True)
        return yaml.load(yaml_text, Loader=Loader)