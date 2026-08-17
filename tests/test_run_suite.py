from run_suite import Experiment, aggregate, build_experiments, command_for, parse_args


def test_build_experiments_covers_baselines_ranks_and_seeds():
    experiments = build_experiments(seeds=[13, 42], ranks=[2, 8])

    assert experiments == [
        Experiment("frozen", 13),
        Experiment("lora", 13, 2),
        Experiment("lora", 13, 8),
        Experiment("full", 13),
        Experiment("frozen", 42),
        Experiment("lora", 42, 2),
        Experiment("lora", 42, 8),
        Experiment("full", 42),
    ]


def test_command_resumes_incomplete_run(tmp_path):
    args = parse_args(["--output-dir", str(tmp_path), "--device", "cpu"])
    run_dir = tmp_path / "lora-r8-seed42"
    run_dir.mkdir()
    (run_dir / "latest.pt").touch()

    command = command_for(Experiment("lora", 42, 8), run_dir, args)

    assert "--resume-from" in command
    assert command[command.index("--rank") + 1] == "8"
    assert command[command.index("--precision") + 1] == "auto"
    assert command[command.index("--gradient-accumulation-steps") + 1] == "1"


def test_aggregate_reports_mean_and_sample_standard_deviation():
    rows = [
        {
            "mode": "lora",
            "rank": 8,
            "trainable_parameters": 10,
            "trainable_fraction": 0.1,
            "validation_accuracy": 0.4,
            "validation_macro_f1": 0.3,
            "test_accuracy": 0.4,
            "test_macro_f1": 0.3,
            "train_seconds": 8.0,
            "wall_seconds": 10.0,
            "peak_cuda_memory_mb": 100.0,
            "best_checkpoint_mb": 1.0,
        },
        {
            "mode": "lora",
            "rank": 8,
            "trainable_parameters": 10,
            "trainable_fraction": 0.1,
            "validation_accuracy": 0.6,
            "validation_macro_f1": 0.5,
            "test_accuracy": 0.6,
            "test_macro_f1": 0.5,
            "train_seconds": 16.0,
            "wall_seconds": 20.0,
            "peak_cuda_memory_mb": 120.0,
            "best_checkpoint_mb": 1.2,
        },
    ]

    result = aggregate(rows)[0]

    assert result["runs"] == 2
    assert result["test_accuracy_mean"] == 0.5
    assert result["test_accuracy_std"] > 0
