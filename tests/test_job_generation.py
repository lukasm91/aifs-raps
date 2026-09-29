"""Check the public benchmark defaults and generated launch sequence."""

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class JobGenerationTests(unittest.TestCase):
    def test_profiles_keep_reference_defaults_and_launch_steps(self):
        with tempfile.TemporaryDirectory(prefix=".generated-test-", dir=ROOT) as temp:
            output = Path(temp)
            for profile in ("highres", "throughput", "inference", "correctness"):
                subprocess.run(
                    ["python3", "-B", str(ROOT / "generate_aifs_jobs.py"),
                     "--config", "example", "--profile", profile, "--output", str(output)],
                    check=True, capture_output=True, text=True,
                )
            configs = output / "configs"
            jobs = output / "jobs"
            highres = (configs / "highres-real-64n-4g-nw4.args").read_text()
            throughput = (configs / "throughput-real-4n-4g-nw4.args").read_text()
            correctness = (configs / "correctness-real-1n-4g-nw8.args").read_text()
            self.assertIn("+hardware.num_gpus_per_ensemble=16\n", highres)
            self.assertIn("hardware.num_nodes=64\n", highres)
            self.assertIn("training.max_epochs=2\n", throughput)
            self.assertIn("dataloader.num_workers.training=4\n", throughput)
            self.assertIn("model.processor.num_chunks=4\n", throughput)
            self.assertIn("model.encoder.num_chunks=4\n", throughput)
            self.assertIn("hardware.paths.output=${oc.env:RUNDIR}/train-outputs/\n", correctness)
            self.assertNotIn("diagnostics.log.mlflow.enabled=false", highres + throughput)
            for name in ("highres-real-64n-4g-nw4", "throughput-real-4n-4g-nw4", "correctness-real-1n-4g-nw8"):
                script = (jobs / f"{name}.slurm").read_text()
                self.assertLess(script.index("--cfg all"), script.index('srun "${srun_args[@]}"'))
                self.assertIn("exec torchrun --no-python", script)
                self.assertIn("NCCL_DEBUG_SUBSYS=", script)
            inference_jobs = sorted(jobs.glob("inference-*.slurm"))
            self.assertEqual(2, len(inference_jobs))
            for script in jobs.glob("*.slurm"):
                subprocess.run(["bash", "-n", str(script)], check=True)


if __name__ == "__main__":
    unittest.main()
