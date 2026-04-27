from __future__ import annotations

import subprocess
from pathlib import Path

from gdf.utils.logging import log


class JetsonDeployer:
    def __init__(
        self,
        host: str,
        user: str = "nvidia",
        remote_dir: str = "/home/nvidia/gdf_deploy",
        key_path: str | None = None,
    ) -> None:
        self.host = host
        self.user = user
        self.remote_dir = remote_dir
        self.key_path = key_path

    def _ssh_args(self) -> list[str]:
        args = ["-o", "StrictHostKeyChecking=no"]
        if self.key_path:
            args.extend(["-i", self.key_path])
        return args

    def deploy(
        self,
        engine_path: Path,
        class_names: list[str],
        extra_files: list[Path] | None = None,
    ) -> None:
        ssh_args = self._ssh_args()
        remote = f"{self.user}@{self.host}:{self.remote_dir}"

        log.info(f"Deploying to Jetson: {self.host}:{self.remote_dir}")

        subprocess.run(
            ["ssh", *ssh_args, f"{self.user}@{self.host}", f"mkdir -p {self.remote_dir}"],
            check=True,
        )

        files_to_copy = [engine_path]
        if extra_files:
            files_to_copy.extend(extra_files)

        for f in files_to_copy:
            log.info(f"Copying {f.name} → {remote}/")
            subprocess.run(
                ["scp", *ssh_args, str(f), f"{remote}/"],
                check=True,
            )

        class_names_path = engine_path.parent / "class_names.txt"
        class_names_path.write_text("\n".join(class_names))
        subprocess.run(
            ["scp", *ssh_args, str(class_names_path), f"{remote}/"],
            check=True,
        )

        log.info(f"Deployment complete. Files at {self.remote_dir}/")

    def benchmark(
        self,
        remote_engine_path: str | None = None,
        iterations: int = 100,
    ) -> str:
        engine = remote_engine_path or f"{self.remote_dir}/model.engine"
        ssh_args = self._ssh_args()

        cmd = [
            "ssh", *ssh_args, f"{self.user}@{self.host}",
            f"trtexec --loadEngine={engine} --iterations={iterations} --avgRuns=10",
        ]

        log.info(f"Running benchmark on Jetson: {self.host}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            log.error(f"Benchmark failed: {result.stderr}")
            return result.stderr

        log.info("Benchmark complete")
        return result.stdout
