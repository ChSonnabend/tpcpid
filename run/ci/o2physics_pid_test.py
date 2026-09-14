import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def latest_full_onnx(repo_root):
    candidates = sorted(
        repo_root.glob("output/**/networks/network_full/net_onnx_full.onnx"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def require_executable(name):
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required O2Physics executable not found in PATH: {name}")
    return path


def write_config(path, aod_file, onnx_file):
    config = {
        "internal-dpl-aod-reader": {
            "aod-file-private": str(aod_file),
            "aod-max-io-rate": "0",
            "time-limit": "0",
            "orbit-offset-enumeration": "0",
            "orbit-multiplier-enumeration": "0",
            "start-value-enumeration": "0",
            "end-value-enumeration": "-1",
            "step-value-enumeration": "1",
        },
        "internal-dpl-aod-spawner": "",
        "internal-dpl-aod-index-builder": "",
        "internal-dpl-injected-dummy-sink": "",
        "timestamp": {
            "timestamp": {
                "verbose": "false",
                "fatalOnInvalidTimestamp": "false",
                "rct-path": "RCT/Info/RunInformation",
                "orbit-reset-path": "CTP/Calib/OrbitReset",
                "isRun2MC": "-1",
            }
        },
        "eventselection-run3": {
            "timestamp": {
                "verbose": "false",
                "fatalOnInvalidTimestamp": "false",
                "rct-path": "RCT/Info/RunInformation",
                "orbit-reset-path": "CTP/Calib/OrbitReset",
                "isRun2MC": "-1",
            },
            "bcselOpts": {
                "amIneeded": "-1",
                "triggerBcShift": "0",
                "ITSROFrameStartBorderMargin": "-1",
                "ITSROFrameEndBorderMargin": "-1",
                "TimeFrameStartBorderMargin": "-1",
                "TimeFrameEndBorderMargin": "-1",
                "checkRunDurationLimits": "false",
                "NumberOfOrbitsPerTF": "-1",
            },
            "evselOpts": {
                "amIneeded": "-1",
                "muonSelection": "0",
                "maxDiffZvtxFT0vsPV": "1",
                "isMC": "-1",
                "confSigmaBCforHighPtTracks": "4",
                "TimeIntervalForOccupancyCalculationMin": "-40",
                "TimeIntervalForOccupancyCalculationMax": "100",
                "TimeRangeVetoOnCollStrict": "10",
                "TimeRangeVetoOnCollNarrow": "0.25",
                "FT0CamplPerCollCutVetoOnCollInTimeRange": "8000",
                "FT0CamplPerCollCutVetoOnCollInROF": "5000",
                "EpsilonVzDiffVetoInROF": "0.3",
                "UseWeightsForOccupancyEstimator": "true",
                "NumberOfOrbitsPerTF": "-1",
                "VzDiffNsigma": "3",
                "VzDiffMargin": "0.2",
            },
            "lumiOpts": {"amIneeded": "-1"},
            "ccdburl": "http://alice-ccdb.cern.ch",
        },
        "pid-tpc-base": {
            "processIU": "true",
            "processStandard": "false",
            "processRun3": "false",
            "processDummy": "true",
        },
        "pid-tpc-service": {
            "ccdburl": "http://alice-ccdb.cern.ch",
            "pidTPC": {
                "param-file": "",
                "ccdbPath": "Analysis/PID/TPC/Response",
                "recoPass": "",
                "ccdb-timestamp": "0",
                "useNetworkCorrection": "true",
                "autofetchNetworks": "false",
                "skipTPCOnly": "0",
                "devicesRequiringTPCOnlyPID": {"values": ["photon-conversion-builder"]},
                "networkPathLocally": str(onnx_file),
                "networkPathCCDB": "Analysis/PID/TPC/ML",
                "enableNetworkOptimizations": "true",
                "networkSetNumThreads": "1",
                "savedEdxsCorrected": "0",
                "useCorrecteddEdx": "false",
                "pid-full-el": "0",
                "pid-full-mu": "0",
                "pid-full-pi": "1",
                "pid-full-ka": "0",
                "pid-full-pr": "0",
                "pid-full-de": "0",
                "pid-full-tr": "0",
                "pid-full-he": "0",
                "pid-full-al": "0",
                "pid-tiny-el": "0",
                "pid-tiny-mu": "0",
                "pid-tiny-pi": "0",
                "pid-tiny-ka": "0",
                "pid-tiny-pr": "0",
                "pid-tiny-de": "0",
                "pid-tiny-tr": "0",
                "pid-tiny-he": "0",
                "pid-tiny-al": "0",
                "enableTuneOnDataTable": "0",
                "useNetworkEl": "0",
                "useNetworkMu": "0",
                "useNetworkPi": "1",
                "useNetworkKa": "0",
                "useNetworkPr": "0",
                "useNetworkDe": "0",
                "useNetworkTr": "0",
                "useNetworkHe": "0",
                "useNetworkAl": "0",
                "networkBetaGammaCutoff": "0.45",
                "ccdb-path-grplhcif": "GLO/Config/GRPLHCIF",
            },
            "processTracksIU": "true",
            "processTracksIUWithTracksQA": "false",
            "processTracksMCIU": "false",
        },
    }

    with path.open("w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Run an O2Physics TPC PID service smoke test.")
    parser.add_argument("--onnx", default=None, help="Path to net_onnx_full.onnx. Defaults to newest CI output.")
    parser.add_argument("--aod", default="run/ci/data/mini_AO2D_full.root", help="Path to a full AO2D test file.")
    parser.add_argument("--required", action="store_true", help="Fail if O2Physics executables are unavailable.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    aod_file = (repo_root / args.aod).resolve()
    onnx_file = Path(args.onnx).resolve() if args.onnx else latest_full_onnx(repo_root)
    if onnx_file is None:
        raise RuntimeError("Could not find output/**/networks/network_full/net_onnx_full.onnx")
    if not onnx_file.is_file():
        raise RuntimeError(f"ONNX file does not exist: {onnx_file}")
    if not aod_file.is_file():
        raise RuntimeError(f"AO2D test file does not exist: {aod_file}")

    needed = [
        "o2-analysis-timestamp",
        "o2-analysis-event-selection-service",
        "o2-analysis-pid-tpc-base",
        "o2-analysis-pid-tpc-service",
    ]
    missing = [name for name in needed if shutil.which(name) is None]
    if missing:
        message = "Missing O2Physics executables: " + ", ".join(missing)
        if args.required or os.environ.get("O2PHYSICS_CI_REQUIRED") == "1":
            raise RuntimeError(message)
        print(f"Skipping O2Physics PID workflow smoke test: {message}")
        return 0

    workdir = repo_root / "run" / "ci" / "o2physics_pid_work"
    workdir.mkdir(parents=True, exist_ok=True)
    config = workdir / "o2physics_pid_config.json"
    log_file = workdir / "o2physics_pid.log"
    write_config(config, aod_file, onnx_file)

    option = f"-b --configuration json://{shlex.quote(str(config))} --aod-memory-rate-limit 2000000000 --shm-segment-size 4000000000 --resources-monitoring 0 --min-failure-level error"
    command = (
        f"o2-analysis-timestamp {option} --aod-file {shlex.quote(str(aod_file))} | "
        f"o2-analysis-event-selection-service {option} | "
        f"o2-analysis-pid-tpc-base {option} | "
        f"o2-analysis-pid-tpc-service {option}"
    )

    print("Running O2Physics PID workflow smoke test...")
    print(f"ONNX: {onnx_file}")
    print(f"AO2D: {aod_file}")
    print(f"Config: {config}")
    print(f"Log: {log_file}")

    with log_file.open("w") as log:
        proc = subprocess.run(["bash", "-lc", command], cwd=workdir, stdout=log, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        print(log_file.read_text(errors="replace")[-12000:])
        raise RuntimeError(f"O2Physics PID workflow failed with exit code {proc.returncode}")

    print("O2Physics PID workflow smoke test completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
