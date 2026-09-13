import glob
import os
import onnx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default=".", help="Directory containing the ONNX model and data files")
args = parser.parse_args()

# Usage: --dir is flexible and discovers the .onnx.data files in the directory and its subdirectories. It will then combine the .onnx and .onnx.data files in each same subdirectory into a single .onnx file.
# Example: python3 combine_onnx_and_onnxdata.py --dir /lustre/alice/users/csonnab/TPC/o2-tpc-pid/output/LHC24/ar/apass3/LHC24ar_apass3_TPCSignal_small_HadronicRate_DeltaPhi/20260907/training/networks
# This generates a new directory called combined_onnx in each subdirectory (here: network_mean, network_sigma and network_full) containing the final .onnx file with the graph and weights.

for d in glob.glob(f"{args.dir}/**/*.onnx.data", recursive=True):

    parent_dir = os.path.dirname(d)
    onnx_model_path = glob.glob(f"{parent_dir}/*.onnx")
    model = onnx.load(onnx_model_path[0])

    # Use a fresh directory for this export.
    out = os.path.join(parent_dir, "combined_onnx")
    os.system("rm -rf " + out)
    os.makedirs(out, exist_ok=True)

    onnx.save_model(
        model,
        os.path.join(out, "net_onnx_full.onnx"),
        save_as_external_data=False,
        all_tensors_to_one_file=True,
        location="net_onnx_full.onnx.data",
        size_threshold=0,
    )

    onnx.checker.check_model(os.path.join(out, "net_onnx_full.onnx"))