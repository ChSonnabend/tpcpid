import glob
import os
import onnx
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default=".", help="Directory containing the ONNX model and data files")
args = parser.parse_args()

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