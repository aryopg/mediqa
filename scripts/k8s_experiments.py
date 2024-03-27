import argparse
import os
import sys

sys.path.append(os.getcwd())

import yaml
from kubejobs.jobs import KubernetesJob


def argument_parser():
    parser = argparse.ArgumentParser(description="MEDIQA experiments")
    parser.add_argument("--run_configs_filepath", type=str, required=True)
    parser.add_argument("--user_email", type=str, required=True)
    parser.add_argument("--git_branch", type=str, default="main")
    parser.add_argument("--namespace", type=str, default="eidf106ns")
    parser.add_argument(
        "--train_script", type=str, default="scripts/train_binary_classifier.py"
    )
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    configs = yaml.safe_load(open(args.run_configs_filepath, "r"))

    base_args = f"pip uninstall -y huggingface_hub && pip install huggingface_hub && pip install peft && git clone https://$GIT_TOKEN@github.com/aryopg/mediqa.git --branch {args.git_branch} && cd mediqa && huggingface-cli download aryopg/mediqa --repo-type dataset --local-dir data --token $HF_DOWNLOAD_TOKEN --quiet && "
    command = f"python {args.train_script}"

    secret_env_vars = configs["env_vars"]

    run_name = args.train_script.split("/")[1].replace(".py", "").replace("_", "-")
    print(f"Creating job for: {run_name}")
    job = KubernetesJob(
        name=run_name[:63],
        cpu_request="8",
        ram_request="64Gi",
        image=configs["image"],
        gpu_type="nvidia.com/gpu",
        gpu_limit=configs["gpu_limit"],
        gpu_product=configs["gpu_product"],
        backoff_limit=4,
        command=["/bin/bash", "-c", "--"],
        args=[base_args + command],
        secret_env_vars=secret_env_vars,
        user_email=args.user_email,
        namespace=args.namespace,
        kueue_queue_name=f"{args.namespace}-user-queue",
    )

    # Run the Job on the Kubernetes cluster
    job.run()


if __name__ == "__main__":
    main()
