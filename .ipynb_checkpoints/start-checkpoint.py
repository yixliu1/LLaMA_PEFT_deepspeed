import os
import json
import socket
import argparse
import subprocess

if __name__ == "__main__":    
#     os.system("chmod +x ./s5cmd")
    os.environ["WANDB_MODE"] = "dryrun"
    subprocess.run(["git", "config", "--global", "--add", "safe.directory", "/opt/ml/code"])
    os.system("pip install deepspeed --upgrade")

    def upgrade_to_specific_version(package_name, version):
        subprocess.check_call(["pip", "install", f"{package_name}=={version}", "--upgrade"])

    # Example usage:
    upgrade_to_specific_version("transformers", "4.31.0")

    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--evaluation_strategy",type=str,default="epoch")
    parser.add_argument("--gradient_accumulation_steps",type=int,default=4)
    parser.add_argument("--eval_steps",type=int,default=500)
    parser.add_argument("--save_steps",type=int,default=500)
    parser.add_argument("--save_strategy",type=str,default="steps")
    parser.add_argument("--save_total_limit",type=int,default=4)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=1)
    # parser.add_argument("--fsdp", type=str, default="full_shard auto_wrap")
    parser.add_argument("--fsdp_transformer_layer_cls_to_wrap", type=str, default="LlamaDecoderLayer")
    parser.add_argument("--tf32", type=bool, default=False)
    parser.add_argument("--bf16",type=bool,default=True)
    parser.add_argument("--deepspeed",type=str)
    

    # Data, model, and output directories
    parser.add_argument("--output_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_name_or_path", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data_path", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
#     parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    

    args = parser.parse_args()

    os.environ['model_name_or_path'] = args.model_name_or_path
    os.environ['data_path'] = args.data_path
    os.environ['bf16'] = str(args.bf16)
    os.environ['output_dir'] = args.output_dir
    os.environ['num_train_epochs'] = str(args.num_train_epochs)
    os.environ['per_device_train_batch_size'] = str(args.per_device_train_batch_size)
    os.environ['per_device_eval_batch_size'] = str(args.per_device_eval_batch_size)
    os.environ['gradient_accumulation_steps'] = str(args.gradient_accumulation_steps)
    os.environ['evaluation_strategy'] = args.evaluation_strategy
    os.environ['save_strategy'] = args.save_strategy
    os.environ["eval_steps"] = str(args.eval_steps)
    os.environ["learning_rate"] = str(args.learning_rate)
    os.environ['save_steps'] = str(args.save_steps)
    os.environ['save_total_limit'] = str(args.save_total_limit)
    os.environ['weight_decay'] = str(args.weight_decay)
    os.environ['warmup_ratio'] = str(args.warmup_ratio)
    os.environ['lr_scheduler_type'] = args.lr_scheduler_type
    os.environ['logging_steps'] = str(args.logging_steps)
    # os.environ['fsdp'] = args.fsdp
    os.environ['fsdp_transformer_layer_cls_to_wrap'] = args.fsdp_transformer_layer_cls_to_wrap
    os.environ['tf32'] = str(args.tf32)
    os.environ["deepspeed"] = str(args.deepspeed)
    
    
    
    os.system("chmod +x ./start.sh")
    os.system("/bin/bash -c ./start.sh")
    
#     import os

#     path = "/opt/ml"

#     for dirpath, dirnames, filenames in os.walk(path):
#         for dirname in dirnames:
#             print(os.path.join(dirpath, dirname))
#         for filename in filenames:
#             print(os.path.join(dirpath, filename))