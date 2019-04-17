from subprocess import check_output, call, Popen, PIPE
import numpy as np
import os

# This script will run resnet50 training validation with synthetic data and real data
# and compare the results with the desired reference run.
# If the reference files are not provided it runs on TF(w/o nGraph) and uses its output
# as reference
# Assumed this validation.py script is under a tensorflow/benchmarks/ repo
# with git head at commit ab01ecc.
# TODO:
#    1. num_bathces set to 100
#    2. Makes certain assumptions about the reference_file 's name and the batch size
#    3. Add Arguments to take in the backend, the reference log files, the number of iterations/batches,
#       the data type (real or synthetic)
#    4. Automate the cloning of benchmarks repo and running the script

validate_with_real_data_command_NG = 'NGRAPH_TF_BACKEND=GPU python tf_cnn_benchmarks.py ' \
    + '--num_inter_threads=2 --data_format=NCHW --model=resnet50 --batch_size=32 ' \
    + '--num_gpus=1 --data_dir /mnt/data/TF_ImageNet_latest/ --data_name=imagenet ' \
    + '--datasets_use_prefetch=False --print_training_accuracy=True ' \
    + '--num_learning_rate_warmup_epochs=0 --num_batches=100'
validate_with_real_data_command_TF = 'NGRAPH_TF_DISABLE=1 python tf_cnn_benchmarks.py ' \
    + '--num_inter_threads=2 --data_format=NHWC --model=resnet50 --batch_size=32 ' \
    + '--num_gpus=1 --data_dir /mnt/data/TF_ImageNet_latest/ --data_name=imagenet ' \
    + '--datasets_use_prefetch=False --print_training_accuracy=True ' \
    + '--num_learning_rate_warmup_epochs=0 --num_batches=100'
validate_with_synthetic_data_command_NG = 'NGRAPH_TF_BACKEND=GPU python tf_cnn_benchmarks.py ' \
    + '--num_inter_threads=2 --tf_random_seed=1234 --data_format=NCHW ' \
    + '--model=resnet50 --batch_size=32 --num_gpus=1 --data_name=imagenet ' \
    + '--datasets_use_prefetch=False --print_training_accuracy=True ' \
    + '--num_learning_rate_warmup_epochs=0 --num_batches=100'
validate_with_synthetic_data_command_TF = 'NGRAPH_TF_DISABLE=1 python tf_cnn_benchmarks.py ' \
    + '--num_inter_threads=2 --tf_random_seed=1234 --data_format=NHWC --model=resnet50 ' \
    + '--batch_size=32 --num_gpus=1 --data_name=imagenet --datasets_use_prefetch=False ' \
    + '--print_training_accuracy=True --num_learning_rate_warmup_epochs=0 --num_batches=100'


def command_executor(cmd, verbose=False, msg=None, stdout=None):
    if verbose or msg is not None:
        tag = 'Running COMMAND: ' if msg is None else msg
        print(tag + cmd)

    p = Popen(
        cmd,
        shell=True,
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        close_fds=True,
        bufsize=1)
    output = p.stdout.read()
    error_output = p.stderr.read()

    return output, error_output


def write_to_file(filename, content):
    with open(filename, "w") as text_file:
        text_file.write(content)


def parse_training_output(output):
    to_parse = False
    total_loss = []
    top1_acc = []
    top5_acc = []

    for line in output.strip().split("\n"):
        if line.split()[0] == 'Step':
            to_parse = True
            continue

        elif line.startswith('-----'):
            to_parse = False
            continue

        if to_parse:
            total_loss.append(line.split()[-3])
            top1_acc.append(line.split()[-2])
            top5_acc.append(line.split()[-1])

    return total_loss, top1_acc, top5_acc


def parse_reference_file(filename):
    to_parse = False
    total_loss = []
    top1_acc = []
    top5_acc = []

    with open(filename) as reference_result:
        for line in reference_result:
            if line.split()[0] == 'Step':
                to_parse = True
                continue

            elif line.startswith('-----'):
                to_parse = False
                continue

            if to_parse:
                total_loss.append(line.split()[-3])
                top1_acc.append(line.split()[-2])
                top5_acc.append(line.split()[-1])

    return total_loss, top1_acc, top5_acc


def check_validation_results(norm_dict, metric):
    test_pass = True
    for norm in norm_dict:
        if norm_dict[norm] > 0.1:
            print(metric + " " + norm +
                  " is greater than the threshold 0.1, validation failed")
            test_pass = False
    return test_pass


# Return L1, L2, inf norm of the input arrays
def calculate_norm_values(result1, result2):
    l1_norm = np.linalg.norm(
        (np.array(result1, dtype=np.float) - np.array(result2, dtype=np.float)),
        1)

    l2_norm = np.linalg.norm(
        (np.array(result1, dtype=np.float) - np.array(result2, dtype=np.float)),
        2)

    inf_norm = np.linalg.norm(
        (np.array(result1, dtype=np.float) - np.array(result2, dtype=np.float)),
        np.inf)
    return {"l1_norm": l1_norm, "l2_norm": l2_norm, "inf_norm": inf_norm}


def run_validation(data_format, reference_file_name, batch_size):
    # Apply the patch to make input data loader deterministic for real data validation
    # Assume the current directory already has the required patch
    if os.path.isfile('./datasets_make_deterministic.patch'):
        output, error_output = command_executor(
            'git apply --check --whitespace=nowarn ' +
            './datasets_make_deterministic.patch')
        if error_output:
            print(
                "Warning: datasets_make_determinitic.patch is already applied")
        else:
            command_executor('git apply --whitespace=nowarn ' +
                             './datasets_make_deterministic.patch')

    # Run the validation command on NGraph
    if (data_format == "real_data"):
        command_to_run = validate_with_real_data_command_NG + str(batch_size)
    elif (data_format == "synthetic_data"):
        command_to_run = validate_with_synthetic_data_command_NG + \
            str(batch_size)

    print("Running: ", command_to_run)
    output, error_output = command_executor(command_to_run)
    output_string = str(output, 'utf-8')

    if output:
        ngraph_outputs_total_loss, ngraph_outputs_top1_acc, ngraph_outputs_top5_acc = parse_training_output(
            output_string)

    elif error_output:
        print("Something went wrong executing the command ",
              validate_with_real_data_command_NG)
        print(str(error_output, 'utf-8'))
        exit(1)

    print("ngraph total loss ", ngraph_outputs_total_loss)
    print("ngraph top1 Accuracy ", ngraph_outputs_top1_acc)
    print("ngraph top5 Accuracy ", ngraph_outputs_top5_acc)

    write_to_file(
        "resnet50_validationResult_NG_" + data_format + "_BS" + str(batch_size)
        + ".txt", output_string)

    # Get TF output: Either from a reference file or from actual run command
    # check if already has some TF result file
    cwd = os.getcwd()
    reference_file_path = cwd + reference_file_name + \
        '_BS' + str(batch_size) + ".txt"
    print("Finding reference file ", reference_file_path)
    if os.path.isfile(reference_file_path):
        # parse the text file directly
        reference_outputs_total_loss, reference_outputs_top1_acc, reference_outputs_top5_acc = parse_reference_file(
            reference_file_path)
    else:
        # Run the validation command on TF
        # This requires the TF needs to build with GPU
        print("No reference output file found, begin running reference command")
        print("Running: ", validate_with_real_data_command_TF)
        output, error_output = command_executor(
            validate_with_real_data_command_TF)
        output_string = str(output, 'utf-8')

        if output:
            reference_outputs_total_loss, reference_outputs_top1_acc, reference_outputs_top5_acc = parse_training_output(
                output_string)
        elif error_output:
            print("Something went wrong executing the command ",
                  validate_with_real_data_command_NG)
            print(str(error_output, 'utf-8'))
            exit(1)

        write_to_file(
            "resnet50_validaionResultReference" + str(batch_size) + ".txt",
            output_string)

    print("reference total loss ", reference_outputs_total_loss)
    print("reference top1Acc ", reference_outputs_top1_acc)
    print("reference top5Acc ", reference_outputs_top5_acc)

    # Compare the TF output and NG output
    # TF CPU results and GPU results are not the same, so for TF results
    # Need to run with TF GPU
    assert len(ngraph_outputs_total_loss) == len(
        reference_outputs_total_loss), "Number of total_loss values mismatch"
    assert len(ngraph_outputs_top1_acc) == len(
        reference_outputs_top1_acc), "Number of top1_accuracy values mismatch"
    assert len(ngraph_outputs_top5_acc) == len(
        reference_outputs_top5_acc), "Number of top5_accuracy values mismatch"

    loss_norms = calculate_norm_values(ngraph_outputs_total_loss,
                                       reference_outputs_total_loss)
    top1Acc_norms = calculate_norm_values(ngraph_outputs_top1_acc,
                                          reference_outputs_top1_acc)
    top5Acc_norms = calculate_norm_values(ngraph_outputs_top5_acc,
                                          reference_outputs_top5_acc)

    print(
        "loss norms are %f %f %f " %
        (loss_norms["l1_norm"], loss_norms["l2_norm"], loss_norms["inf_norm"]))
    print("top1Acc norms are %f %f %f " %
          (top1Acc_norms["l1_norm"], top1Acc_norms["l2_norm"],
           top1Acc_norms["inf_norm"]))
    print("top5Acc norms are %f %f %f " %
          (top5Acc_norms["l1_norm"], top5Acc_norms["l2_norm"],
           top5Acc_norms["inf_norm"]))

    loss_result = check_validation_results(loss_norms, "total_loss")
    top1Acc_result = check_validation_results(loss_norms, "top1 Accuracy")
    top5Acc_result = check_validation_results(loss_norms, "top5 Accuracy")

    if ((loss_result and top1Acc_result and top5Acc_result)):
        print("Validation test pass")

    # reapply the patch
    output, error_output = command_executor(
        'git apply -R ' + './datasets_make_deterministic.patch')


# Validation with synthetic data

if __name__ == "__main__":
    reference_file_name_realData = ''
    reference_file_name_syntheticData = ''
    batch_size = 100
    run_validation("real_data", reference_file_name_realData, batch_size)
    batch_size = 100
    run_validation("synthetic_data", reference_file_name_syntheticData,
                   batch_size)
