#==============================================================================
#  Copyright 2019 Intel Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# =============================================================================

import pdb, time
from subprocess import check_output, call, Popen, PIPE
import json, os, argparse, sys
import sys
# expects tools to be present at this relative location. Need access to build_utils
sys.path.insert(0, os.path.abspath('../../tools'))
from build_utils import download_repo
from tf2ngraph import get_gdef
from log_parser import parse_logs, compare_parsed_values
import atexit

# TODO: update ngraph_enable.patch to be compatible with grappler build


def get_expected_from_json(json_file_name, configuration):
    with open(json_file_name) as f:
        expected_vals = json.load(f)[configuration]
        possible_keys_1 = set(['logparse', 'time'])
        assert all([(k in possible_keys_1) for k in expected_vals])
        for k in expected_vals:
            assert k in possible_keys_1, "Got unexpected key in json: " + k + ". Expected: " + possible_keys_1
        possible_keys_2 = set(['num_nodes_in_graph', 'num_nodes_marked_for_clustering', 'num_ng_clusters'])
        for k in expected_vals.get('logparse', {}):
            current_keys = set(expected_vals['logparse'][k].keys())
            assert len(current_keys.difference(possible_keys_2)) == 0, "Got unexpected keys in json: " + current_keys + ". Expected: " + possible_keys_2
        return expected_vals


def generate_functional_check_checkpoint(loc, chkpoint_save_patch, run_command):
    pass


def command_executor(cmd, verbose=False, msg=None, stdout=None, stderr=None):
    command_executor.commands += ('' if (msg is None) else
                                  '# ' + msg.strip('\n') + '\n') + cmd + '\n'
    if verbose or msg is not None:
        tag = 'Running Command: ' if msg is None else msg
        print(tag + cmd)
    if 'cd ' == cmd[:3]:
        os.chdir(cmd.split(' ')[1])
    else:
        ps = Popen(cmd, stdin=PIPE, stdout=stdout, stderr=stderr, shell=True)
        so, se = ps.communicate()
        errcode = ps.returncode
        assert errcode == 0, "Error in running command: " + cmd
        return so, se, errcode


command_executor.commands = ''  # TODO: slightly ugly


def return_to_cwd(f):

    def _helper(*args, **kwargs):
        so, _, __ = command_executor('pwd', stdout=PIPE)
        cwd = so.decode("utf-8").strip('\n')
        retval = f(*args, **kwargs)
        command_executor('cd ' + cwd)
        return retval

    return _helper


@return_to_cwd
def apply_patch_and_test(test_folder):
    model_dir = os.path.abspath(test_folder + '/..')
    downloaded_repo = os.path.abspath(model_dir + '/downloaded_model')
    command_executor('cd ' + model_dir)
    # To generate the patch use: git diff > enable_ngraph.patch
    patch_in_test_folder = os.path.abspath(test_folder + '/enable_ngraph.patch')
    patch_in_model_folder = os.path.abspath(test_folder +
                                            '../enable_ngraph.patch')
    if os.path.isfile(patch_in_test_folder):
        patch_file = patch_in_test_folder
    elif os.path.isfile(patch_in_model_folder):
        patch_file = patch_in_test_folder
    else:
        patch_file = None
    assert patch_file is not None, "Did not fine any patch file"

    command_executor('cd ' + downloaded_repo)
    if patch_file is not None:
        command_executor('git apply ' + patch_file)

    # TODO: Add the NGRAPH_TF_LOG_PLACEMENT=1 flag, only when there is no user-specified parser in the sub-test folder
    so, se, errcode = command_executor(
        'NGRAPH_TF_LOG_PLACEMENT=1 ' + test_folder + '/core_rewrite_test.sh',
        msg="Running test config " + test_folder.split('/')[-1] + ': ',
        stdout=PIPE,
        stderr=PIPE)

    command_executor('git reset --hard')  # remove applied patch (if any)
    return so.decode("utf-8"), se.decode("utf-8")


@return_to_cwd
def ready_repo(model_dir, repo_dl_loc):
    command_executor('cd ' + repo_dl_loc)
    command_executor('git reset --hard')
    # getting the repo ready is common to both check_rewrite_test and get_checkpoint
    if os.path.isfile(model_dir + '/getting_repo_ready.sh'):
        command_executor(model_dir + '/getting_repo_ready.sh', verbose=True)

# TODO: this function needs a name change
# TODO: this function needs to accept "do-i-dump-pbtxt"? and if so, a cleanup needs to happen later.
# Also this function could return the list of pbtxts it generated (but does it need to? we can infer it)
# TODO: this function should also take the level/intensity of test to run
def rewrite_test(model_dir, configuration):
    # TODO: assert TF version. Some models may not run on TF1.12 etc
    model_dir = os.path.abspath(model_dir)

    # download/prepare repo if needed:
    repo_filename = model_dir + '/repo.txt'
    repo_based = False # Is this test dir repo based or pb/pbtxt/savedmodel based?
    if os.path.isfile(repo_filename):
        repo_based = True
        repo_info = [
            line.strip()
            for line in open(repo_filename).readlines()
            if len(line.strip()) > 0
        ]
        repo_name = repo_info[0]
        repo_version = repo_info[1] if len(repo_info) == 2 else 'master'
        repo_dl_loc = model_dir + '/downloaded_model'
        # TODO: download only when needed?
        download_repo(repo_dl_loc, repo_name, repo_version)
        ready_repo(model_dir, repo_dl_loc)

    # Iterate through each sub-test
    for flname in os.listdir(model_dir):
        sub_test_dir = model_dir + '/' + flname
        # if its  directory starting with test, and not containing "disabled" in its name
        if not os.path.isfile(sub_test_dir) and flname.startswith('test') and 'disabled' not in flname:
            if repo_based:
                # TODO: shift the timing inside apply_patch_and_test
                sub_test_dir = model_dir + '/' + flname
                tstart = time.time()
                so, se = apply_patch_and_test(sub_test_dir)
                tend = time.time()
                command_executor.commands += '\n'
                parsed_vals = parse_logs(so)
                expected = get_expected_from_json(sub_test_dir + '/expected.json', configuration)
                passed, fail_help_string = compare_parsed_values(parsed_vals, expected.get('logparse', {}))
                # TODO: call compare_parsed_values. Move parse and compare logs, outside this if repo_based
            else:
                model = [i for i in os.listdir(sub_test_dir) if '.md' not in i and '.json' not in i]
                assert len(model) == 1
                model = model[0]
                split_on_dot = model.split('.')
                assert len(split_on_dot) <= 2
                if len(split_on_dot) == 1:
                    model_format = 'savedmodel'
                elif split_on_dot[1] in ['pb', 'pbtxt']:
                    model_format = split_on_dot[1]
                else:
                    assert False, "Unknown input format. Expected savedmodel, pb or pbtxt"
                # TODO: support checkpoint too later
                gdef = get_gdef(model_format, sub_test_dir + '/' + model)
                # TODO: run Level1 tests on gdef


    # Clean up if needed
    cleanup_script = model_dir + '/cleanup.sh'
    if os.path.isfile(cleanup_script):
        assert repo_based, 'Did not expect a cleanup script in non-repo based test'
        command_executor(cleanup_script)
    command_executor.commands += '# Exiting. Done with tests in ' + model_dir.split('/')[-1]
    # TODO: delete downloaded model repo

    # TODO: use gdef to run
    # TODO: add axpy test folders for pb. pbtxt and savedmodel
    # TODO integrate the if-else paths as much as possible

    # TODO: check if failed or passed
    # TODO: check if ran to completion
    # TODO: check if ran within a prefixed amount of time
    # TODO: check throughput/latency
    # TODO: check if atleast some stuff was placed on ngraph. Leverage LOG_PLACEMENT


def get_checkpoint():
    pass


def dump_commands_in_shellscript(dir):
    with open(dir + '/dump.sh', 'w') as f:
        f.write(command_executor.commands)


def check_functional(model_dir):
    #check if there exists a check_functional.sh in the model folder
    #if not, then use run_functional
    pass


def get_test_list_string(string):
    available_dirs = os.listdir('./models')
    dirs_to_scan = available_dirs if string == '' else string.split(',')
    help_string = ''
    for dir in dirs_to_scan:
        assert dir in available_dirs, "Requested to list " + dir + ", but that directory is not present in available directories: " + ','.join(
            available_dirs)
        help_string += 'Test directory: ' + dir + '\n' + '*' * 50 + '\n'
        currdir = './models/' + dir
        if os.path.isfile(currdir + '/README.md'):
            with open(currdir + '/README.md') as f:
                help_string += '\n'.join(f.readlines()) + '\n'
        for c in os.listdir(currdir):
            if os.path.isdir(c):
                help_string += 'Test: ' + c + '\n'
                currtest_readme = currdir + '/' + c + '/README.md'
                if os.path.isfile(currtest_readme):
                    with open(currtest_readme) as f:
                        help_string += '\n'.join(f.readlines()) + '\n'
        help_string += '\n'
    return help_string


# TODO: what of same model but different configs?
# TODO: what if the same repo supports multiple models?

if __name__ == '__main__':
    cwd = os.getcwd()
    atexit.register(dump_commands_in_shellscript, cwd)
    parser = argparse.ArgumentParser(
        description=
        'Testing framework for TF models. Performs 2 types of tests. A) run_only and B) functional'
    )

    parser.add_argument(
        '--rewrite_test',
        action='store_true',
        help='perform type a tests (rewrite_test)')
    parser.add_argument(  # TODO: revisit this flag
        '--functional',
        action='store_true',
        help='perform type b tests (functional)')
    parser.add_argument(
        '--models',
        action='store',
        type=str,
        help='comma separated list of model names',
        default='')
    parser.add_argument(
        '--list',
        action='store',
        type=str,
        help=
        'List all tests if empty string is passed, else list tests of the directories in the comma separated string that was passed',
        default=None)
    # TODO: add some pre-set configuration types. We already have "default", add "grappler", "var-opt", etc
    parser.add_argument(
        '--configuration',
        action='store',
        type=str,
        help=
        "The configuration in which the test is run (to choose which expected values current run's results will be compared against)",
        default='default')

    # This script must be run from this location
    assert cwd.split('/')[-1] == 'model_level_tests'

    args = parser.parse_args()

    if args.list is not None:
        print(get_test_list_string(args.list))
        exit(0)

    assert (args.rewrite_test or args.functional), 'No type of test enabled. Please choose --rewrite_test, --functional or both'

    model_list = os.listdir(
        'models') if args.models == '' else args.models.split(',')
    assert len(model_list) != 0, "Number of tests expected to be > 0"
    assert len(set(model_list).difference(set(
        os.listdir('./models')))) == 0, "The requested tests are not present"

    for model_name in model_list:
        print('Testing model: ' + model_name)
        if args.rewrite_test:
            rewrite_test('./models/' + model_name, args.configuration)
        if args.functional:
            print('Functional tests not implemented yet!!')

# TODO verbose or quiet?
# TODO: add a way to disable tests and subtests through the CLI

# TODO: what happens in case of shrestha's change. maybe expected number of clusters etc is different for normal path and var-opt path. Can be taken care of by --configuration. However user will have to decide if grappler, then use this config. it could possibly be auto-detected

# TODO: we have a way to control which model/test-dirs run (using --models). But we do not have a flag for test "intensity".
# each subtest folder has a "enable" patch and a run command.
# Level1: These are run with "parse the NGRAPH_TF_LOG_PLACEMENT=1". These tests should be short
# Level2: Dump pbtxts and run verify models (needs an input file that specifies certain layers. (what about all layers?)). These tests should be short
# When Level1 is running, dump out pbtxts that can be used for Level2 tests
# Level3: parse prints we put. These tests are run without "NGRAPH_TF_LOG_PLACEMENT=1". the framework can provide some default parsers, but users are free to add pyscripts that provide functions for custom script parsers
# These tests can be long
# So we can offer options to do: {1}, {1,2}, {1,2,3}, {3}  (or do we allow options for any combination of tests?)
# NOTE: Level3 and Level1 test are same (mechanics wise). Merge them. Then we have only 2 types of tests

# Each model dir represents 1 repo to download. A model dir can have multiple sub tests (each sub-test could represent a different model, or the same model tested under different settings)

# Structure of "expected json"
# dictionary of expected values. key is a config, value is the expected values json. there is a "default" config, but one can add other configs (for example for other backends etc)


# TODO: update main README.md. Document "how-to-use" and features

# Sample run script:
# python test_main.py --rewrite_test --models MLP

# feature 1: dumps shell script at the end. dumps shell script even when the framework crashes
# feature 2: prints list of tests and their descriptions (--list)
# feature 3: "expected" values can be varied by different configs
# feature 4: cleanup script
# feature 5: sub tests folders must start with 'test' (else ignored). Can have 'disabled' in their names to disable
# feature 6: default and user-specified log parsers
# feature 7: filename is supposed to be expected.json
