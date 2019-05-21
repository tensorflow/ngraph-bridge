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

import pdb
from subprocess import check_output, call, Popen, PIPE
import json, os, argparse, sys
import sys
# expects tools to be present at this relative location. Need access to build_utils
sys.path.insert(0, os.path.abspath('../../tools'))
from build_utils import download_repo


#TODO: update ngraph_enable.patch to be compatible with grappler build

def parse_json(json_file_name):
    #TODO: check if the json is in agreed-upon format and has all relevant info
    with open(json_file_name) as f:
        return json.load(f)

def generate_functional_check_checkpoint(loc, chkpoint_save_patch, run_command):
    pass

def command_executor(cmd, verbose = False, msg=None):
    if verbose or msg is not None:
        tag = 'Running COMMAND: ' if msg is None else msg
        print(tag + cmd)

    ps = Popen(cmd, stdin=PIPE, shell=True)
    so, se = ps.communicate()

def rewrite_test(model_dir):
    #TODO: assert TF version. Some models may not run on TF1.12 etc
    model_dir = os.path.abspath(model_dir)

    repo_filename = model_dir + '/repo.txt'
    if os.path.isfile(repo_filename):
        repo_info = [line.strip() for line in open(repo_filename).readlines() if len(line.strip())>0]
        repo_name = repo_info[0]
        repo_version = repo_info[1] if len(repo_info)==2 else 'master'
        repo_dl_loc = model_dir + '/downloaded_model'
        #TODO: download only when needed?
        download_repo(repo_dl_loc, repo_name, repo_version)

        cwd = os.getcwd()
        os.chdir(repo_dl_loc)
        # getting the repo ready is common to both check_rrewrite_test and get_checkpoint
        if os.path.isfile(model_dir + '/getting_repo_ready.sh'):

            command_executor(model_dir + '/getting_repo_ready.sh', verbose=True)

        # To generate the patch use: git diff > enable_ngraph.patch
        if os.path.isfile(model_dir + '/enable_ngraph.patch'):  #/foo/bar/.git
            command_executor('git apply ' + model_dir + '/enable_ngraph.patch')

        #It is assumed that we need to be in the "model repo" for core_rewrite_test to run
        #core_rewrite_test is written assuming we are currently in the downloaded repo
        for flname in os.listdir(model_dir): # The model folder can have multiple tests
            if flname.startswith('core_rewrite_test') and 'disabled' not in flname:
                command_executor(model_dir + '/' + flname, msg="Running test config: " + flname)
        command_executor('git reset --hard') # remove applied patch (if any)
        # TODO: each core_rewrite_test.sh could have its own enable_ngraph.patch
        os.chdir(cwd)
    else:
        # TODO: found a pbtxt or pb or saved model. Load and run that
        pass

        #TODO: check if failed or passed
        # TODO: check if ran to completion
        # TODO: check if ran within a prefixed amount of time
        # TODO: check throughput/latency
        #TODO: check if atleast some stuff was placed on ngraph. Leverage LOG_PLACEMENT

def get_checkpoint():
    pass

def check_functional(model_dir):
    #check if there exists a check_functional.sh in the model folder
    #if not, then use run_functional
    pass

# TODO: what of same model but different configs?
# TODO: what if the same repo supports multiple models?

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing framework for TF models. Performs 2 types of tests. A) run_only and B) functional')

    parser.add_argument('--rewrite_test', action='store_true', help='perform type a tests (rewrite_test)')
    parser.add_argument('--functional', action='store_true', help='perform type b tests (functional)')
    parser.add_argument('--models', action='store', type=str, help='comma separated list of model names', default='')

    cwd = os.getcwd()
    # This script must be run from this location
    assert cwd.split('/')[-1] == 'model_level_tests'

    args = parser.parse_args()

    if not(args.rewrite_test or args.functional):
        print("No type of test enabled. Please choose --rewrite_test, --functional or both")
        sys.exit(0)

    model_list = os.listdir('models') if args.models == '' else args.models.split(',')

    for model_name in model_list:
        print('Testing model: ' + model_name)
        if args.rewrite_test:
            rewrite_test('./models/' + model_name)
        if args.functional:
            print('Functional tests not implemented yet!!')


# Sample run script:
# python test_main.py --rewrite_test --models MLP


