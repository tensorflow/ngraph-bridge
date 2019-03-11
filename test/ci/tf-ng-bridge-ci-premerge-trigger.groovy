// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This script acts as a trigger script for the main ngraph-unittest.groovy
// Jenkins job.  This script is part of a Jenkins multi-branch pipeline job
// which can trigger GitHub jobs more effectively than the GitHub Pull
// Request Builder (GHPRB) plugin, in our environment.

// The original ngraph-tf-unittest job required the following parameters.  We
// set these up below as global variables, so we do not need to rewrite the
// original script -- we only need to provide this new trigger hook.
//
// Parameters which ngraph-unittest uses:
String  PR_URL = CHANGE_URL
String  PR_COMMIT_AUTHOR = CHANGE_AUTHOR
//ORIG: String  JENKINS_BRANCH = "master"
String  JENKINS_BRANCH = "chrisl/ngraph-bridge"

// Constants
JENKINS_DIR = 'jenkins'

env.MB_PIPELINE_CHECKOUT = true

timestamps {
    node("trigger") {

        deleteDir()  // Clear the workspace before starting

        dir(JENKINS_DIR){
            // Clone the cje-algo directory which contains our Jenkins groovy script
            git(branch: JENKINS_BRANCH, changelog: false, poll: false,
                url: 'https://github.intel.com/AIPG/cje-algo')
        }

        // Call the main job script.
        //
        // NOTE: We keep the main job script in github.intel.com because it may
        //       contain references to techn which has not yet been released.
        //
        echo "Calling tf-ng-bridge-ci-premerge.groovy"
        def ciPreMerge = load("${JENKINS_DIR}/tf-ng-bridge-ci-premerge.groovy")
        ciPreMerge(PR_URL, PR_COMMIT_AUTHOR)
        echo "tf-ng-bridge-ci-premerge.groovy completed"

    }  // End:  node
}  // End:  timestamps

echo "Done"
