String  PR_URL = CHANGE_URL
String  PR_COMMIT_AUTHOR = CHANGE_AUTHOR
String  PR_TARGET = CHANGE_TARGET
String  JENKINS_BRANCH = "master"
Integer TIMEOUTTIME = "7200"

// Constants
JENKINS_DIR = '.'

timestamps {
    node("trigger") {

        deleteDir()  // Clear the workspace before starting
        println("INFO: CI job completed")

    }  // End:  node
}  // End:  timestamps

echo "Done"
