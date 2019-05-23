# Model Level Testing for ngraph bridge

## Terminology/Structure
This section describes the directory structure. Indentation means sub-directories. Top to bottom, roughly, they are in the order in which they are executed
1. `test_main.py`: The CLI to trigger different tests
2. `models`: The directory containing all test suites
    1. `Test suite` or `Model test directory` (*Repeated*): A directory in `models`. Each `test suite` represents an external repo that must be downloaded. Usually that corresponds to a particular network topology (but certain repos might run multiple topologies)
        1. `README.md` (*Optional*): Information about this `test suite`. Expected to be short as this text is used by `test_main.py` to print a short help on available tests
        2. `repo.txt`: Clonable git URL in the first line, Optionally branch/SHA/tag in the second line (by default master will be used)
        3. `getting_repo_ready.sh` (*Optional*): An executable shell script to run before the tests start running for this repo. Can be used to install prerequisites.
        4. `enable_ngraph.patch` (*Optional*): A patch to be applied to the downloaded repo (**2.1.2**). If this file exists it is used by all `sub-test` directories (**2.1.5**), unless overridden by their own patch file (**2.1.5.1**)
        5. `Sub-test` (*Repeated*): Should start with the string `test`. Can be disabled by adding the string `disabled` to its name.
            1. `enable_ngraph.patch` (*Optional*): If a `sub-test` contains its own patch, that takes precedence over the global patch (**2.1.4**)
            2. `core_rewrite_test.sh`: The main run script for this `sub-test`
            3. `expected.json` (*Optional*): **_TODO_**
            4. `custom_log_parser.py` (*Optional*): **_TODO_**
            5. `README.md` (*Optional*): Information about this `sub-test`. Expected to be short as this text is used by
        6. `cleanup.sh` (*Optional*): An executable shell script that will be used to clean up, potentially the effects of `getting_repo_ready.sh` (**2.1.3**)
    2. `Non-repo based Test Suite` (*Repeated*): `Test suites` can also be based on `pb`, `pbtxt` or `savedmodel` instead of being based on a repo
        1. ~~`repo.txt`~~: A non repo based `test suite` should not contain a `repo.txt`
        2. ~~`getting_repo_ready.sh`~~: It should not contain `getting_repo_ready.sh`
        3. ~~`enable_ngraph.patch`~~: Neither can it have a patch file
        4. `README.md` (*Optional*): It might have a `README`. Same description as 2.1.1
        5. `Sub-test` (*Repeated*):
            1. `pbtxt/pb/savedmodel`: One TF model file per `sub-test`
            2. `expected.json` (*Optional*): **_TODO_**

---

## Expected results format

---
## Sample uses

---

## Features
