# Model Level Testing for ngraph bridge

## Terminology/Structure
This section describes the directory structure. Indentation means sub-directories. Top to bottom, roughly, they are in the order in which they are executed
1. `test_main.py`: The CLI to trigger different tests
2. `models`: The directory containing all test suites
    1. `Test suite` or `Model test directory` (*Repeated*): A directory in `models`. Each `test suite` represents an external repo that must be downloaded. Usually that corresponds to a particular network topology (but certain repos might run multiple topologies)
        1. `README.md` (*Optional*): Information about this `test suite`. Expected to be short as this text is used by `test_main.py` to print a short help on available tests
        2. <a name="head1234"></a>`repo.txt`: Clonable git URL in the first line, Optionally branch/SHA/tag in the second line (by default master will be used)
        3. `getting_repo_ready.sh` (*Optional*): An executable shell script to run before the tests start running for this repo. Can be used to install prerequisites.
        4. `enable_ngraph.patch` (*Optional*): A patch to be applied to the downloaded repo (**2.1.2**). If this file exists it is used by all `sub-test` directories (**2.1.5**)
        5. `Sub-test` (*Repeated*):
            1. `core_rewrite_test.sh`
            2. `expected.json` (*Optional*):
            3. `enable_ngraph.patch` (*Optional*): If a `sub-test` contains its own patch, that takes precedence over the global patch (**2.1.4**)
            4. `custom_log_parser.py` (*Optional*):
            5. `README.md` (*Optional*):
        6. `cleanup.sh` (*Optional*): An executable shell script that will be
    2. `Non-repo based Test Suite` (*Repeated*):
        1. ~~`repo.txt`~~
        2. ~~`getting_repo_ready.sh`~~
        3. `README.md` (*Optional*):
        4. `Sub-test` (*Repeated*):
            1. `pbtxt/pb/savedmodel`
            2. `expected.json` (*Optional*):

[Custom foo description](#foo)


#foo


[link](#head1234)



---
## Sample uses

---

## Features
