# ==============================================================================
#  Copyright 2018-2020 Intel Corporation
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
# ==============================================================================
import os
import pytest
from tensorflow.python.framework import ops


@pytest.fixture(autouse=True)
def reset_graph():
    yield
    ops.reset_default_graph()


@pytest.fixture(scope='session', autouse=True)
def cleanup():
    yield


def pytest_configure(config):
    pytest.tests_to_skip = []
    print("pytest_load_initial_conftests args=", config.invocation_params.args,
          "dir=", config.invocation_params.dir)
    # Get list of tests to run
    if ('PYTEST_SKIPFILTERS' in os.environ):
        filterfile = os.path.abspath(os.environ['PYTEST_SKIPFILTERS'])
        file = open(filterfile, mode='r')
        if file is not None:
            file_str = file.read()
            file.close()
            tests_to_skip = eval(file_str)
            print("[ skip-filter = " + filterfile + " ]")
            pytest.tests_to_skip = tests_to_skip


# ==============================================================================
# ==============================================================================


def pytest_collection_modifyitems(config, items):
    skip_listed = pytest.mark.skip(reason="skipped as per conftest.py")
    print("")
    skip_counter = 0
    for item in items:
        skip = False
        #print("\n\nchecking item:", item.name, item.function.__name__, "\nDetails:", item , "\n", item.module.__name__, item.cls.__qualname__, item.function.__qualname__)
        #pprint.pprint(dir(item.cls))
        #print(item.function.__name__, item.function.__qualname__)
        #debug_object(item)
        #pprint.pprint(dir(item.function))
        #pprint.pprint(item.__dict__)
        #debug_object(item)

        #print("fspath", item.fspath, "parent", item.parent)
        tests_to_skip = pytest.tests_to_skip
        if item.name in tests_to_skip:
            skip = True
            print("will skip test by name:", item.name,
                  "(" + item.function.__qualname__ + ")")
        elif item.cls.__qualname__ + "." + item.name in tests_to_skip:
            skip = True
            print("will skip test by class.name:",
                  item.cls.__qualname__ + "." + item.name)
        elif item.module.__name__ + "." + item.cls.__qualname__ + "." + item.name in tests_to_skip:
            skip = True
            print(
                "will skip test by module.class.name:", item.module.__name__ +
                "." + item.cls.__qualname__ + "." + item.name)

        # for parametrized tests, if we specify filter to exclude a test (i.e. all params)
        elif item.function.__name__ in tests_to_skip:
            skip = True
            print("will skip test by name[...]:", item.function.__name__,
                  "(" + item.name + ")")
        elif item.cls.__qualname__ + "." + item.function.__name__ in tests_to_skip:
            skip = True
            print("will skip test by class.name[...]:",
                  item.cls.__qualname__ + "." + item.name)
        elif item.module.__name__ + "." + item.cls.__qualname__ + "." + item.function.__name__ in tests_to_skip:
            skip = True
            print(
                "will skip test by module.class.name[...]:",
                item.module.__name__ + "." + item.cls.__qualname__ + "." +
                item.name)

        elif item.cls.__qualname__ in tests_to_skip:
            skip = True
            print("will skip test by class:", item.cls.__qualname__)
        elif item.module.__name__ + "." + item.cls.__qualname__ in tests_to_skip:
            skip = True
            print("will skip test by module.class:",
                  item.module.__name__ + "." + item.cls.__qualname__)
        elif item.module.__name__ in tests_to_skip:
            skip = True
            print("will skip test by module:", item.module.__name__)

        if skip:
            item.add_marker(skip_listed)
            skip_counter += 1

    print("\nTotal skipped via filter:", skip_counter, "\n")
