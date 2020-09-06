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
#from tensorflow.tf_unittest_runner import regex_walk

# Note: to see a list of tests, run:
# .../test/python$ python -m pytest --collect-only <optional-args-to-specify-tests>
# e.g. ROOT=/localdisk/WS1 PYTHONPATH=$ROOT:$ROOT/test/python:$ROOT/tools:$ROOT/examples:$ROOT/examples/mnist python -m pytest --collect-only test_elementwise_ops.py


@pytest.fixture(autouse=True)
def reset_graph():
    yield
    ops.reset_default_graph()


@pytest.fixture(scope='session', autouse=True)
def cleanup():
    yield


# PyTestHook: called at early stage of pytest setup
def pytest_configure(config):
    pytest.tests_to_skip = []
    print("pytest_load_initial_conftests args=", config.invocation_params.args,
          "dir=", config.invocation_params.dir)
    pytest.arg_collect_only = (
        '--collect-only' in config.invocation_params.args)

    # Get list of tests to run
    if ('NGRAPH_TF_TEST_MANIFEST' in os.environ):
        filename = os.path.abspath(os.environ['NGRAPH_TF_TEST_MANIFEST'])
        skipitems = []
        with open(filename) as skipfile:
            print("[ skip-filter = " + filename + " ]")
            for line in skipfile.readlines():
                line = line.split('#')[0].rstrip('\n').strip(' ')
                if line == '':
                    continue
                skipitems.append(line)
        pytest.tests_to_skip = list(dict.fromkeys(skipitems))  # remove dups


def print_skip_debug(*args):
    if False and pytest.arg_collect_only:
        print(' '.join(args))


def should_skip_test(item):
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
        print_skip_debug("will skip test by name:", item.name,
                         "(" + item.function.__qualname__ + ")")
    elif item.cls.__qualname__ + "." + item.name in tests_to_skip:
        skip = True
        print_skip_debug("will skip test by class.name:",
                         item.cls.__qualname__ + "." + item.name)
    elif item.module.__name__ + "." + item.cls.__qualname__ + "." + item.name in tests_to_skip:
        skip = True
        print_skip_debug(
            "will skip test by module.class.name:", item.module.__name__ + "." +
            item.cls.__qualname__ + "." + item.name)

    # for parametrized tests, if we specify filter to exclude a test (i.e. all params)
    elif item.function.__name__ in tests_to_skip:
        skip = True
        print_skip_debug("will skip test by name[...]:", item.function.__name__,
                         "(" + item.name + ")")
    elif item.cls.__qualname__ + "." + item.function.__name__ in tests_to_skip:
        skip = True
        print_skip_debug("will skip test by class.name[...]:",
                         item.cls.__qualname__ + "." + item.name)
    elif item.module.__name__ + "." + item.cls.__qualname__ + "." + item.function.__name__ in tests_to_skip:
        skip = True
        print_skip_debug(
            "will skip test by module.class.name[...]:", item.module.__name__ +
            "." + item.cls.__qualname__ + "." + item.name)

    elif item.cls.__qualname__ in tests_to_skip:
        skip = True
        print_skip_debug("will skip test by class:", item.cls.__qualname__)
    elif item.module.__name__ + "." + item.cls.__qualname__ in tests_to_skip:
        skip = True
        print_skip_debug("will skip test by module.class:",
                         item.module.__name__ + "." + item.cls.__qualname__)
    elif item.module.__name__ in tests_to_skip:
        skip = True
        print_skip_debug("will skip test by module:", item.module.__name__)

    # finally...
    return skip


pytest.skip_marker_tag = "skipped via manifest"


# item -> Function
def get_item_fullname(item):
    return item.module.__name__ + "." + item.cls.__qualname__ + "." + item.name


# param items is list of Function type
# returns set
def count_skipped_items(items, reason=pytest.skip_marker_tag):
    skip_items = set()
    for item in items:
        # item -> Function
        if is_marked_skip(item, reason):
            skip_items.add(get_item_fullname(item))
    return skip_items


# param item is of Function type
def is_marked_skip(item, reason=pytest.skip_marker_tag):
    has_skip_marker = False
    for skip_marker in item.iter_markers(name='skip'):
        # skip_marker -> Mark(name='skip', args=(), kwargs={'reason': 'skipped via manifest'})
        if reason is None:
            has_skip_marker = True
            break
        elif skip_marker.kwargs['reason'] == reason:
            has_skip_marker = True
            break
    return has_skip_marker


# PyTestHook: called after collection has been performed, but
# we may filter or re-order the items in-place
def pytest_collection_modifyitems(config, items):
    skip_marker = pytest.mark.skip(reason=pytest.skip_marker_tag)
    for item in items:
        if should_skip_test(item):
            item.add_marker(skip_marker)
    # summary
    print("\n\nTotal Available Tests:", len(items))
    print("Skipped via manifest:", len(count_skipped_items(items)))
    all_skipped_count = len(count_skipped_items(items, None))
    print("All skipped:", all_skipped_count)
    print("Active Tests:", len(items) - all_skipped_count, "\n")


# PyTestHook: called after collection has been performed & modified
def pytest_collection_finish(session):
    if pytest.arg_collect_only:
        active_items = set(
            get_item_fullname(item)
            for item in session.items
            if not is_marked_skip(item, None))
        print('======================================================')
        print('Active tests...\n', sorted(active_items))
        print('======================================================')
        print('Total active tests:', len(active_items))
