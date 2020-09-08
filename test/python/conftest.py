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
import re
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


def list_tests(manifest_line):
    items = set()
    items.add(manifest_line)
    return items


def read_tests_from_manifest(manifestfile, g_imported_files=set()):
    """
    Reads a file that has include & exclude patterns,
    Returns a set of leaf-level single testcase, no duplicates
    """
    run_items = set()
    skipped_items = set()
    g_imported_files.add(manifestfile)
    with open(manifestfile) as fh:
        curr_section = ''
        for line in fh.readlines():
            line = line.split('#')[0].rstrip('\n').strip(' ')
            if line == '':
                continue
            if re.search(r'\[IMPORT\]', line):
                curr_section = 'import_section'
                continue
            if re.search(r'\[RUN\]', line):
                curr_section = 'run_section'
                continue
            if re.search(r'\[SKIP\]', line):
                curr_section = 'skip_section'
                continue
            if curr_section == 'import_section':
                if not os.path.isabs(line):
                    line = os.path.abspath(
                        os.path.dirname(manifestfile) + '/' + line)
                if line in g_imported_files:
                    sys.exit("ERROR: re-import of manifest " + line + " in " +
                             manifestfile)
                g_imported_files.add(line)
                new_runs, new_skips = read_tests_from_manifest(
                    line, g_imported_files)
                assert (new_runs.isdisjoint(new_skips))
                run_items |= new_runs
                skipped_items |= new_skips
                run_items -= skipped_items
                continue
            if curr_section == 'run_section':
                new_runs = list_tests(line)
                skipped_items -= new_runs
                run_items |= new_runs
            if curr_section == 'skip_section':
                new_skips = list_tests(line)
                run_items -= new_skips
                skipped_items |= new_skips
        assert (run_items.isdisjoint(skipped_items))
        print('#Tests to Run={}, Skip={} (manifest = {})\n'.format(
            len(run_items), len(skipped_items), manifestfile))

    return run_items, skipped_items  # 2 sets


# PyTestHook: called at early stage of pytest setup
def pytest_configure(config):
    pytest.tests_to_skip = set()
    pytest.tests_to_run = set()
    print("\npytest args=", config.invocation_params.args, "dir=",
          config.invocation_params.dir)
    pytest.arg_collect_only = (
        '--collect-only' in config.invocation_params.args)

    # Get list of tests to run/skip
    if ('NGRAPH_TF_TEST_MANIFEST' in os.environ):
        filename = os.path.abspath(os.environ['NGRAPH_TF_TEST_MANIFEST'])
        pytest.tests_to_run, pytest.tests_to_skip = read_tests_from_manifest(
            filename)


def pattern_to_regex(pattern):
    pattern_noparam = re.sub(r'\[.*$', '', pattern)
    pattern = re.sub(r'\-', '\\-', pattern)
    pattern = re.sub(r'\.', '\\.', pattern)
    pattern = re.sub(r'\[', '\\[', pattern)
    pattern = re.sub(r'\]', '\\]', pattern)
    if pattern_noparam.count('.') == 0:
        pattern = pattern + '\..*\..*'
    if pattern_noparam.count('.') == 1:
        pattern = pattern + '\..*'
    return pattern


def testfunc_matches_set(item, tests_to_match):
    itemfullname = get_item_fullname(item)
    # print('checking for item:', itemfullname)
    for pattern in tests_to_match:
        if pattern == get_item_fullname:
            # print('  matched exact pattern')
            return True
        pattern = '^' + pattern_to_regex(pattern) + '$'
        # print('  pattern regex:', pattern)
        #p = re.compile(pattern)
        if re.search(pattern, itemfullname):
            # print('  matched pattern', pattern)
            return True
    # print('  no match')
    return False


# constant
pytest.skip_marker_reason_manifest = "skipped via manifest"


# item -> Function
def get_item_fullname(item):
    return item.module.__name__ + "." + item.cls.__qualname__ + "." + item.name


# param items is list of Function type
# returns set
def count_skipped_items(items, reason=pytest.skip_marker_reason_manifest):
    skip_items = set()
    for item in items:
        # item -> Function
        if is_marked_skip(item, reason):
            skip_items.add(get_item_fullname(item))
    return skip_items


# param item is of Function type
def is_marked_skip(item, reason=pytest.skip_marker_reason_manifest):
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


# param item is of Function type
def remove_skip_marker(item):
    for skip_marker in item.iter_markers(name='skip'):
        # skip_marker -> Mark(name='skip', args=(), kwargs={'reason': 'skipped via manifest'})
        print('removing skip marker:', get_item_fullname(item))
        item.iter_markers.remove(skip_marker)


# PyTestHook: called after collection has been performed, but
# we may filter or re-order the items in-place
def pytest_collection_modifyitems(config, items):
    skip_marker = pytest.mark.skip(reason=pytest.skip_marker_reason_manifest)
    for item in items:
        if testfunc_matches_set(item, pytest.tests_to_skip):
            # print('  match skip:', get_item_fullname(item))
            item.add_marker(skip_marker)
        elif testfunc_matches_set(item, pytest.tests_to_run):
            # print('    match RUN:',  get_item_fullname(item))
            remove_skip_marker(item)
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
        skipped_items = count_skipped_items(session.items, None)
        print('Skipped tests...\n', sorted(skipped_items))
        print('\nTotal skipped tests:', len(skipped_items))
        print('======================================================')
        print('Active tests...\n', sorted(active_items))
        print('\nTotal active tests:', len(active_items))
