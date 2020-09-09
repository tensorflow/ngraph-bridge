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
import _pytest.skipping
from tensorflow.python.framework import ops

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


def pattern_to_regex(pattern):
    no_param = (re.search(r'\[.*\]$', pattern) is None)
    pattern_noparam = re.sub(r'\[.*$', '', pattern)
    pattern = re.sub(r'\-', '\\-', pattern)
    pattern = re.sub(r'\.', '\\.', pattern)
    pattern = re.sub(r'\[', '\\[', pattern)
    pattern = re.sub(r'\]', '\\]', pattern)
    # special case for M.C.F when it possibly macthes with parameterized tests
    if pattern_noparam.count('.') == 2 and no_param:
        pattern = '^' + pattern + '\[.*'
    if pattern_noparam.count('.') == 0:
        pattern = '^' + pattern + '\..*\..*' + '$'
    if pattern_noparam.count('.') == 1:
        pattern = '^' + pattern + '\..*' + '$'
    return pattern


def testfunc_matches_manifest(item, pattern):
    itemfullname = get_item_fullname(item)  # Module.Class.Func
    # print('checking for item:', itemfullname, 'with pattern:', pattern)
    if pattern == get_item_fullname:  # trivial case
        # print('  matched exact pattern')
        return True
    pattern = pattern_to_regex(pattern)
    # print('  pattern regex:', pattern)
    if re.search(pattern, itemfullname):
        # print('  matched pattern', pattern)
        return True
    # print('  no match')
    return False


# must be called after pytest.all_test_items has been set
def list_matching_tests(manifest_line):
    items = set()
    if (not pytest.all_test_items) or (len(pytest.all_test_items) == 0):
        return items
    for item in pytest.all_test_items:
        # item: Function type
        if testfunc_matches_manifest(item, manifest_line):
            items.add(get_item_fullname(item))
    # print('DBG: list_matching_tests for manifest_line:', manifest_line, '=>', items)
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
                new_runs = list_matching_tests(line)
                skipped_items -= new_runs
                run_items |= new_runs
            if curr_section == 'skip_section':
                new_skips = list_matching_tests(line)
                run_items -= new_skips
                skipped_items |= new_skips
        assert (run_items.isdisjoint(skipped_items))
        print('#Tests to Run={}, Skip={} (manifest = {})'.format(
            len(run_items), len(skipped_items), manifestfile))

    return run_items, skipped_items  # 2 sets


def testfunc_matches_set(item, tests_to_match):
    itemfullname = get_item_fullname(item)
    return (itemfullname in tests_to_match)


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


pytest.run_marker = pytest.mark.run


def is_marked_run(item):
    for run_marker in item.iter_markers(name='run'):
        return True


# param item is of Function type
def remove_skip_marker(item):
    for skip_marker in item.iter_markers(name='skip'):
        # skip_marker -> Mark(name='skip', args=(), kwargs={'reason': 'abc'})
        print('removing skip marker:', get_item_fullname(item))
        # print(' skip_marker:', skip_marker)
        item.add_marker(pytest.run_marker)

        #item.iter_markers(name='skip').remove(skip_marker)
        #item.iter_markers().remove(skip_marker)
        #item.owner_markers.remove(skip_marker)
        #item.get_closest_marker(name='skip') = None
        # somelist[:] = (x for x in somelist if determine(x))
        #?? skip_marker.name = 'remove_skip'


# param items is list of Function type
# returns set
def count_run_items(items):
    run_items = set()
    for item in items:
        # item -> Function
        if is_marked_run(item):
            run_items.add(get_item_fullname(item))
    return run_items


def attach_run_markers(items):
    for item in items:
        if is_marked_run(item) or (
                get_item_fullname(item) in pytest.tests_to_run):
            item.add_marker(pytest.mark.run_via_manifest)


#--------------------------------------------------------------------------
#--------------------------------------------------------------------------


# PyTestHook: ahead of command line option parsing
def pytest_cmdline_preparse(args):
    if ('NGRAPH_TF_TEST_MANIFEST' in os.environ):
        args[:] = ["-m", 'run_via_manifest'] + args


# def pytest_addoption(parser):
#     os.environ['PYTEST_ADDOPTS'] = '-m run_via_manifest'

# @pytest.fixture
# def marker(request):
#     return request.config.getoption("-m")

# def pytest_collection(session):
#     x


# PyTestHook: called at early stage of pytest setup
def pytest_configure(config):
    if ('NGRAPH_TF_TEST_MANIFEST' in os.environ):
        pytest.tests_to_skip = set()
        pytest.tests_to_run = set()
        # config.invocation_params.args['-m'] = 'run_via_manifest'
        print("\npytest args=", config.invocation_params.args, "dir=",
              config.invocation_params.dir)
        pytest.arg_collect_only = (
            '--collect-only' in config.invocation_params.args)

        # register an additional marker
        config.addinivalue_line(
            "markers",
            "run_via_manifest: mark test to run via manifest filters")
        config.addinivalue_line("addopts", "-m run_via_manifest")

        # # Force ignore all skips present in test-scripts
        # def no_skip(*args, **kwargs):
        #     return
        # _pytest.skipping.skip = no_skip


# PyTestHook: called after collection has been performed, but
# we may modify or re-order the items in-place
def pytest_collection_modifyitems(config, items):
    skip_marker = pytest.mark.skip(reason=pytest.skip_marker_reason_manifest)
    # Get list of tests to run/skip
    if ('NGRAPH_TF_TEST_MANIFEST' in os.environ):
        filename = os.path.abspath(os.environ['NGRAPH_TF_TEST_MANIFEST'])
        pytest.all_test_items = items
        print('\nChecking manifest...')
        pytest.tests_to_run, pytest.tests_to_skip = read_tests_from_manifest(
            filename)
        # print('pytest.tests_to_run:', pytest.tests_to_run)
        # print('pytest.tests_to_skip:', pytest.tests_to_skip)
        for item in items:
            if testfunc_matches_set(item, pytest.tests_to_skip):
                # print('  match skip:', get_item_fullname(item))
                item.add_marker(skip_marker)
            elif testfunc_matches_set(item, pytest.tests_to_run):
                # print('    match RUN:',  get_item_fullname(item))
                remove_skip_marker(item)
        attach_run_markers(items)
        # summary
        print("\n\nTotal Available Tests:", len(items))
        print("Enabled/Run via manifest:", len(pytest.tests_to_run))
        print("Skipped via manifest:", len(count_skipped_items(items)))
        # re-enable skip markers, if not explicitly requested in manifest

        # unskipped = count_run_items(items)
        # all_skipped_count = len(count_skipped_items(items, None)) - len(unskipped)
        # print("All skipped:", all_skipped_count)
        # print("Active Tests:", len(items) - all_skipped_count, "\n")


# PyTestHook: called after collection has been performed & modified
def pytest_collection_finish(session):
    if ('NGRAPH_TF_TEST_MANIFEST' in os.environ):
        if pytest.arg_collect_only:
            active_items = set(
                get_item_fullname(item)
                for item in session.items
                if not is_marked_skip(item, None))
            # ?? active_items |= pytest.tests_to_run
            print('======================================================')
            skipped_items = count_skipped_items(session.items, None)
            # ?? skipped_items -= pytest.tests_to_run
            print('Skipped tests... ({})\n'.format(len(skipped_items)),
                  sorted(skipped_items))
            print('======================================================')
            unskipped = count_run_items(session.items)
            print("Un-skipped Tests... ({})\n".format(len(unskipped)),
                  unskipped)
            print('======================================================')
            print('Active tests... ({})\n'.format(len(active_items)),
                  sorted(active_items))
