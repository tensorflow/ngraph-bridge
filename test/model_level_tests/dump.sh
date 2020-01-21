pwd
cd /ec/pdx/disks/aipg_lab_home_pool_01/ashahba/source_code/tensorflow/ngraph-bridge/test/model_level_tests/models/MLP/downloaded_model
chmod +x /ec/pdx/disks/aipg_lab_home_pool_01/ashahba/source_code/tensorflow/ngraph-bridge/test/model_level_tests/models/MLP/getting_repo_ready.sh
/ec/pdx/disks/aipg_lab_home_pool_01/ashahba/source_code/tensorflow/ngraph-bridge/test/model_level_tests/models/MLP/getting_repo_ready.sh
cd /nfs/pdx/home/ashahba/source_code/tensorflow/ngraph-bridge/test/model_level_tests
pwd
cd /ec/pdx/disks/aipg_lab_home_pool_01/ashahba/source_code/tensorflow/ngraph-bridge/test/model_level_tests/models/MLP
cd /ec/pdx/disks/aipg_lab_home_pool_01/ashahba/source_code/tensorflow/ngraph-bridge/test/model_level_tests/models/MLP/downloaded_model
git apply /ec/pdx/disks/aipg_lab_home_pool_01/ashahba/source_code/tensorflow/ngraph-bridge/test/model_level_tests/models/MLP/test1/enable_ngraph.patch
chmod +x /ec/pdx/disks/aipg_lab_home_pool_01/ashahba/source_code/tensorflow/ngraph-bridge/test/model_level_tests/models/MLP/test1/core_run.sh
# Running test config test1: 
NGRAPH_TF_LOG_PLACEMENT=1 /ec/pdx/disks/aipg_lab_home_pool_01/ashahba/source_code/tensorflow/ngraph-bridge/test/model_level_tests/models/MLP/test1/core_run.sh
git reset --hard
cd /nfs/pdx/home/ashahba/source_code/tensorflow/ngraph-bridge/test/model_level_tests

pwd
cd /ec/pdx/disks/aipg_lab_home_pool_01/ashahba/source_code/tensorflow/ngraph-bridge/test/model_level_tests/models/MLP
cd /ec/pdx/disks/aipg_lab_home_pool_01/ashahba/source_code/tensorflow/ngraph-bridge/test/model_level_tests/models/MLP/downloaded_model
git apply /ec/pdx/disks/aipg_lab_home_pool_01/ashahba/source_code/tensorflow/ngraph-bridge/test/model_level_tests/models/MLP/test2/enable_ngraph.patch
chmod +x /ec/pdx/disks/aipg_lab_home_pool_01/ashahba/source_code/tensorflow/ngraph-bridge/test/model_level_tests/models/MLP/test2/core_run.sh
# Running test config test2: 
NGRAPH_TF_LOG_PLACEMENT=1 /ec/pdx/disks/aipg_lab_home_pool_01/ashahba/source_code/tensorflow/ngraph-bridge/test/model_level_tests/models/MLP/test2/core_run.sh
git reset --hard
cd /nfs/pdx/home/ashahba/source_code/tensorflow/ngraph-bridge/test/model_level_tests

chmod +x /ec/pdx/disks/aipg_lab_home_pool_01/ashahba/source_code/tensorflow/ngraph-bridge/test/model_level_tests/models/MLP/cleanup.sh
/ec/pdx/disks/aipg_lab_home_pool_01/ashahba/source_code/tensorflow/ngraph-bridge/test/model_level_tests/models/MLP/cleanup.sh
# Exiting. Done with tests in MLPrm -rf /ec/pdx/disks/aipg_lab_home_pool_01/ashahba/source_code/tensorflow/ngraph-bridge/test/model_level_tests/models/MLP/downloaded_model
