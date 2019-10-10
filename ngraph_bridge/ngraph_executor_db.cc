
#include "ngraph_bridge/ngraph_executor_db.h"

namespace tensorflow {

namespace ngraph_bridge {

NGraphExecutorDB::~NGraphExecutorDB() {
  m_executable_pipelined_tensors_map.clear();
  m_ng_function_map.clear();
  m_ng_exec_map.clear();
}

bool NGraphExecutorDB::MaybeGetNgExecutable(
    std::string signature,
    std::shared_ptr<ngraph::runtime::Executable>& ng_exec) {
  lock_guard<mutex> lock(m_mutex);
  auto it = m_ng_exec_map.find(signature);
  if (it == m_ng_exec_map.end()) {
    return false;
  }
  ng_exec = it->second;
  UpdateLRU(signature);
  return true;
}
// make pair func and exece
void NGraphExecutorDB::AddItem(
    std::string signature,
    std::pair<std::shared_ptr<ngraph::runtime::Executable>,
              std::shared_ptr<ngraph::Function>>
        ng_exec_func,
    std::shared_ptr<ngraph::runtime::Executable>& evicted_ng_exec, int depth) {
  auto ng_exec = ng_exec_func.first;
  auto ng_function = ng_exec_func.second;
  if (!signature.empty() && ng_function && ng_exec) {
    lock_guard<mutex> lock(m_mutex);
    const char* cache_depth_specified =
        std::getenv("NGRAPH_TF_FUNCTION_CACHE_ITEM_DEPTH");
    if (cache_depth_specified != nullptr) {
      if (m_ng_exec_map.size() >= atoi(cache_depth_specified)) {
        RemoveItem(m_lru.back(), evicted_ng_exec);
      }
    }
    auto ng_exec_map_output_pair = m_ng_exec_map.emplace(signature, ng_exec);
    if (ng_exec_map_output_pair.second) {
      auto ng_func_map_output_pair =
          m_ng_function_map.emplace(ng_exec, ng_function);
      if (ng_func_map_output_pair.second) {
        m_lru.push_front(signature);
        auto it = m_executable_pipelined_tensors_map.find(
            ng_exec);  // line no. 443-444

        // IsPipelinedTensorsStoreAvailable
        if (it == m_executable_pipelined_tensors_map.end()) {
          size_t num_inputs = ng_exec->get_parameters().size();
          size_t num_outputs = ng_exec->get_results().size();
          PipelinedTensorMatrix pipelined_input_tensors(num_inputs);
          PipelinedTensorMatrix pipelined_output_tensors(num_outputs);
          for (size_t i = 0; i < num_inputs; i++) {
            pipelined_input_tensors[i] = ng_exec->create_input_tensor(i, depth);
          }
          for (size_t i = 0; i < num_outputs; i++) {
            pipelined_output_tensors[i] =
                ng_exec->create_output_tensor(i, depth);
          }
          // InsertExecPipelineTesornMap
          shared_ptr<PipelinedTensorsStore> pts(new PipelinedTensorsStore(
              pipelined_input_tensors, pipelined_output_tensors));
          m_executable_pipelined_tensors_map.emplace(ng_exec, pts);
        }
      }
    }
  }
}

bool NGraphExecutorDB::MaybeGetNgFunction(
    std::shared_ptr<ngraph::runtime::Executable> ng_exec,
    std::shared_ptr<ngraph::Function>& ng_function)  // line no. 363, 364
{
  lock_guard<mutex> lock(m_mutex);
  auto it = m_ng_function_map.find(ng_exec);
  if (it == m_ng_function_map.end()) return false;
  ng_function = it->second;
  return true;
}

Status NGraphExecutorDB::GetDeviceTensors(
    const std::shared_ptr<ngraph::runtime::Executable>& ng_exec,
    std::tuple<int, PipelinedTensorVector, PipelinedTensorVector>& io_tensors) {
  lock_guard<mutex> lock(m_mutex);
  PipelinedTensorsStore* pts(nullptr);
  try {
    const auto& item = m_executable_pipelined_tensors_map.at(ng_exec);
    pts = item.get();
  } catch (...) {
    return errors::Internal("Executable not found in the cache");
  }
  io_tensors = pts->get_tensors();
  if (std::get<0>(io_tensors) < 0) {
    return errors::Internal("No free tensor available");
  }
  return Status::OK();
}

void NGraphExecutorDB::RemoveItem(
    std::string signature,
    std::shared_ptr<ngraph::runtime::Executable>&
        evicted_ng_exec)  // line no. 261, 262, 263
{
  // lock_guard<mutex> lock(m_mutex1);
  evicted_ng_exec = m_ng_exec_map[signature];
  m_ng_exec_map.erase(signature);
  m_ng_function_map.erase(evicted_ng_exec);
  m_lru.pop_back();
}

void NGraphExecutorDB::UpdateLRU(std::string signature) {
  // lock_guard<mutex> lock(m_mutex);
  if (signature != m_lru.front()) {
    m_lru.remove(signature);
    m_lru.push_front(signature);
  }
}
}  // namespace ngraph_bridge
}  // namespace tensorflow