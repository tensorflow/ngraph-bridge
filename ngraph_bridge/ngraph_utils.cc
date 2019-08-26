/*******************************************************************************
 * Copyright 2017-2019 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <dirent.h>

#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/events_writer.h"
#include "tensorflow/core/util/event.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/lib/io/record_writer.h"

#if defined NGRAPH_DISTRIBUTED
#include "ngraph/distributed.hpp"
#endif

#include "ngraph_bridge/ngraph_utils.h"
#include "ngraph_bridge/version.h"

using namespace std;
namespace ng = ngraph;

namespace tensorflow {

namespace ngraph_bridge {

Status IsNgraphTFLogTensorCopiesEnabled(int graph_id,
                                        bool& is_copy_log_enabled) {
  const char* copy_env_var = std::getenv("NGRAPH_TF_LOG_TENSOR_COPIES");
  if (copy_env_var == nullptr) {
    is_copy_log_enabled = false;
    return Status::OK();
  }
  int test_graph_id;
  try {
    test_graph_id = stoi(string(copy_env_var));
  } catch (const std::invalid_argument& ia) {
    return errors::InvalidArgument(
        "Invalid argument for NGRAPH_TF_LOG_TENSOR_COPIES");
  }
  // if -1 copies are logged for all graphs
  is_copy_log_enabled = (test_graph_id == -1 || test_graph_id == graph_id);
  return Status::OK();
}
void PrintTFTensor(Tensor& T1) {
  NGRAPH_VLOG(4) << "all tensor values" << (T1).SummarizeValue(64) << endl;
}
std::string DebugNode(Node* node) {
  std::string temp = node->name();
  temp += "[" + node->type_string() + "]";
  return temp;
}

std::string PrintBool(bool var) { return (var ? "Yes" : "No"); }

bool IsNGVariableType(string node_type) {
  if (ngraph_tf_are_variables_enabled())
    return (node_type == "NGraphVariable" || node_type == "NGraphAssign");
  else
    return node_type == "NGraphVariable";
}

bool IsNGSupportedType(string node_type) {
  return (IsNGVariableType(node_type) || node_type == "NGraphEncapsulate");
};

// Read from this ng_tensor into tf_tensor
void ReadNGTensor(shared_ptr<ng::runtime::Tensor> ng_tensor,
                  Tensor* tf_tensor) {
  ngraph::Event event_sync_ng_tf_tensors("Tensor Read D2H", "", "");
  void* tf_src_ptr = (void*)DMAHelper::base(tf_tensor);
  ng_tensor->read(tf_src_ptr, 0, ng_tensor->get_element_count() *
                                     ng_tensor->get_element_type().size());
  event_sync_ng_tf_tensors.Stop();
  ngraph::Event::write_trace(event_sync_ng_tf_tensors);
}

// Write into this ng_tensor from tf_tensor
void WriteNGTensor(shared_ptr<ng::runtime::Tensor> ng_tensor,
                   Tensor* tf_tensor) {
  ngraph::Event event_sync_ng_tf_tensors("Tensor Write H2D", "", "");
  void* tf_src_ptr = (void*)DMAHelper::base(tf_tensor);
  ng_tensor->write(tf_src_ptr, 0, ng_tensor->get_element_count() *
                                      ng_tensor->get_element_type().size());
  event_sync_ng_tf_tensors.Stop();
  ngraph::Event::write_trace(event_sync_ng_tf_tensors);
}

void SummarizeOp(OpKernelConstruction* ctx, std::ostream& out) {
  auto node_def = ctx->def();
  out << "Node name: " << node_def.name() << " Op: " << node_def.op() << "\n";
  out << "Inputs: " << node_def.input().size() << "\n    ";
  for (const std::string& input : node_def.input()) {
    out << input << "\n    ";
  }
  out << "\n";
}

//---------------------------------------------------------------------------
//  TensorToStream
//---------------------------------------------------------------------------
Status TensorToStream(std::ostream& ostream, const Tensor& tensor) {
  const char* data = tensor.tensor_data().data();
  int64 n_elements = tensor.NumElements();
  switch (tensor.dtype()) {
    case DT_HALF:
      TensorDataToStream<Eigen::half>(ostream, n_elements, data);
      break;
    case DT_FLOAT:
      TensorDataToStream<float>(ostream, n_elements, data);
      break;
    case DT_DOUBLE:
      TensorDataToStream<double>(ostream, n_elements, data);
      break;
    case DT_UINT32:
      TensorDataToStream<uint32>(ostream, n_elements, data);
      break;
    case DT_INT32:
      TensorDataToStream<int32>(ostream, n_elements, data);
      break;
    case DT_UINT8:
    case DT_QUINT8:
      TensorDataToStream<uint8>(ostream, n_elements, data);
      break;
    case DT_UINT16:
    case DT_QUINT16:
      TensorDataToStream<uint16>(ostream, n_elements, data);
      break;
    case DT_INT8:
    case DT_QINT8:
      TensorDataToStream<int8>(ostream, n_elements, data);
      break;
    case DT_INT16:
    case DT_QINT16:
      TensorDataToStream<int16>(ostream, n_elements, data);
      break;
    case DT_UINT64:
      TensorDataToStream<uint64>(ostream, n_elements, data);
      break;
    case DT_INT64:
      TensorDataToStream<int64>(ostream, n_elements, data);
      break;
    case DT_BOOL:
      TensorDataToStream<bool>(ostream, n_elements, data);
      break;
    default:
      return errors::Internal("TensorToStream got unsupported data type ",
                              DataType_Name(tensor.dtype()));
      break;
  }
  return Status::OK();
}

Status TFDataTypeToNGraphElementType(DataType tf_dt,
                                     ngraph::element::Type* ng_et) {
  switch (tf_dt) {
    case DataType::DT_FLOAT:
      *ng_et = ng::element::f32;
      break;
    case DataType::DT_DOUBLE:
      *ng_et = ng::element::f64;
      break;
    case DataType::DT_INT32:
      *ng_et = ng::element::i32;
      break;
    case DataType::DT_UINT8:
      *ng_et = ng::element::u8;
      break;
    case DataType::DT_UINT16:
      *ng_et = ng::element::u16;
      break;
    case DataType::DT_INT64:
      *ng_et = ng::element::i64;
      break;
    case DataType::DT_UINT32:
      *ng_et = ng::element::u32;
      break;
    case DataType::DT_UINT64:
      *ng_et = ng::element::u64;
      break;
    case DataType::DT_BOOL:
      *ng_et = ng::element::boolean;
      break;
    case DataType::DT_QINT8:
      *ng_et = ng::element::i8;
      break;
    case DataType::DT_QUINT8:
      *ng_et = ng::element::u8;
      break;
    case DataType::DT_QINT32:
      *ng_et = ng::element::i32;
      break;
    default:
      return errors::Unimplemented("Unsupported TensorFlow data type: ",
                                   DataType_Name(tf_dt));
  }
  return Status::OK();
}

Status TFTensorShapeToNGraphShape(const TensorShape& tf_shape,
                                  ngraph::Shape* ng_shape) {
  for (int i = 0; i < tf_shape.dims(); i++) {
    if (tf_shape.dim_size(i) < 0) {
      return errors::InvalidArgument(
          "TensorFlow shape has a negative dimension size");
    }
  }

  *ng_shape = ngraph::Shape(tf_shape.dims());
  for (int i = 0; i < tf_shape.dims(); i++) {
    (*ng_shape)[i] = tf_shape.dim_size(i);
  }

  return Status::OK();
}

void print_node_histogram(const std::unordered_map<string, int>& histogram,
                          bool sorted) {
  int histogram_size = histogram.size();
  if (histogram_size == 0) {
    std::cout << "None";
  } else {
    vector<std::pair<string, int>> vec(begin(histogram), end(histogram));
    if (sorted) {
      sort(begin(vec), end(vec),
           [](const pair<string, int>& a, const pair<string, int>& b) {
             // descending sort
             return a.second > b.second;
           });
    }

    for (auto node : vec) {
      bool endelem = node == vec.back();
      std::cout << " " << node.first << " -> " << node.second
                << (endelem ? " " : ",");
    }
  }
}

const gtl::ArraySlice<DataType>& NGraphDTypes() {
  static gtl::ArraySlice<DataType> result{
      DT_FLOAT,  DT_DOUBLE, DT_INT8,   DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
      DT_UINT16, DT_UINT32, DT_UINT64, DT_BOOL,  DT_QINT8, DT_QUINT8};
  return result;
}

const gtl::ArraySlice<DataType>& NGraphNumericDTypes() {
  static gtl::ArraySlice<DataType> result{
      DT_FLOAT, DT_DOUBLE, DT_INT8,   DT_INT16,  DT_INT32,
      DT_INT64, DT_UINT8,  DT_UINT16, DT_UINT32, DT_UINT64};
  return result;
}

const gtl::ArraySlice<DataType>& NGraphNumericAndQuantizedDTypes() {
  static gtl::ArraySlice<DataType> result{
      DT_FLOAT, DT_DOUBLE, DT_INT8,   DT_INT16,  DT_INT32, DT_INT64,
      DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_QINT8, DT_QUINT8};
  return result;
}

const gtl::ArraySlice<DataType>& NGraphIndexDTypes() {
  static gtl::ArraySlice<DataType> result{DT_INT32, DT_INT64};
  return result;
}

const gtl::ArraySlice<DataType>& NGraphSupportedQuantizedDTypes() {
  static gtl::ArraySlice<DataType> result{DT_QINT8, DT_QUINT8};
  return result;
}

const gtl::ArraySlice<DataType>& NGraphRealDTypes() {
  static gtl::ArraySlice<DataType> result{DT_FLOAT, DT_DOUBLE};
  return result;
}

const gtl::ArraySlice<DataType>& NGraphBiasDTypes() {
  static gtl::ArraySlice<DataType> result{DT_FLOAT, DT_QINT32};
  return result;
}

Status CheckAxisDimInRange(std::vector<int64> axes, size_t rank) {
  for (auto i : axes) {
    if (i < (int)-rank || i >= (int)rank) {
      return errors::InvalidArgument("Axis Dimension is out of range. Got ", i,
                                     ", should be in range [-", rank, ", ",
                                     rank, ")");
    }
  }
  return Status::OK();
}

void NgraphSerialize(const std::string& file_name,
                     const std::shared_ptr<ngraph::Function>& ng_function) {
  NGRAPH_VLOG(0) << "Serializing graph to: " << file_name << std::endl;
  std::string js = ngraph::serialize(ng_function, 4);
  std::ofstream f;
  f.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  try {
    f.open(file_name);
    f << js;
    f.close();
  } catch (std::ofstream::failure& e) {
    NGRAPH_VLOG(0) << "Exception opening/closing file " << file_name
                   << std::endl;
    NGRAPH_VLOG(0) << e.what() << std::endl;
  }
}

void MemoryProfile(long& vm_usage, long& resident_set) {
  vm_usage = 0;
  resident_set = 0;

  // Get the two fields we want
  long vsize;
  long rss;

  std::ifstream ifs("/proc/self/stat", std::ios_base::in);
  std::string mem_in;
  getline(ifs, mem_in);
  if (mem_in != "") {
    vector<string> mem_str = ng::split(mem_in, ' ');
    vsize = std::stol(mem_str[22]);
    rss = std::stol(mem_str[23]);

    long page_size_kb = sysconf(_SC_PAGE_SIZE) /
                        1024;  // in case x86-64 is configured to use 2MB pages
    vm_usage = vsize / 1024;   // unit kb
    resident_set = rss * page_size_kb;
  }
}

std::string DotFilename(std::string kind, int idx) {
  return GraphFilenamePrefix(kind, idx) + ".dot";
}

std::string DotFilename(std::string kind, int idx, int sub_idx) {
  return GraphFilenamePrefix(kind, idx, sub_idx) + ".dot";
}

std::string PbtxtFilename(std::string kind, int idx) {
  return GraphFilenamePrefix(kind, idx) + ".pbtxt";
}

std::string PbtxtFilename(std::string kind, int idx, int sub_idx) {
  return GraphFilenamePrefix(kind, idx, sub_idx) + ".pbtxt";
}

std::string GraphFilenamePrefix(std::string kind, int idx) {
  std::stringstream ss;
  ss << kind << "_" << std::setfill('0') << std::setw(4) << idx;
#if defined NGRAPH_DISTRIBUTED
  int rank_id = ng::get_distributed_interface()->get_rank();
  ss << "_" << std::setfill('0') << std::setw(4) << rank_id;
#endif
  return ss.str();
}

std::string GraphFilenamePrefix(std::string kind, int idx, int sub_idx) {
  std::stringstream ss;
  ss << GraphFilenamePrefix(kind, idx) << "_" << std::setfill('0')
     << std::setw(4) << sub_idx;
#if defined NGRAPH_DISTRIBUTED
  int rank_id = ng::get_distributed_interface()->get_rank();
  ss << "_" << std::setfill('0') << std::setw(4) << rank_id;
#endif
  return ss.str();
}

bool DumpAllGraphs() { return std::getenv("NGRAPH_TF_DUMP_GRAPHS") != nullptr; }

bool DumpPrecaptureGraphs() {
  return DumpAllGraphs() ||
         std::getenv("NGRAPH_TF_DUMP_PRE_CAPTURED_GRAPHS") != nullptr;
}

bool DumpCapturedGraphs() {
  return DumpAllGraphs() ||
         std::getenv("NGRAPH_TF_DUMP_CAPTURED_GRAPHS") != nullptr;
}

bool DumpUnmarkedGraphs() {
  return DumpAllGraphs() ||
         std::getenv("NGRAPH_TF_DUMP_UNMARKED_GRAPHS") != nullptr;
}

bool DumpMarkedGraphs() {
  return DumpAllGraphs() ||
         std::getenv("NGRAPH_TF_DUMP_MARKED_GRAPHS") != nullptr;
}

bool DumpClusteredGraphs() {
  return DumpAllGraphs() ||
         std::getenv("NGRAPH_TF_DUMP_CLUSTERED_GRAPHS") != nullptr;
}

bool DumpDeclusteredGraphs() {
  return DumpAllGraphs() ||
         std::getenv("NGRAPH_TF_DUMP_DECLUSTERED_GRAPHS") != nullptr;
}

bool DumpEncapsulatedGraphs() {
  return DumpAllGraphs() ||
         std::getenv("NGRAPH_TF_DUMP_ENCAPSULATED_GRAPHS") != nullptr;
}

bool DumpTrackedGraphs() {
  return DumpAllGraphs() ||
         std::getenv("NGRAPH_TF_DUMP_TRACKED_GRAPHS") != nullptr;
}

#if defined(NGRAPH_DISTRIBUTED)
void OpControlOrder(const std::shared_ptr<ngraph::Function>& ng_function,
                    const std::string& op_name) {
  // Get the serialized ops and stored the allreduce ops to a vector and
  ng::NodeVector op_list;
  for (const shared_ptr<ng::Node>& node : ng_function->get_ordered_ops()) {
    if (node->description() == op_name) {
      op_list.push_back(node);
    }
  }
  // Sort the allreduce ops according to the TF names
  std::sort(op_list.begin(), op_list.end(),
            [](const shared_ptr<ng::Node>& x, const shared_ptr<ng::Node>& y) {
              return x->get_friendly_name() < y->get_friendly_name();
            });
  // Add control dependency in for the allreduce ops
  if (op_list.size() > 1) {
    for (size_t i = 1; i < op_list.size(); ++i) {
      auto pre_node = op_list[i - 1];
      auto cur_node = op_list[i];
      cur_node->add_control_dependency(pre_node);
    }
  }
}
#endif

bool IsProcessedByNgraphPass(Graph* g) {
  // TODO: place a dummy node as a marker
  // Current method may fail when graph has no encapsulates after first pass
  // Also variable/optimizer change introduces other types of ng nodes
  for (Node* node : g->nodes()) {
    if (node->type_string() == "NGraphEncapsulate") return true;
  }
  return false;
}

Status DumpNGraph(int file_idx, tensorflow::GraphDef* graph_def, std::set<std::string> nodes)
{
  const char* path = std::getenv("NGRAPH_TF_TB_LOGDIR");

  if (path == nullptr)
  {
    return Status::OK();
  }

  std::string str_path(path);

  if (str_path.back() != '/')
  {
    str_path += "/";
  }

  struct stat buffer;
  if (stat(str_path.c_str(), &buffer) != 0) // path doesn't exist
  {
    mkdir(str_path.c_str(), 0777);
  }
  
  std::string dname = GetSessionName(file_idx, nodes);
  str_path += (dname.insert(0, "ngraph") + "/");

  if (stat(str_path.c_str(), &buffer) != 0) // path doesn't exist
  {
    mkdir(str_path.c_str(), 0777);
  }
  
  str_path += ("ngraph" + to_string(file_idx));

  CreateSummaryFromGraphDef(graph_def, str_path); // create tensorflow event

  return Status::OK();
}

Status UpdateComputeTime(int file_idx, std::string cluster, std::string sess_name, int step, int compute_time)
{
  const char* path = std::getenv("NGRAPH_TF_TB_LOGDIR");

  if (path == nullptr)
  {
    return Status::OK();
  }

  std::string str_path(path);
  if (str_path.back() != '/')
  {
    str_path += "/";
  }

  std::string dname(sess_name);
  str_path += (dname.insert(0, "stats") + "/");

  struct stat buffer;
  if (stat(str_path.c_str(), &buffer) != 0) // path doesn't exist
  {
    mkdir(str_path.c_str(), 0777);
  }

  // if grappler just began and if dir already exists, return error if tf event files exist inside dir
  if (step == 1)
  {
    TF_RETURN_IF_ERROR(VerifyEmptyTBDir(str_path));
  }

  // inspect directory's files
  DIR* dir = opendir(str_path.c_str());
  struct dirent* dp;
  vector<std::string> files;

  while ((dp = readdir(dir)) != nullptr)
  {
    if (dp->d_type == DT_REG) // look at all files in directory
    {
      files.push_back(std::string(dp->d_name));
    }
  }

  closedir(dir);
  // done inspecting directory's files

  // create event object
  tensorflow::Event event;
  tensorflow::Summary::Value* summary_value;
  tensorflow::Env* env = Env::Default();

  event.set_wall_time(env->NowMicros() / 1e6);
  event.set_step(step);
  summary_value = event.mutable_summary()->add_value();
  summary_value->set_tag(dname + " Compute Time (ms)/" + cluster);
  summary_value->set_simple_value((float) compute_time);
  // done creating event object

  if (files.empty()) // create new tf event file
  {
    str_path += ("stats" + to_string(file_idx));    

    tensorflow::EventsWriter writer(str_path);

    writer.WriteEvent(event);
  }
  else // append to tf event file in dir
  {
    std::string file_path = str_path + files[0];

    // open tf event file at location file_path
    std::unique_ptr<tensorflow::WritableFile> writable_file;

    env->NewAppendableFile(file_path, &writable_file);
    tensorflow::io::RecordWriter record_writer(writable_file.get());
    // done opening tf event file at location file_path
    

    record_writer.WriteRecord(event.SerializeAsString());
  }
  
  return Status::OK();
}

Status CreateSummaryFromGraph(tensorflow::Graph* graph, std::string filename_prefix)
{
  // convert Graph* to GraphDef*
  tensorflow::GraphDef* gdef = nullptr;
  graph->ToGraphDef(gdef);

  // create event summary writer
  tensorflow::EventsWriter writer(filename_prefix);
  tensorflow::Event event;

  // write graph to event summary
  event.set_graph_def(gdef->SerializeAsString());
  
  return Status::OK();
}

Status CreateSummaryFromGraphDef(tensorflow::GraphDef* graph_def, std::string filename_prefix)
{
  // create event summary writer
  tensorflow::EventsWriter writer(filename_prefix);
  tensorflow::Event event;

  // write graph to event summary
  event.set_graph_def(graph_def->SerializeAsString());
  writer.WriteEvent(event);

  return Status::OK();
}

Status AddSessionNameAttr(int file_idx, std::set<string> nodes, Graph* graph)
{ 
  for (auto node : graph->op_nodes())
  {
    if (node->type_string() == "NGraphEncapsulate")
    {
      node->AddAttr(("_session_name" + to_string(file_idx)), GetSessionName(file_idx, nodes));
    }
  }

  return Status::OK();
}

std::string GetSessionName(int file_idx, std::set<std::string> nodes)
{
  std::string name = "";

  if (nodes.size() == 0)
  {
    name = to_string(file_idx);
  }
  else
  {
    std::string entry;

    for (auto it = nodes.begin(); it != nodes.end(); ++it) // get last element in set
    {
      entry = *it;
    }

    int scope_idx = entry.find_first_of("/");

    if (scope_idx != std::string::npos)
    {
      name = to_string(file_idx) + "_" + entry.substr(0, scope_idx);
    }
    else
    {
      name = to_string(file_idx) + "_" + entry;
    }
  }

  return name;
}

Status VerifyEmptyTBDir(std::string path)
{
  // make sure directory is empty

  DIR* dir = opendir(path.c_str());
  struct dirent* dp;
  int num_obj = 0;

  while ((dp = readdir(dir)) != nullptr)
  {
    num_obj++;
  }

  if (num_obj)
  {
    std::string err = "Directory " + path + " contains " + to_string(num_obj) + " objects. Directory must be empty.";
    NGRAPH_VLOG(0) << err << endl;
    return errors::Internal(err);
  }

  return Status::OK();
}

}  // namespace ngraph_bridge

}  // namespace tensorflow
