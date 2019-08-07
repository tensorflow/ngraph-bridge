#include "tensorflow/core/common_runtime/optimization_registry.h"

#ifdef NGRAPH_BRIDGE_STATIC_LIB_ENABLE
namespace tensorflow {

namespace ngraph_bridge {

void register_ngraph_bridge();
void register_ngraph_ops();
#if defined(NGRAPH_TF_ENABLE_VARIABLES_AND_OPTIMIZERS)
void register_ngraph_enable_variable_ops();
#endif
}
}
#endif