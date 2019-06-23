# TensorFlow C++ application example using nGraph

This directory contains an example C++ application that uses TensorFlow and nGraph. The example creates a simple computation graph and executes using nGraph computation backend.

The application is linked with TensorFlow C++ library and nGraph-TensorFlow bridge library. 

## prerequisites

The example application requires nGraph-TensorFlow bridge to be built first. Build the ngraph-tf by executing `build_ngtf.py` as per the [Option 3] instructions in the main readme. All the files needed to build this example application are located in the ngraph-tf/build directory.

### Dependencies

The following include files are needed to build the C++ application:

1. nGraph core header files
2. nGraph-TensorFlow bridge header files
3. TensorFlow header files

The application links with the following dynamic shared object (DSO) libraries

1. libngraph_bridge.(so/dylib)
2. libtensorflow_framework.so.1
3. libtensorflow_cc.so.1

## Build the example

Run the `make` command to build the application that will produce the executable: `infer_multiple_network`.

## Json file configuration 
Check template.json file to configure how you run the multi-thread multi-session profile. 
1. models : define all the models that you want to run (each model will correponding to one tensorflow session) 
2. profiles: each profile will open one thread, and will run the models in order("model_order") in a loop("loop")
ex>  {"name":"engine0", "model_order":[0,1,2], "loop":10} : Run model0, model1, model2 in order for 10 loop.  
     {"name":"engine0", "model_order":[0], "loop":10000} : Run model0 for 10000 loop.
3. models.label_list : This can be provided in two types
 - image file(png,jpeg,bmp) : this will infer, but will not check the result 
 - image_list_file(txt) : This will infer and also check the label index.  
     ex> image_list.txt - 5,dog,data/image_00016.png

### Run

Before running the application, set the `LD_LIBRARY_PATH` (or `DYLD_LIBRARY_PATH` for MacOS) to point to the library directories referred to in the `Makefile`:

    export LD_LIBRARY_PATH=$NGRAPH_BRIDGE_DIR/build_cmake/artifacts/lib:$NGRAPH_BRIDGE_DIR/build_cmake/artifacts/tensorflow

Where `NGRAPH_BRIDGE_DIR` should point to the directory where ngraph-tf was cloned.

:warning: Note: If this example is built on CentOS then the library directory 
is `lib64` - so please set the `LD_LIBRARY_PATH` accordingly

Next run the executable `./infer_multiple_network --json_file="path/to/the/json_file.json"`

[Option 3]: ../../README.md#option-3-build-ngraph-from-source
