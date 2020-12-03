The convention followed here for CI jobs is as follows:

```
[job type]-[plugin]-[additional qualifier].yml
```

The default job type runs the basic sanity tests and is left empty.

Some examples are shown below:

cpu (Basic sanity tests for CPU plugin)
models-cpu (Model tests)

vpu
models-vpu
igpu
models-igpu

cpu-grappler (Basic sanity tests for CPU plugin using grappler pass)
cpu-prebuilt (Basic sanity tests for CPU plugin using prebuilt TF)
cpu-intel-tf
