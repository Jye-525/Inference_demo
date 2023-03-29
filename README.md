# This is a c++ demo program for image classification inference query.

To compile the program. You need to download the tensorflow c api from tensorflow official website.
The url is https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.11.0.tar.gz

tar -xvf libtensorflow-gpu-linux-x86_64-2.11.0.tar.gz
mv libtensorflow-gpu-linux-x86_64-2.11.0 libtensorflow_gpu_c

Note: put `libtensorflow_gpu_c` under the project folder. Otherwise you need to change the CMakeLists.txt file.
