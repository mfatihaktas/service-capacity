print('begin')

# install.packages("keras", repos="http://cran.r-project.org", lib="~/R_libs/")

# library(devtools)
# load_all("~/R_libs/")
# library(keras, lib.loc="~/R_libs/")
# ? keras

Sys.getenv("PATH")
# Sys.which('virtualenv')

library(tensorflow, lib.loc="~/R_libs/")
reticulate::use_python('/opt/sw/packages/gcc-4.8/python/3.5.2/bin/python')
install_tensorflow(method="virtualenv")

# Sys.setenv(TENSORFLOW_PYTHON="/opt/sw/packages/gcc-4.8/python/3.5.2")
# use_python("/opt/sw/packages/gcc-4.8/python/3.5.2")
# use_virtualenv("/opt/sw/packages/gcc-4.8/python/3.5.2")


# install_keras(tensorflow="gpu")

print('end')
