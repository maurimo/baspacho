#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# cmake -DINPUT=<file> -DOUTPUT=<file.cpp> -DSYMBOL=<name> -P EmbedFile.cmake
# Writes a C++ source defining `extern "C" const unsigned char <SYMBOL>_data[]` with the bytes of
# INPUT and `extern "C" const size_t <SYMBOL>_size`.
file(READ "${INPUT}" hex HEX)
string(LENGTH "${hex}" hexLength)
math(EXPR size "${hexLength} / 2")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
string(REGEX REPLACE "((0x[0-9a-f][0-9a-f],){32})" "\\1\n" bytes "${bytes}")
file(WRITE "${OUTPUT}"
  "// generated from ${INPUT} by EmbedFile.cmake, do not edit\n"
  "#include <cstddef>\n"
  "extern \"C\" const unsigned char ${SYMBOL}_data[] = {\n${bytes}\n};\n"
  "extern \"C\" const size_t ${SYMBOL}_size = ${size};\n")
