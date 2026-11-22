// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

# CMake generated Testfile for 
# Source directory: /home/neko/chess-ch
# Build directory: /home/neko/chess-ch/build-avx2
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(startpos_perft_depth_4 "/home/neko/chess-ch/build-avx2/asterion" "perft" "--depth" "4" "--verify")
set_tests_properties(startpos_perft_depth_4 PROPERTIES  _BACKTRACE_TRIPLES "/home/neko/chess-ch/CMakeLists.txt;12;add_test;/home/neko/chess-ch/CMakeLists.txt;0;")
