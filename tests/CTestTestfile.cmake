# CMake generated Testfile for 
# Source directory: C:/DATA/TTS/whisper.cpp/tests
# Build directory: C:/DATA/TTS/whisper.cpp/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(test-main-tiny "C:/DATA/TTS/whisper.cpp/bin/Debug/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-tiny.bin" "-l" "fr" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-tiny PROPERTIES  LABELS "tiny;gh" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;16;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(test-main-tiny "C:/DATA/TTS/whisper.cpp/bin/Release/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-tiny.bin" "-l" "fr" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-tiny PROPERTIES  LABELS "tiny;gh" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;16;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(test-main-tiny "C:/DATA/TTS/whisper.cpp/bin/MinSizeRel/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-tiny.bin" "-l" "fr" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-tiny PROPERTIES  LABELS "tiny;gh" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;16;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(test-main-tiny "C:/DATA/TTS/whisper.cpp/bin/RelWithDebInfo/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-tiny.bin" "-l" "fr" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-tiny PROPERTIES  LABELS "tiny;gh" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;16;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
else()
  add_test(test-main-tiny NOT_AVAILABLE)
endif()
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(test-main-tiny.en "C:/DATA/TTS/whisper.cpp/bin/Debug/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-tiny.en.bin" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-tiny.en PROPERTIES  LABELS "tiny;en;gh" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;23;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(test-main-tiny.en "C:/DATA/TTS/whisper.cpp/bin/Release/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-tiny.en.bin" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-tiny.en PROPERTIES  LABELS "tiny;en;gh" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;23;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(test-main-tiny.en "C:/DATA/TTS/whisper.cpp/bin/MinSizeRel/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-tiny.en.bin" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-tiny.en PROPERTIES  LABELS "tiny;en;gh" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;23;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(test-main-tiny.en "C:/DATA/TTS/whisper.cpp/bin/RelWithDebInfo/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-tiny.en.bin" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-tiny.en PROPERTIES  LABELS "tiny;en;gh" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;23;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
else()
  add_test(test-main-tiny.en NOT_AVAILABLE)
endif()
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(test-main-base "C:/DATA/TTS/whisper.cpp/bin/Debug/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-base.bin" "-l" "fr" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-base PROPERTIES  LABELS "base" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;30;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(test-main-base "C:/DATA/TTS/whisper.cpp/bin/Release/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-base.bin" "-l" "fr" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-base PROPERTIES  LABELS "base" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;30;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(test-main-base "C:/DATA/TTS/whisper.cpp/bin/MinSizeRel/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-base.bin" "-l" "fr" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-base PROPERTIES  LABELS "base" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;30;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(test-main-base "C:/DATA/TTS/whisper.cpp/bin/RelWithDebInfo/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-base.bin" "-l" "fr" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-base PROPERTIES  LABELS "base" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;30;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
else()
  add_test(test-main-base NOT_AVAILABLE)
endif()
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(test-main-base.en "C:/DATA/TTS/whisper.cpp/bin/Debug/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-base.en.bin" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-base.en PROPERTIES  LABELS "base;en" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;37;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(test-main-base.en "C:/DATA/TTS/whisper.cpp/bin/Release/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-base.en.bin" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-base.en PROPERTIES  LABELS "base;en" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;37;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(test-main-base.en "C:/DATA/TTS/whisper.cpp/bin/MinSizeRel/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-base.en.bin" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-base.en PROPERTIES  LABELS "base;en" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;37;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(test-main-base.en "C:/DATA/TTS/whisper.cpp/bin/RelWithDebInfo/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-base.en.bin" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-base.en PROPERTIES  LABELS "base;en" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;37;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
else()
  add_test(test-main-base.en NOT_AVAILABLE)
endif()
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(test-main-small "C:/DATA/TTS/whisper.cpp/bin/Debug/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-small.bin" "-l" "fr" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-small PROPERTIES  LABELS "small" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;44;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(test-main-small "C:/DATA/TTS/whisper.cpp/bin/Release/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-small.bin" "-l" "fr" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-small PROPERTIES  LABELS "small" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;44;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(test-main-small "C:/DATA/TTS/whisper.cpp/bin/MinSizeRel/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-small.bin" "-l" "fr" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-small PROPERTIES  LABELS "small" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;44;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(test-main-small "C:/DATA/TTS/whisper.cpp/bin/RelWithDebInfo/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-small.bin" "-l" "fr" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-small PROPERTIES  LABELS "small" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;44;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
else()
  add_test(test-main-small NOT_AVAILABLE)
endif()
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(test-main-small.en "C:/DATA/TTS/whisper.cpp/bin/Debug/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-small.en.bin" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-small.en PROPERTIES  LABELS "small;en" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;51;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(test-main-small.en "C:/DATA/TTS/whisper.cpp/bin/Release/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-small.en.bin" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-small.en PROPERTIES  LABELS "small;en" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;51;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(test-main-small.en "C:/DATA/TTS/whisper.cpp/bin/MinSizeRel/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-small.en.bin" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-small.en PROPERTIES  LABELS "small;en" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;51;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(test-main-small.en "C:/DATA/TTS/whisper.cpp/bin/RelWithDebInfo/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-small.en.bin" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-small.en PROPERTIES  LABELS "small;en" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;51;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
else()
  add_test(test-main-small.en NOT_AVAILABLE)
endif()
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(test-main-medium "C:/DATA/TTS/whisper.cpp/bin/Debug/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-medium.bin" "-l" "fr" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-medium PROPERTIES  LABELS "medium" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;58;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(test-main-medium "C:/DATA/TTS/whisper.cpp/bin/Release/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-medium.bin" "-l" "fr" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-medium PROPERTIES  LABELS "medium" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;58;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(test-main-medium "C:/DATA/TTS/whisper.cpp/bin/MinSizeRel/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-medium.bin" "-l" "fr" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-medium PROPERTIES  LABELS "medium" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;58;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(test-main-medium "C:/DATA/TTS/whisper.cpp/bin/RelWithDebInfo/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-medium.bin" "-l" "fr" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-medium PROPERTIES  LABELS "medium" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;58;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
else()
  add_test(test-main-medium NOT_AVAILABLE)
endif()
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(test-main-medium.en "C:/DATA/TTS/whisper.cpp/bin/Debug/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-medium.en.bin" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-medium.en PROPERTIES  LABELS "medium;en" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;65;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(test-main-medium.en "C:/DATA/TTS/whisper.cpp/bin/Release/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-medium.en.bin" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-medium.en PROPERTIES  LABELS "medium;en" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;65;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(test-main-medium.en "C:/DATA/TTS/whisper.cpp/bin/MinSizeRel/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-medium.en.bin" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-medium.en PROPERTIES  LABELS "medium;en" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;65;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(test-main-medium.en "C:/DATA/TTS/whisper.cpp/bin/RelWithDebInfo/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-medium.en.bin" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-medium.en PROPERTIES  LABELS "medium;en" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;65;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
else()
  add_test(test-main-medium.en NOT_AVAILABLE)
endif()
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(test-main-large "C:/DATA/TTS/whisper.cpp/bin/Debug/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-large.bin" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-large PROPERTIES  LABELS "large" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;72;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(test-main-large "C:/DATA/TTS/whisper.cpp/bin/Release/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-large.bin" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-large PROPERTIES  LABELS "large" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;72;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(test-main-large "C:/DATA/TTS/whisper.cpp/bin/MinSizeRel/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-large.bin" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-large PROPERTIES  LABELS "large" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;72;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(test-main-large "C:/DATA/TTS/whisper.cpp/bin/RelWithDebInfo/main.exe" "-m" "C:/DATA/TTS/whisper.cpp/models/for-tests-ggml-large.bin" "-f" "C:/DATA/TTS/whisper.cpp/samples/jfk.wav")
  set_tests_properties(test-main-large PROPERTIES  LABELS "large" _BACKTRACE_TRIPLES "C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;72;add_test;C:/DATA/TTS/whisper.cpp/tests/CMakeLists.txt;0;")
else()
  add_test(test-main-large NOT_AVAILABLE)
endif()
