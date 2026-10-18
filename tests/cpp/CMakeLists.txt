# Each test_*.cpp becomes its own executable. CONFIGURE_DEPENDS makes CMake re-run
# when a test file is added, so creating test_agent.cpp mid-session registers it on
# the next build without a manual reconfigure.
include(CTest)

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test_*.cpp)

foreach(test_source ${TEST_SOURCES})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})
    target_link_libraries(${test_name} PRIVATE hypergamic_core Catch2::Catch2WithMain)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
