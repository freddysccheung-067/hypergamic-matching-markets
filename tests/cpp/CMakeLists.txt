# Test targets are added as their sources appear, so an early `cmake -B build`
# works before any test file exists in the tree.
include(CTest)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_agent.cpp)
    add_executable(test_agent test_agent.cpp)
    target_link_libraries(test_agent PRIVATE hypergamic_core Catch2::Catch2WithMain)
    add_test(NAME test_agent COMMAND test_agent)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_matching.cpp)
    add_executable(test_matching test_matching.cpp)
    target_link_libraries(test_matching PRIVATE hypergamic_core Catch2::Catch2WithMain)
    add_test(NAME test_matching COMMAND test_matching)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_population.cpp)
    add_executable(test_population test_population.cpp)
    target_link_libraries(test_population PRIVATE hypergamic_core Catch2::Catch2WithMain)
    add_test(NAME test_population COMMAND test_population)
endif()
