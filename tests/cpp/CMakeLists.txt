add_executable(test_agent test_agent.cpp)
target_link_libraries(test_agent PRIVATE hypergamic_core Catch2::Catch2WithMain)

add_executable(test_matching test_matching.cpp)
target_link_libraries(test_matching PRIVATE hypergamic_core Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(test_agent)
catch_discover_tests(test_matching)
