# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/raphael/PhD_local/cpp/colmap

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/raphael/PhD_local/cpp/colmap/build

# Utility rule file for sqlite3_autogen.

# Include the progress variables for this target.
include lib/SQLite/CMakeFiles/sqlite3_autogen.dir/progress.make

lib/SQLite/CMakeFiles/sqlite3_autogen:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/raphael/PhD_local/cpp/colmap/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Automatic RCC for target sqlite3"
	cd /home/raphael/PhD_local/cpp/colmap/build/lib/SQLite && /usr/bin/cmake -E cmake_autogen /home/raphael/PhD_local/cpp/colmap/build/lib/SQLite/CMakeFiles/sqlite3_autogen.dir Release

sqlite3_autogen: lib/SQLite/CMakeFiles/sqlite3_autogen
sqlite3_autogen: lib/SQLite/CMakeFiles/sqlite3_autogen.dir/build.make

.PHONY : sqlite3_autogen

# Rule to build all files generated by this target.
lib/SQLite/CMakeFiles/sqlite3_autogen.dir/build: sqlite3_autogen

.PHONY : lib/SQLite/CMakeFiles/sqlite3_autogen.dir/build

lib/SQLite/CMakeFiles/sqlite3_autogen.dir/clean:
	cd /home/raphael/PhD_local/cpp/colmap/build/lib/SQLite && $(CMAKE_COMMAND) -P CMakeFiles/sqlite3_autogen.dir/cmake_clean.cmake
.PHONY : lib/SQLite/CMakeFiles/sqlite3_autogen.dir/clean

lib/SQLite/CMakeFiles/sqlite3_autogen.dir/depend:
	cd /home/raphael/PhD_local/cpp/colmap/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/raphael/PhD_local/cpp/colmap /home/raphael/PhD_local/cpp/colmap/lib/SQLite /home/raphael/PhD_local/cpp/colmap/build /home/raphael/PhD_local/cpp/colmap/build/lib/SQLite /home/raphael/PhD_local/cpp/colmap/build/lib/SQLite/CMakeFiles/sqlite3_autogen.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/SQLite/CMakeFiles/sqlite3_autogen.dir/depend

