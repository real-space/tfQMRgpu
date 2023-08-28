# This file is part of tfQMRgpu under MIT-License

# CMake file for RapidXML
add_library(RapidXML INTERFACE)
target_include_directories(RapidXML INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/rapidxml-1.13>
	$<INSTALL_INTERFACE:include/third_party/rapidxml>
	)

include(GNUInstallDirs)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/rapidxml-1.13/
	DESTINATION include/third_party/rapidxml)

# install library
install(TARGETS RapidXML
	EXPORT RapidXMLTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
# install CMake files so the library can be used by other cmake projects
install(EXPORT RapidXMLTargets
	FILE RapidXMLTargets.cmake
	NAMESPACE RapidXML::
	DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/RapidXML
	)
