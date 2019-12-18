/*
  Copyright 2019 Equinor ASA

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CUDA_HEADER_HEADER_INCLUDED
#define CUDA_HEADER_HEADER_INCLUDED

#include <iostream>

#define cudaCheckLastError(msg)    __cudaCheckError( __FILE__, __LINE__, #msg )

inline void __cudaCheckError(const char *file, const int line, const char *msg){
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err){
		std::cerr << "cudaCheckError() failed at " << file << ":" << line << ": " << cudaGetErrorString(err) << std::endl;
		std::cerr << "BDA error message: " << msg << std::endl;
		exit(1);
	}
}

#endif
