#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace polyfem
{
	// Constructs a list of unique faces represented in a given mesh (V,T)
	//
	// Inputs:
	//   T: #T × 4 matrix of indices of tet corners
	// Outputs:
	//   F: #F × 3 list of faces in no particular order
	template <typename DerivedT, typename DerivedF>
	void faces(
		const Eigen::MatrixBase<DerivedT> &T,
		Eigen::PlainObjectBase<DerivedF> &F);
} // namespace polyfem
