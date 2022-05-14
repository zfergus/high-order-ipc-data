#include <unordered_set>

#include <polyfem/HashUtils.hpp>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <igl/boundary_facets.h>
#include <igl/oriented_facets.h>

namespace polyfem
{
	namespace
	{
		std::vector<int> sort_face(const Eigen::RowVectorXi f)
		{
			std::vector<int> sorted_face(f.data(), f.data() + f.size());
			std::sort(sorted_face.begin(), sorted_face.end());
			return sorted_face;
		}
	}; // namespace

	// Constructs a list of unique faces represented in a given mesh (V,T)
	//
	// Inputs:
	//   T: #T × 4 matrix of indices of tet corners
	// Outputs:
	//   F: #F × 3 list of faces in no particular order
	template <typename DerivedT, typename DerivedF>
	void faces(
		const Eigen::MatrixBase<DerivedT> &T,
		Eigen::PlainObjectBase<DerivedF> &F)
	{
		Eigen::MatrixXi BF, OF;
		igl::boundary_facets(T, BF);
		igl::oriented_facets(T, OF); // boundary facets + duplicated interior faces

		assert((OF.rows() + BF.rows()) % 2 == 0);
		const int num_faces = (OF.rows() + BF.rows()) / 2;

		F.resize(num_faces, 3);

		F.topRows(BF.rows()) = BF;
		std::unordered_set<std::vector<int>, HashVector> processed_faces;
		for (int fi = 0; fi < BF.rows(); fi++)
		{
			processed_faces.insert(sort_face(BF.row(fi)));
		}

		for (int fi = 0; fi < OF.rows(); fi++)
		{
			std::vector<int> sorted_face = sort_face(OF.row(fi));
			const auto iter = processed_faces.find(sorted_face);
			if (iter == processed_faces.end())
			{
				F.row(processed_faces.size()) = OF.row(fi);
				processed_faces.insert(sorted_face);
			}
		}

		assert(F.rows() == processed_faces.size());
	}
} // namespace polyfem

// Explicit template instantiation
template void polyfem::faces<Eigen::MatrixXi, Eigen::MatrixXi>(
	const Eigen::MatrixBase<Eigen::MatrixXi> &T,
	Eigen::PlainObjectBase<Eigen::MatrixXi> &F);
