#ifndef MVC_HPP
#define MVC_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <igl/per_face_normals.h>
#include <fstream>
#include <string>
#include <igl/copyleft/cgal/mesh_boolean.h>
#include <vector>
#include <array>
#include <igl/is_edge_manifold.h>
#include <igl/remove_unreferenced.h>
#include <igl/orient_outward.h>
#include <igl/centroid.h>
#include <igl/adjacency_matrix.h>
#include <igl/connected_components.h>
#include <stdexcept>

struct Face
{
    std::array<int, 3> vi;
};

struct Mesh
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
};

Eigen::MatrixXd applyDeformation(
    const Eigen::MatrixXd& weights,
    const Eigen::MatrixXd& VdeformedCage);

bool fileExists(const std::string& path);

/// Compute Mean Value Coordinates (MVC) for a single vertex in a robust way.
void computeMVCForOneVertexSimple(
    const Eigen::MatrixXd& C,  // cage vertices (Nx3)
    const Eigen::MatrixXi& CF, // cage faces (Mx3)
    const Eigen::Vector3d eta, // query point (3,)
    Eigen::VectorXd& weights,  // output weights (Nx1)
    Eigen::VectorXd& w_weights // intermediate weights (Nx1)
);

/// Compute MVC for multiple query points.
void computeMVC(
    const Eigen::MatrixXd& C,     // cage vertices (Nx3)
    const Eigen::MatrixXi& CF,    // cage faces (Mx3)
    const Eigen::MatrixXd& eta_m, // query points (Px3)
    Eigen::MatrixXd& phi          // output weights (NxP)
);

Mesh build_visibility_frustum(const Eigen::Vector3d& v, const Eigen::Vector3d& face_center);

Mesh clip_face_along_visibility(
    const Eigen::MatrixXd& cage_V,
    const Eigen::MatrixXi& cage_F,
    const Eigen::Vector3d& v_pos,
    int face_idx);

void computePMVC_CPU(
    const Eigen::MatrixXd& cage_V,
    const Eigen::MatrixXi& cage_F,
    const Eigen::MatrixXd& obj_V,
    Eigen::MatrixXd& pmvc_coords);

Mesh build_closed_prism_around_face(
    const Eigen::MatrixXd& FV,
    double d);

#endif // MVC_HPP