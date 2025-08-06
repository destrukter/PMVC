#include "mvc.hpp"
#include "pmvc.hpp"
#include <igl/read_triangle_mesh.h>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/edges.h>
#include <igl/writeOBJ.h>

int main(int argc, char* argv[])
{
    Eigen::MatrixXd Vmesh;
    Eigen::MatrixXi Fmesh;

    Eigen::MatrixXd Vcage;
    Eigen::MatrixXi Fcage;
    Eigen::MatrixXi Ecage;

    Eigen::MatrixXd VcageDeformed;
    Eigen::MatrixXi FcageDeformed;
    Eigen::MatrixXi EcageDeformed;

    std::string meshPath = "";
    std::string cagePath = "";
    std::string deformedCagePath = "";

    bool mvc = false;
    bool pmvc_cpu = false;

    if (argc == 5)
    {
        meshPath = argv[1];
        cagePath = argv[2];
        deformedCagePath = argv[3];
        std::string arg = argv[4];
        if (arg == "--MVC")
        {
            mvc = true;
        }
        else if (arg == "--PMVCCPU")
        {
            pmvc_cpu = true;
        }
        else
        {
            std::cerr << "Need to specify --MVC or --PMVCCPU as 4th argument" << std::endl;
            return 1;
        }
    }
    else
    {
        if (argc == 4)
            std::cerr << " 4 arguments given.";
        if (argc == 3)
            std::cerr << " 3 arguments given.";
        if (argc == 2)
            std::cerr << " 2 arguments given.";
        if (argc == 1)
            std::cerr << " 1 arguments given." << argv[0];
        if (argc == 0)
            std::cerr << " 0 arguments given.";
        std::cerr << "Call needs to have to following structure: ./example pathtomesh.obj pathtocage.obj pathtodeformedcage.obj and --MVC or --PMVCCPU as 4th argument" << std::endl;
        return 1;
    }
   


    const std::string defaultMesh = "C:/Users/frede/wkspaces/libigl-example-project/meshes/armadilloman.obj";
    const std::string defaultCage = "C:/Users/frede/wkspaces/libigl-example-project/meshes/armadilloman_cages_triangulated.obj";
    const std::string defaultDeformedCage = "C:/Users/frede/wkspaces/libigl-example-project/meshes/armadilloman_cages_triangulated_deformed_1.obj";

    std::string finalMeshPath = fileExists(meshPath) ? meshPath : defaultMesh;
    std::string finalCagePath = fileExists(cagePath) ? cagePath : defaultCage;
    std::string finalDeformedCagePath = fileExists(deformedCagePath) ? deformedCagePath : defaultDeformedCage;

    igl::read_triangle_mesh(finalMeshPath, Vmesh, Fmesh);
    igl::read_triangle_mesh(finalCagePath, Vcage, Fcage);
    igl::read_triangle_mesh(finalDeformedCagePath, VcageDeformed, FcageDeformed);

    igl::edges(Fcage, Ecage);
    igl::edges(FcageDeformed, EcageDeformed);

    Eigen::MatrixXd weights;
    if (mvc)
    {
        computeMVC(Vcage, Fcage, Vmesh, weights);
    }
    else if (pmvc_cpu)
    {
        computePMVC_CPU(Vcage, Fcage, Vmesh, weights);
    }

    Eigen::MatrixXd VmeshDeformed = applyDeformation(weights, VcageDeformed);

    double offset = (Vmesh.col(0).maxCoeff() - Vmesh.col(0).minCoeff()) * 1.5;
    VmeshDeformed.col(0).array() += offset;
    VcageDeformed.col(0).array() += offset;

    igl::opengl::glfw::Viewer viewer;

    viewer.data().set_mesh(Vmesh, Fmesh);
    viewer.data().set_face_based(false);

    viewer.append_mesh();
    // viewer.data(1).set_points(Vcage, Eigen::RowVector3d(1, 0, 0));
    viewer.data(1).set_edges(Vcage, Ecage, Eigen::RowVector3d(0, 0, 1));

    viewer.append_mesh();
    // viewer.data(2).set_points(VcageDeformed, Eigen::RowVector3d(1, 0, 0));
    viewer.data(2).set_edges(VcageDeformed, EcageDeformed, Eigen::RowVector3d(0, 0, 1));

    viewer.append_mesh();
    viewer.data(3).set_mesh(VmeshDeformed, Fmesh);
    viewer.data(3).set_face_based(false);
    viewer.data(3).set_colors(Eigen::RowVector3d(1, 0, 0));

    std::cout << "Starting viewer..." << std::endl;
    viewer.launch();

    return 0;
}
