// C++ include
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <thread> 

// Utilities for the Assignment
#include "utils.h"

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"


// Shortcut to avoid Eigen:: everywhere, DO NOT USE IN .h
using namespace Eigen;
using namespace std;

double intersectionPara(Vector3d A, Vector3d B, Vector3d C, Vector3d ray_origin, Vector3d ray_direction);

void raytrace_sphere() {
	std::cout << "Simple ray tracer, one sphere with orthographic projection" << std::endl;

	const std::string filename("sphere_orthographic.png");
	MatrixXd C = MatrixXd::Zero(800,800); // Store the color
	MatrixXd A = MatrixXd::Zero(800,800); // Store the alpha mask

	// The camera is orthographic, pointing in the direction -z and covering the unit square (-1,1) in x and y
	Vector3d origin(-1,1,1);
	Vector3d x_displacement(2.0/C.cols(),0,0);
	Vector3d y_displacement(0,-2.0/C.rows(),0);

	// Single light source
	const Vector3d light_position(-1,1,1);

	for (unsigned i=0; i < C.cols(); ++i) {
		for (unsigned j=0; j < C.rows(); ++j) {
			// Prepare the ray
			Vector3d ray_origin = origin + double(i)*x_displacement + double(j)*y_displacement;
			Vector3d ray_direction = RowVector3d(0,0,-1);

			// Intersect with the sphere
			// NOTE: this is a special case of a sphere centered in the origin and for orthographic rays aligned with the z axis
			Vector2d ray_on_xy(ray_origin(0),ray_origin(1));
			const double sphere_radius = 0.9;

			if (ray_on_xy.norm() < sphere_radius) {
				// The ray hit the sphere, compute the exact intersection point
				Vector3d ray_intersection(ray_on_xy(0),ray_on_xy(1),sqrt(sphere_radius*sphere_radius - ray_on_xy.squaredNorm()));

				// Compute normal at the intersection point
				Vector3d ray_normal = ray_intersection.normalized();

				// Simple diffuse model
				C(i,j) = (light_position-ray_intersection).normalized().transpose() * ray_normal;

				// Clamp to zero
				C(i,j) = std::max(C(i,j),0.);

				// Disable the alpha mask for this pixel
				A(i,j) = 1;
			}
		}
	}

	// Save to png
	write_matrix_to_png(C,C,C,A,filename);

}

double intersectionPara(Vector3d A, Vector3d B, Vector3d C, Vector3d ray_origin, Vector3d ray_direction)
{
	Vector3d paraOrigin = A;
	Vector3d paraU = B-A;
	Vector3d paraV = C-A;

	Matrix3f matrixVar;

	// matrixVar << 0,1,2,3,4,5,6,7,8;

	// cout << matrixVar << endl;
	matrixVar << A(0)-C(0),A(0)-B(0),ray_direction(0),A(1)-C(1),A(1)-B(1),ray_direction(1),A(2)-C(2),A(2)-B(2),ray_direction(2);
	
	Vector3f solution;

	solution << A(0)-ray_origin(0), A(1)-ray_origin(1), A(2)-ray_origin(2);

	Vector3f result = matrixVar.colPivHouseholderQr().solve(solution);

	if(result(0) <= 1 && result(1) <= 1 && result(0) >= 0 && result(1) >= 0 && result(2) >= 0)
	{
		return result(2);
	}

	return -1.0;
}

double diffuseShading(double diffuse_coefficent, double light_intensity, Vector3d normal, Vector3d ray_to_light)
{
	//Assume the vectors of normal and ray_to_light start from intersection point
	double angleValue = normal.dot(ray_to_light);
	angleValue = std::max(0.0,angleValue);
	
	return diffuse_coefficent * light_intensity * angleValue;
}

double specularShading(double specular_coefficient, double light_intensity, double phong_exponent, Vector3d normal, Vector3d view_ray, Vector3d ray_to_light)
{
	Vector3d h = (view_ray + ray_to_light).normalized();

	double angleValue = normal.dot(h);
	angleValue = std::max(0.0, angleValue);

	//TODO: Take the phong exponent and exponent it to light intensity
	angleValue = pow(angleValue, phong_exponent);
	return specular_coefficient * light_intensity * angleValue;
}

void raytrace_parallelogram() {
	std::cout << "Simple ray tracer, one parallelogram with orthographic projection" << std::endl;

	const std::string filename("plane_orthographic.png");
	MatrixXd C = MatrixXd::Zero(800,800); // Store the color
	MatrixXd A = MatrixXd::Zero(800,800); // Store the alpha mask

	// The camera is orthographic, pointing in the direction -z and covering the unit square (-1,1) in x and y
	Vector3d origin(-1,1,1);
	Vector3d x_displacement(2.0/C.cols(),0,0);
	Vector3d y_displacement(0,-2.0/C.rows(),0);

	// TODO: Parameters of the parallelogram (position of the lower-left corner + two sides)
	Vector3d pgram_origin(-0.5,0.5,0);
	Vector3d pgram_u(0,1,0);
	Vector3d pgram_v(-0.5,0,0);

	// Single light source
	const Vector3d light_position(-1,1,1);

	for (unsigned i=0; i < C.cols(); ++i) {
		for (unsigned j=0; j < C.rows(); ++j) {
			// Prepare the ray
			Vector3d ray_origin = origin + double(i)*x_displacement + double(j)*y_displacement;
			Vector3d ray_direction = RowVector3d(0,0,-1);

			// Vector3d ray_direction(0,0,-1);

			// TODO: Check if the ray intersects with the parallelogram
			double res = intersectionPara(pgram_origin, pgram_u, pgram_v,ray_origin,ray_direction);
			if (res != -1) {
				// TODO: The ray hit the parallelogram, compute the exact intersection point
				
				Vector3d intersectionPoint = ray_origin + res * ray_direction; 
				Vector3d ray_intersection(intersectionPoint(0),intersectionPoint(1),intersectionPoint(2));
				// TODO: Compute normal at the intersection point

				//Original code
				Vector3d ray_normal = ray_intersection.normalized();
				// Vector3d ray_normal = pgram_u.cross(pgram_v).normalized();

				// Simple diffuse model
				C(i,j) = (light_position-ray_intersection).normalized().transpose() * ray_normal;

				// Clamp to zero
				C(i,j) = std::max(C(i,j),0.);

				// Disable the alpha mask for this pixel
				A(i,j) = 1;
			}
		}
	}

	// Save to png
	write_matrix_to_png(C,C,C,A,filename);
}

void raytrace_perspective() {
	std::cout << "Simple ray tracer, one parallelogram with perspective projection" << std::endl;

	const std::string filename("plane_perspective.png");
	MatrixXd C = MatrixXd::Zero(800,800); // Store the color
	MatrixXd A = MatrixXd::Zero(800,800); // Store the alpha mask

	// The camera is perspective, pointing in the direction -z and covering the unit square (-1,1) in x and y
	//-1,1,1
	Vector3d origin(1,-1,1);
	Vector3d x_displacement(2.0/C.cols(),0,0);
	Vector3d y_displacement(0,-2.0/C.rows(),0);

	// TODO: Parameters of the parallelogram (position of the lower-left corner + two sides)
	Vector3d pgram_origin(0,-2,-4);
	Vector3d pgram_u(0,-1,-4);
	Vector3d pgram_v(1,-1,-4);

	// Single light source
	const Vector3d light_position(-100,5,1);

	for (unsigned i=0; i < C.cols(); ++i) {
		for (unsigned j=0; j < C.rows(); ++j) {
			// TODO: Prepare the ray (origin point and direction)
			Vector3d ray_origin = double(i)*x_displacement + double(j)*y_displacement;
			Vector3d ray_direction = (ray_origin - origin);

			// TODO: Check if the ray intersects with the parallelogram
			double res = intersectionPara(pgram_origin, pgram_u, pgram_v, ray_origin, ray_direction);
			if (res != -1) {
				// TODO: The ray hit the parallelogram, compute the exact intersection point
				// Vector3d ray_intersection(0,0,0);
				Vector3d intersectionPoint = ray_origin + res * ray_direction; 
				Vector3d ray_intersection(intersectionPoint(0),intersectionPoint(1),intersectionPoint(2));

				// TODO: Compute normal at the intersection point
				Vector3d ray_normal = -ray_intersection.normalized();

				// Simple diffuse model
				C(i,j) = (light_position-ray_intersection).normalized().transpose() * ray_normal;

				// Clamp to zero
				C(i,j) = std::max(C(i,j),0.);

				// Disable the alpha mask for this pixel
				A(i,j) = 1;
			}
		}
	}

	// Save to png
	write_matrix_to_png(C,C,C,A,filename);
}

void raytrace_shading(){
	std::cout << "Simple ray tracer, one sphere with different shading" << std::endl;

	const std::string filename("shading.png");
	MatrixXd C = MatrixXd::Zero(800,800); // Store the color
	MatrixXd A = MatrixXd::Zero(800,800); // Store the alpha mask

	// The camera is perspective, pointing in the direction -z and covering the unit square (-1,1) in x and y
	Vector3d origin(-1,1,1);
	Vector3d x_displacement(2.0/C.cols(),0,0);
	Vector3d y_displacement(0,-2.0/C.rows(),0);

	// Single light source
	const Vector3d light_position(-1,1,1);
	// const Vector3d light_position(x,y,z);
	double ambient = 0.1;
	MatrixXd diffuse = MatrixXd::Zero(800, 800);
	MatrixXd specular = MatrixXd::Zero(800, 800);

	for (unsigned i=0; i < C.cols(); ++i) {
		for (unsigned j=0; j < C.rows(); ++j) {
			// Prepare the ray
			Vector3d ray_origin = origin + double(i)*x_displacement + double(j)*y_displacement;
			Vector3d ray_direction = RowVector3d(0,0,-1);

			// Intersect with the sphere
			// NOTE: this is a special case of a sphere centered in the origin and for orthographic rays aligned with the z axis
			Vector2d ray_on_xy(ray_origin(0),ray_origin(1));
			const double sphere_radius = 0.6;

			if (ray_on_xy.norm() < sphere_radius) {
				// The ray hit the sphere, compute the exact intersection point
				Vector3d ray_intersection(ray_on_xy(0),ray_on_xy(1),sqrt(sphere_radius*sphere_radius - ray_on_xy.squaredNorm()));

				// Compute normal at the intersection point
				Vector3d ray_normal = ray_intersection.normalized();

				//========================================================
				
				Vector3d ray_to_view = origin - ray_intersection;
				Vector3d ray_to_light = light_position - ray_intersection;

				double diffuseCo = 0.5;
				double specularCo = 0.2;
				double lightIntensity = 1;
				double phongExponent = 10;

				double diffuseValue = diffuseShading(diffuseCo, lightIntensity, ray_normal, ray_to_light);
				double specularValue = specularShading(specularCo, lightIntensity, phongExponent, ray_normal, ray_to_view, ray_to_light);


				// double diffuseValue = diffuseShading();

				// TODO: Add shading parameter here
				// diffuse(i,j) = (light_position-ray_intersection).normalized().transpose() * ray_normal;
				// specular(i,j) = (light_position-ray_intersection).normalized().transpose() * ray_normal;
				diffuse(i,j) = diffuseValue;
				specular(i,j) = specularValue;

				// Simple diffuse model
				C(i,j) = lightIntensity * ambient + diffuse(i,j) + specular(i,j);

				// Clamp to zero
				C(i,j) = std::max(C(i,j),0.);

				// Disable the alpha mask for this pixel
				A(i,j) = 1;
			}
		}
	}

	// Save to png
	write_matrix_to_png(C * 0.0125,C*0.5,C*0.0125,A,filename);
}

int main() {
	raytrace_sphere();
	raytrace_parallelogram();
	raytrace_perspective();
	raytrace_shading();
	

	return 0;
}
