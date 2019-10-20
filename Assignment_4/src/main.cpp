////////////////////////////////////////////////////////////////////////////////
// C++ include
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <stack>

// Eigen for matrix operations
#include <Eigen/Dense>
#include <Eigen/Geometry>

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"
#include "utils.h"

// JSON parser library (https://github.com/nlohmann/json)
#include "json.hpp"
using json = nlohmann::json;

// Shortcut to avoid Eigen:: everywhere, DO NOT USE IN .h
using namespace Eigen;

////////////////////////////////////////////////////////////////////////////////
// Define types & classes
////////////////////////////////////////////////////////////////////////////////

struct Ray {
	Vector3d origin;
	Vector3d direction;
	Ray() { }
	Ray(Vector3d o, Vector3d d) : origin(o), direction(d) { }
};

struct Light {
	Vector3d position;
	Vector3d intensity;
};

struct Intersection {
	Vector3d position;
	Vector3d normal;
	double ray_param;
};

struct Camera {
	bool is_perspective;
	Vector3d position;
	double field_of_view; // between 0 and PI
	double focal_length;
	double lens_radius; // for depth of field
};

struct Material {
	Vector3d ambient_color;
	Vector3d diffuse_color;
	Vector3d specular_color;
	double specular_exponent; // Also called "shininess"

	Vector3d reflection_color;
	Vector3d refraction_color;
	double refraction_index;
};

struct Object {
	Material material;
	virtual ~Object() = default; // Classes with virtual methods should have a virtual destructor!
	virtual bool intersect(const Ray &ray, Intersection &hit) = 0;
};

// We use smart pointers to hold objects as this is a virtual class
typedef std::shared_ptr<Object> ObjectPtr;

struct Sphere : public Object {
	Vector3d position;
	double radius;

	virtual ~Sphere() = default;
	virtual bool intersect(const Ray &ray, Intersection &hit) override;
};

struct Parallelogram : public Object {
	Vector3d origin;
	Vector3d u;
	Vector3d v;

	virtual ~Parallelogram() = default;
	virtual bool intersect(const Ray &ray, Intersection &hit) override;
};

struct Triangle
{
	Vector3d A;
	Vector3d B;
	Vector3d C;
	int index;

	Vector3d centroid;

	Triangle() = default;
	Triangle(const Vector3d &A, const Vector3d &B, const Vector3d &C, int i);

};

Triangle::Triangle(const Vector3d &a, const Vector3d &b, const Vector3d &c, int i)
{
	A = a;
	B = b;
	C = c;
	index = i;


	Vector3d temp(A(0)+B(0)+C(0)/3, A(1)+B(1)+C(1)/3, A(2)+B(2)+C(2)/3);
	centroid = temp;
}



struct AABBTree {
	struct Node {
		AlignedBox3d bbox;
		Triangle triangle;
		int parent; // Index of the parent node (-1 for root)
		int left; // Index of the left child (-1 for a leaf)
		int right; // Index of the right child (-1 for a leaf)
		// int triangle; // Index of the node triangle (-1 for internal nodes)
	};

	std::vector<Node> nodes;
	int root;

	int recurseTree(std::vector<Triangle> &list, int parentIndex);
	AABBTree() = default; // Default empty constructor
	AABBTree(const MatrixXd &V, const MatrixXi &F); // Build a BVH from an existing mesh
};

struct Mesh : public Object {
	MatrixXd vertices; // n x 3 matrix (n points)
	MatrixXi facets; // m x 3 matrix (m triangles)

	AABBTree bvh;

	Mesh() = default; // Default empty constructor
	Mesh(const std::string &filename);
	int traverseTree(const Ray &ray, std::vector<AABBTree::Node> node, Intersection &hit);
	virtual ~Mesh() = default;
	virtual bool intersect(const Ray &ray, Intersection &hit) override;
};

struct Scene {
	Vector3d background_color;
	Vector3d ambient_light;

	Camera camera;
	std::vector<Material> materials;
	std::vector<Light> lights;
	std::vector<ObjectPtr> objects;
};

////////////////////////////////////////////////////////////////////////////////

// Read a triangle mesh from an off file
void load_off(const std::string &filename, MatrixXd &V, MatrixXi &F) {
	std::ifstream in(filename);
	std::string token;
	in >> token;
	int nv, nf, ne;
	in >> nv >> nf >> ne;
	V.resize(nv, 3);
	F.resize(nf, 3);
	for (int i = 0; i < nv; ++i) {
		in >> V(i, 0) >> V(i, 1) >> V(i, 2);
	}
	for (int i = 0; i < nf; ++i) {
		int s;
		in >> s >> F(i, 0) >> F(i, 1) >> F(i, 2);
		assert(s == 3);
	}
}

Mesh::Mesh(const std::string &filename) {
	// Load a mesh from a file (assuming this is a .off file), and create a bvh
	load_off(filename, vertices, facets);

	bvh = AABBTree(vertices, facets);
}

////////////////////////////////////////////////////////////////////////////////
// BVH Implementation
////////////////////////////////////////////////////////////////////////////////

//Sorting functions
bool sortVectorX(Triangle &current, Triangle &other)
{
	return current.centroid(0) < other.centroid(0);
}

bool sortVectorY(Triangle &current, Triangle &other)
{
	return current.centroid(1) < other.centroid(1);
}

bool sortVectorZ(Triangle &current, Triangle &other)
{
	return current.centroid(2) < other.centroid(2);
}


// Bounding box of set of triangles
int getBoundBox(std::vector<Triangle> &list)
{

	double minX = INT_MAX;
	double maxX = INT_MIN;
	double minY = INT_MAX;
	double maxY = INT_MIN;
	double minZ = INT_MAX;
	double maxZ = INT_MIN;

	int max = list.size();

	for(int i = 0; i < max; i++)
	{
		if(list[i].centroid(0) < minX)
		{
			minX = list[i].centroid(0);
			// triMinX = list[i];
		}
		
		if(list[i].centroid(0) > maxX)
		{
			maxX = list[i].centroid(0);
			// triMaxX = list[i];
		}

		if(list[i].centroid(1) < minY)
		{
			minY = list[i].centroid(1);
			// triMinY = list[i];
		}

		if(list[i].centroid(1) > maxY)
		{
			maxY = list[i].centroid(1);
			// triMaxY = list[i];
		}

		if(list[i].centroid(2) < minZ)
		{
			minZ = list[i].centroid(2);
			// triMinZ = list[i];
		}

		if(list[i].centroid(2) > maxZ)
		{
			maxZ = list[i].centroid(2);
			// triMaxZ = list[i];
		}

	}

	double xlen = maxX - minX;
	double ylen = maxY - minY;
	double zlen = maxZ - minZ;

	if(xlen >= ylen && xlen >= zlen)
	{
		return 0;
	}
	else if(ylen >= xlen && ylen >= zlen)
	{
		return 1;
	}
	else if(zlen >= ylen && zlen >= xlen)
	{
		return 2;
	}

	return -1;

}

// Bounding box of a triangle
AlignedBox3d bbox_triangle(const Vector3d &a, const Vector3d &b, const Vector3d &c) {
	AlignedBox3d box;
	box.extend(a);
	box.extend(b);
	box.extend(c);
	return box;
}

int AABBTree::recurseTree(std::vector<Triangle> &list,  int parentIndex)
{
	AABBTree::Node newNode;


	// std::cout << std::endl;
	// std::cout << list.size() << std::endl;

	//Base case
	if(list.size() == 1)
	{
		// std::cout << "Found leaf" << std::endl;

		newNode.bbox = bbox_triangle(list[0].A, list[0].B, list[0].C);
		newNode.parent = parentIndex;
		newNode.left = -1;
		newNode.right = -1;
		newNode.triangle = list[0];

		//Add to node vector
		
		nodes.push_back(newNode);

		// std::cout << "At Node " << nodes.size()-1 << std::endl;
		// for(Triangle tri : list)
		// {
		// 	std::cout << tri.index;
		// }

		// std::cout << std::endl;

		// std::cout << "Left " << nodes[nodes.size()-1].left << std::endl;
		// std::cout << "Right " << nodes[nodes.size()-1].right << std::endl;

		std::cout << "Reach leaf " << nodes.size()-1 << std::endl;
		std::cout << nodes[nodes.size()-1].bbox.min() << std::endl;
		std::cout << nodes[nodes.size()-1].bbox.max() << std::endl;
		// std::cout << nodes[nodes.size()-1].triangle.C << std::endl;
		std::cout << std::endl;

		return nodes.size() - 1;
	}

	//Get bounding box and figure out longest dim

	
	int longestLength = getBoundBox(list);

	//Sort according to which is the longest length
	if(longestLength == 0)
	{
		std::sort(list.begin(), list.end(), &sortVectorX);
	}
	else if(longestLength == 1)
	{
		std::sort(list.begin(), list.end(), &sortVectorY);
	}
	else if(longestLength == 2)
	{
		std::sort(list.begin(), list.end(), &sortVectorZ);
	}

	// std::cout << "After sorting" << std::endl;

	//Divide the sorted array into two and get two seperate arrays
	int dividePoint = list.size()/2;
	std::vector<Triangle> left;
	std::vector<Triangle> right;

	int maxLen = list.size();
	for(int i = 0; i < maxLen; i++)
	{
		if(i < dividePoint)
		{	
			left.push_back(list[i]);
		}
		else
		{
			right.push_back(list[i]);
		}
	}

	// std::cout << "Before recursion" << std::endl;
	nodes.push_back(newNode);
	int index = nodes.size() - 1;


	// std::cout << std::endl;

	//Do recursion on both 
	int leftIndex = recurseTree(left, index);
	int rightIndex = recurseTree(right, index);
	AlignedBox3d leftBox; 
	leftBox.extend(nodes[leftIndex].bbox);
	AlignedBox3d rightBox;
	rightBox.extend(nodes[rightIndex].bbox);

	// std::cout << "After recursion" << std::endl;

	AlignedBox3d boundingBox;
	boundingBox.extend(leftBox);
	boundingBox.extend(rightBox);

	//Add information to current node
	nodes[index].bbox = boundingBox;
	//Branch, has no triangle

	nodes[index].parent = parentIndex;
	nodes[index].left = leftIndex;
	nodes[index].right = rightIndex;

	// std::cout << "At Node " << index << std::endl;
	// for(Triangle tri : list)
	// {
	// 	std::cout << tri.index;
	// }

	// std::cout << std::endl;

	// std::cout << "Left " << nodes[index].left << std::endl;
	// std::cout << "Right " << nodes[index].right << std::endl;


	return index;
}

AABBTree::AABBTree(const MatrixXd &V, const MatrixXi &F) {
	// Compute the centroids of all the triangles in the input mesh
	MatrixXd centroids(F.rows(), V.cols());
	centroids.setZero();

	Vector3d A;
	Vector3d B;
	Vector3d C;

	std::vector<Triangle> vectorTriangle;

	std::cout << "Start reading mesh file" << std::endl;
	//Read from input and create a vector of triangles, triangle has centroid and sides
	for (int i = 0; i < F.rows(); ++i) {
		for (int k = 0; k < F.cols(); ++k) {

			if(k == 0)
			{
				//A = V.row(F(i,k));
				A(0) = V(F(i,k),0);
				A(1) = V(F(i,k),1);
				A(2) =  V(F(i,k),2);
			}
			else if(k == 1)
			{
				
				B(0) = V(F(i,k),0);
				B(1) = V(F(i,k),1);
				B(2) =  V(F(i,k),2);
			}
			else if(k == 2)
			{
				C(0) = V(F(i,k),0);
				C(1) = V(F(i,k),1);
				C(2) =  V(F(i,k),2);

			}
			centroids.row(i) += V.row(F(i, k));
		}

		Triangle newTri(A,B,C,i);
		vectorTriangle.push_back(newTri);
		// AlignedBox3d newBox = bbox_triangle(A,B,C);
		// vectorCentroid.push_back(centroids.row(i) /= F.cols());
		centroids.row(i) /= F.cols();
	}


	std::cout << "Finish reading mesh" << std::endl;
	// TODO (Assignment 3)

	// Method (1): Top-down approach.
	// Split each set of primitives into 2 sets of roughly equal size,
	// based on sorting the centroids along one direction or another.


	//Do basical setup

	int nodeLen = F.rows();

	//set temp
	// Node temp;
	// for(int i = 0; i < (nodeLen*2)-1; i++)
	// {
	// 	nodes.push_back(temp);
	// }
	root = 0;

	// std::cout << vectorTriangle.size() << std::endl;
	// std::cout << nodes.size() << std::endl;

	recurseTree(vectorTriangle, -1);

	std::cout << "Finish building tree" << std::endl;

	// Method (2): Bottom-up approach.
	// Merge nodes 2 by 2, starting from the leaves of the forest, until only 1 tree is left.
}

////////////////////////////////////////////////////////////////////////////////

bool Sphere::intersect(const Ray &ray, Intersection &hit) {
	// TODO (Assignment 2)
	
	//Look at slides on how to intersect a sphere
	// std::cout << position(0) << std::endl;
	// std::cout << radius << std::endl;

	Vector3d rayLine = ray.direction - ray.origin;

	double A = (rayLine(0)*rayLine(0)) + (rayLine(1) * rayLine(1)) + (rayLine(2) * rayLine(2));
	double B = (2 * rayLine(0) * (ray.origin(0) - position(0)) + (2 * rayLine(1) * (ray.origin(1) - position(1))) + (2 * rayLine(2) * (ray.origin(2) - position(2))));
	double C = (position(0)*position(0) + position(1)*position(1) + position(2)*position(2) + ray.origin(0)*ray.origin(0) + ray.origin(1)*ray.origin(1) + ray.origin(2)*ray.origin(2) - 2 * (ray.origin(0) * position(0) + ray.origin(1)*position(1) + ray.origin(2)*position(2)));
	
	double discriminant = pow(B,2) - (4*A*C);

	if(discriminant < 0)
	{
		return false;
	}
	else if(discriminant == 0)
	{
		double t = -B/(2 * A);
		if(t >= 0)
		{
			Vector3d intersection = ray.origin + t * rayLine;
			hit.position = intersection;
			hit.normal = intersection.normalized();
			hit.ray_param = t;
			return true;
		}
		else 
		{
			return false;
		}
		
	}
	else
	{
		double t1 = (-B - sqrt(discriminant))/(2*A);
		double t2 = (-B + sqrt(discriminant))/(2*A);

		//Get shortest point
		if(t1 < t2 && t1 >= 0)
		{
			Vector3d intersection = ray.origin + t1 * rayLine;
			hit.position = intersection;
			hit.normal = intersection.normalized();
			hit.ray_param = t1;

			return true;
		}
		else if(t2 < t1 && t2 >= 0)
		{
			Vector3d intersection = ray.origin + t2 * rayLine;
			hit.position = intersection;
			hit.normal = intersection.normalized();
			hit.ray_param = t2;

			return true;
		}
		else
		{	
			return false;
		}
	}
}
bool Parallelogram::intersect(const Ray &ray, Intersection &hit) {
	// TODO (Assignment 2)


	// Assume u and v are vectors from parallelogram origin, rather than actual origin
	Vector3d A = origin;
	Vector3d B = u;
	Vector3d C = v;

	Matrix3f matrixVar;

	// Vector3d ray_direction = ray.direction - ray.origin;
	Vector3d ray_direction = ray.direction;

	matrixVar << A(0)-C(0),A(0)-B(0),ray_direction(0),A(1)-C(1),A(1)-B(1),ray_direction(1),A(2)-C(2),A(2)-B(2),ray_direction(2);
	
	Vector3f solution;

	solution << A(0)-ray.origin(0), A(1)-ray.origin(1), A(2)-ray.origin(2);

	Vector3f result = matrixVar.colPivHouseholderQr().solve(solution);

	if(result(0) <= 1 && result(1) <= 1 && result(0) >= 0 && result(1) >= 0 && result(2) >= 0)
	{
		Vector3d intersection = ray.origin + result(2) * ray_direction;
		hit.position = intersection;
		hit.normal = intersection.normalized();
		hit.ray_param = result(2);
		return true;
	}

	return false;
}

// -----------------------------------------------------------------------------
int leftHitBox = 0;
int rightHitBox = 0;
int leftHitTri = 0;
int rightHitTri = 0;

bool intersect_triangle(const Ray &ray, const Vector3d &a, const Vector3d &b, const Vector3d &c, Intersection &hit, const int test) {
	// TODO (Assignment 3)
	// 
	// Compute whether the ray intersects the given triangle.
	// If you have done the parallelogram case, this should be very similar to it.

	Vector3d A = a;
	Vector3d B = b;
	Vector3d C = c;

	Matrix3f matrixVar;

	// Vector3d ray_direction = ray.direction - ray.origin;
	Vector3d ray_direction = ray.direction;

	matrixVar << A(0)-C(0),A(0)-B(0),ray_direction(0),A(1)-C(1),A(1)-B(1),ray_direction(1),A(2)-C(2),A(2)-B(2),ray_direction(2);
	
	Vector3f solution;

	solution << A(0)-ray.origin(0), A(1)-ray.origin(1), A(2)-ray.origin(2);

	Vector3f result = matrixVar.colPivHouseholderQr().solve(solution);

	if(result(0) >= 0 && result(1) >= 0 && result(0)+result(1) <= 1 && result(2) >= 0)
	{
		if(test == 1)
		{
			leftHitTri++;
		}

		if(test == 2)
		{	
			rightHitTri++;
		}
		// std::cout << "Found triangle hit" << std::endl;
		// std::cout << a << std::endl;
		// std::cout << b << std::endl;
		// std::cout << c << std::endl;
		// std::cout << std::endl;
		Vector3d intersection = ray.origin + result(2) * ray_direction;
		hit.position = intersection;
		hit.normal = intersection.normalized();
		hit.ray_param = result(2);

		// std::cout << "Found triangle hit " << hit.position << std::endl;

		return true;
	}
	return false;
}

bool intersect_box(const Ray &ray, const AlignedBox3d &box, Intersection &intersect) {
	// TODO (Assignment 3)
	// 
	// Compute whether the ray intersects the given box.
	// There is no need to set the resulting normal and ray parameter, since
	// we are not testing with the real surface here anyway.

	Vector3d A = box.corner(box.BottomLeftFloor);
	Vector3d B = box.corner(box.BottomRightFloor);
	Vector3d C = box.corner(box.TopLeftFloor);
	Vector3d D = box.corner(box.TopRightFloor);
	Vector3d E = box.corner(box.BottomLeftCeil);
	Vector3d F = box.corner(box.BottomRightCeil);
	Vector3d G = box.corner(box.TopLeftCeil);
	Vector3d H = box.corner(box.TopRightCeil);

	Intersection temp;
	double distance;
	

	Parallelogram planeOne;
	planeOne.origin = A;
	planeOne.u = B;
	planeOne.v = C;
	bool one = planeOne.intersect(ray,temp);

	distance = (temp.position - ray.origin).norm();

	Parallelogram planeTwo;
	planeTwo.origin = A;
	planeTwo.u = E;
	planeTwo.v = C;
	bool two = planeTwo.intersect(ray,temp);

	if((temp.position - ray.origin).norm() < distance)
	{
		distance = (temp.position - ray.origin).norm();
		intersect.position = temp.position;
		intersect.normal = temp.normal;

		intersect.ray_param = temp.ray_param;
	}

	Parallelogram planeThree;
	planeThree.origin = A;
	planeThree.u = B;
	planeThree.v = E;
	bool three = planeThree.intersect(ray,temp);

	if((temp.position - ray.origin).norm() < distance)
	{
		distance = (temp.position - ray.origin).norm();
		intersect.position = temp.position;
		intersect.normal = temp.normal;

		intersect.ray_param = temp.ray_param;
	}

	Parallelogram planeFour;
	planeFour.origin = H;
	planeFour.u = G;
	planeFour.v = D;
	bool four = planeFour.intersect(ray,temp);

	if((temp.position - ray.origin).norm() < distance)
	{
		distance = (temp.position - ray.origin).norm();
		intersect.position = temp.position;
		intersect.normal = temp.normal;

		intersect.ray_param = temp.ray_param;
	}

	Parallelogram planeFive;
	planeFive.origin = H;
	planeFive.u = G;
	planeFive.v = F;
	bool five = planeFive.intersect(ray,temp);

	if((temp.position - ray.origin).norm() < distance)
	{
		distance = (temp.position - ray.origin).norm();
		intersect.position = temp.position;
		intersect.normal = temp.normal;

		intersect.ray_param = temp.ray_param;
	}

	Parallelogram planeSix; 
	planeSix.origin = H;
	planeSix.u = D;
	planeSix.v = F;
	bool six = planeSix.intersect(ray,temp);

	if((temp.position - ray.origin).norm() < distance)
	{
		distance = (temp.position - ray.origin).norm();
		intersect.position = temp.position;
		intersect.normal = temp.normal;

		intersect.ray_param = temp.ray_param;
	}

	if(one || two || three || four || five || six)
	{
		return true;
	}

	return false;
}

int d = 1;
bool Mesh::intersect(const Ray &ray, Intersection &closest_hit) {
	// TODO (Assignment 3)

	// Method (1): Traverse every triangle and return the closest hit.

	if(d == 0)
	{
		int maxLength = facets.rows();

		Vector3d A;
		Vector3d B;
		Vector3d C;

		Intersection min_inter;
		Vector3d temp(0,0,0);
		min_inter.position = temp;
		min_inter.ray_param = INT_MAX;

		bool hit;
		bool overallHit = false;

		for(int i = 0; i < maxLength; i++)
		{
			for(int j = 0; j < 4; j++)
			{
				if(j == 0)
				{
					int index = facets(i,j);
					Vector3d a(vertices(index,0),vertices(index,1),vertices(index,2));
					A = a;
				}
				else if(j == 1)
				{
					int index = facets(i,j);
					Vector3d b(vertices(index,0),vertices(index,1),vertices(index,2));
					B = b;
				}
				else if(j == 2)
				{
					int index = facets(i,j);
					Vector3d c(vertices(index,0),vertices(index,1),vertices(index,2));
					C = c;
				}
			}
			
			
			hit = intersect_triangle(ray, A, B, C, closest_hit, 0);
			if(hit && min_inter.ray_param > closest_hit.ray_param)
			{
				min_inter = closest_hit;
				// std::cout << "Found hit" << std::endl;
				overallHit = true;
			}
		}	

		closest_hit = min_inter;
		return overallHit;
	}

	// Method (2): Traverse the BVH tree and test the intersection with a
	// triangles at the leaf nodes that intersects the input ray.

	//Check if ray hits first box

	if(d == 1)
	{
		AlignedBox3d box = bvh.nodes[0].bbox;


		int nodeIndex = 0;
		int maxLength = bvh.nodes.size();
		int left = bvh.nodes[nodeIndex].left;
		int right = bvh.nodes[nodeIndex].right; 

		Intersection wholeIntersect;

		while(nodeIndex < maxLength && intersect_box(ray,box, wholeIntersect))
		{
			// if(nodeIndex == 0)
			// {
			// 	std::cout << "Hit Total box" << std::endl;
			
			// }
			// else 
			// {
			// 	std::cout << "Hit Box in Node " << nodeIndex << std::endl;

			// }
			
			left = bvh.nodes[nodeIndex].left;
			right = bvh.nodes[nodeIndex].right;

			if(bvh.nodes[nodeIndex].left == -1 && bvh.nodes[nodeIndex].right == -1)
			{
				Triangle hitTri = bvh.nodes[nodeIndex].triangle;
				// std::cout << "Found triangle at leaf" << std::endl;
				// std::cout << hitTri.A << std::endl;
				// std::cout << hitTri.B << std::endl;
				// std::cout << hitTri.C << std::endl;
				if(nodeIndex == 1)
				{
					leftHitBox++;
				}

				if(nodeIndex == 2)
				{
					rightHitBox++;
				}

				return intersect_triangle(ray,hitTri.A, hitTri.B, hitTri.C, closest_hit, nodeIndex);
			}
			else 
			{
				// std::cout << "Found branch at " << nodeIndex << std::endl;
				AlignedBox3d leftBox = bvh.nodes[left].bbox;
				AlignedBox3d rightBox = bvh.nodes[right].bbox;

				bool leftBol = false;
				bool rightBol = false;

				Intersection leftIntersect;
				Intersection rightIntersect;

				if(intersect_box(ray, rightBox, leftIntersect))
				{
					// std::cout << "Intersect left at " << nodeIndex << std::endl;
					nodeIndex = right;
					box = rightBox;
					rightBol = true;
				}

				if(intersect_box(ray, leftBox, rightIntersect))
				{
					nodeIndex = left;
					box = leftBox;
					leftBol = true;
				}
				
				if(leftBol && rightBol)
				{
					//If hit twice, take the one closest to array.
					double leftDis = (leftIntersect.position - ray.origin).norm();
					double rightDis = (rightIntersect.position - ray.origin).norm();

					if(leftDis > rightDis)
					{
						nodeIndex = right;
						box = rightBox;

						// std::cout << "Hit both select right " << std::endl;
					}
					else
					{
						nodeIndex = left;
						box = leftBox;

						// std::cout << "Hit both select left " << std::endl;
					}

					
				}
				
				if(!leftBol && !rightBol)
				{
					//Go back through parents 
					// int parentIndex = bvh.nodes[nodeIndex].parent;
					// if(parentIndex != -1)
					// {
					// 	Node parent = bvh.nodes[parentIndex];

					// 	if(parent.left == nodeIndex)
					// 	{
					// 		//Go right
					// 		nodeIndex = parent.right;
					// 		box = bvh.nodes[parent.right].bbox;
					// 	}
					// 	else
					// 	{
					// 		//Go left
					// 		nodeIndex = parent.left;
					// 		box = bvh.nodes[parent.left].bbox;
					// 	}
					// }

					nodeIndex = maxLength;
					

					std::cout << "No box intersect. This is an error " << std::endl;
				}

				leftBol = false;
				rightBol = false;
			}
		}

			

			return false;
	}

	return false;
}

////////////////////////////////////////////////////////////////////////////////
// Define ray-tracing functions
////////////////////////////////////////////////////////////////////////////////

// Function declaration here (could be put in a header file)
Vector3d ray_color(const Scene &scene, const Ray &ray, const Object &object, const Intersection &hit, int max_bounce);
Object * find_nearest_object(const Scene &scene, const Ray &ray, Intersection &closest_hit);
bool is_light_visible(const Scene &scene, const Ray &ray, const Light &light);
Vector3d shoot_ray(const Scene &scene, const Ray &ray, int max_bounce);

Mesh meshObj;

// -----------------------------------------------------------------------------

Vector3d ray_color(const Scene &scene, const Ray &ray, const Object &obj, const Intersection &hit, int max_bounce) {
	// Material for hit object
	const Material &mat = obj.material;

	// Ambient light contribution
	Vector3d ambient_color = obj.material.ambient_color.array() * scene.ambient_light.array();

	// Punctual lights contribution (direct lighting)
	Vector3d lights_color(0, 0, 0);
	for (const Light &light : scene.lights) {
		Vector3d Li = (light.position - hit.position).normalized();
		Vector3d N = hit.normal;

		// TODO (Assignment 2, shadow rays)

		// Diffuse contribution
		Vector3d diffuse = mat.diffuse_color * std::max(Li.dot(N), 0.0);

		// TODO (Assignment 2, specular contribution)
		Vector3d specular(0, 0, 0);

		// Attenuate lights according to the squared distance to the lights
		Vector3d D = light.position - hit.position;
		lights_color += (diffuse + specular).cwiseProduct(light.intensity) /  D.squaredNorm();
	}

	// TODO (Assignment 2, reflected ray)
	Vector3d reflection_color(0, 0, 0);

	// TODO (Assignment 2, refracted ray)
	Vector3d refraction_color(0, 0, 0);

	// Rendering equation
	Vector3d C = ambient_color + lights_color + reflection_color + refraction_color;

	return C;
}

// -----------------------------------------------------------------------------

Object * find_nearest_object(const Scene &scene, const Ray &ray, Intersection &closest_hit) {
	int closest_index = -1;
	// TODO (Assignment 2, find nearest hit)
	std::shared_ptr<Mesh> sceneMesh = std::dynamic_pointer_cast<Mesh>(scene.objects.at(0));

	if (!(sceneMesh->intersect(ray,closest_hit))) {
		// Return a NULL pointer
		return nullptr;
	} else {
		// Return a pointer to the hit object. Don't forget to set 'closest_hit' accordingly!
		return scene.objects[0].get();
	}
}

bool is_light_visible(const Scene &scene, const Ray &ray, const Light &light) {
	// TODO (Assignment 2, shadow ray)
	return true;
}

Vector3d shoot_ray(const Scene &scene, const Ray &ray, int max_bounce) {
	Intersection hit;
	if (Object * obj = find_nearest_object(scene, ray, hit)) {
		// 'obj' is not null and points to the object of the scene hit by the ray
		return ray_color(scene, ray, *obj, hit, max_bounce);
	} else {
		// 'obj' is null, we must return the background color
		return scene.background_color;
	}
}

////////////////////////////////////////////////////////////////////////////////

void render_scene(const Scene &scene) {
	std::cout << "Simple ray tracer." << std::endl;

	int w = 640;
	int h = 480;
	MatrixXd R = MatrixXd::Zero(w, h);
	MatrixXd G = MatrixXd::Zero(w, h);
	MatrixXd B = MatrixXd::Zero(w, h);
	MatrixXd A = MatrixXd::Zero(w, h); // Store the alpha mask

	// The camera always points in the direction -z
	// The sensor grid is at a distance 'focal_length' from the camera center,
	// and covers an viewing angle given by 'field_of_view'.
	double aspect_ratio = double(w) / double(h);
	double scale_y = 2.5; // TODO: Stretch the pixel grid by the proper amount here
	double scale_x = 2.5; //

	// The pixel grid through which we shoot rays is at a distance 'focal_length'
	// from the sensor, and is scaled from the canonical [-1,1] in order
	// to produce the target field of view.
	//Original: Vector3d grid_origin(-scale_x, scale_y, -scene.camera.focal_length);
	Vector3d grid_origin(-scale_x, scale_y, -scene.camera.focal_length);
	Vector3d x_displacement(2.0/w*scale_x, 0, 0);
	Vector3d y_displacement(0, -2.0/h*scale_y, 0);

	for (unsigned i = 0; i < w; ++i) {
		std::cout << std::fixed << std::setprecision(2);
		std::cout << "Ray tracing: " << (100.0 * i) / w << "%\r" << std::flush;
		for (unsigned j = 0; j < h; ++j) {
			// TODO (Assignment 2, depth of field)
			Vector3d shift = grid_origin + (i+0.5)*x_displacement + (j+0.5)*y_displacement;

			// Prepare the ray
			Ray ray;


			if (scene.camera.is_perspective) {
				// Perspective camera
				// TODO (Assignment 2, perspective camera)
				ray.origin = scene.camera.position;
				ray.direction = (shift - scene.camera.position).normalized();

				//use an intersection point

			} else {
				// Orthographic camera
				ray.origin = scene.camera.position + Vector3d(shift[0], shift[1], 0);
				ray.direction = Vector3d(0, 0, -1);
			}

			Intersection hit;


			int max_bounce = 5;
			Vector3d C = shoot_ray(scene, ray, max_bounce);
			R(i, j) = C(0);
			G(i, j) = C(1);
			B(i, j) = C(2);
			A(i, j) = 1;
		}
	}

	std::cout << "Ray tracing: 100%  " << std::endl;

	// Save to png
	const std::string filename("raytrace.png");
	write_matrix_to_png(R, G, B, A, filename);
}

////////////////////////////////////////////////////////////////////////////////

Scene load_scene(const std::string &filename) {
	Scene scene;

	// Load json data from scene file
	json data;
	std::ifstream in(filename);
	in >> data;

	// Helper function to read a Vector3d from a json array
	auto read_vec3 = [] (const json &x) {
		return Vector3d(x[0], x[1], x[2]);
	};

	// Read scene info
	scene.background_color = read_vec3(data["Scene"]["Background"]);
	scene.ambient_light = read_vec3(data["Scene"]["Ambient"]);

	// Read camera info
	scene.camera.is_perspective = data["Camera"]["IsPerspective"];
	scene.camera.position = read_vec3(data["Camera"]["Position"]);
	scene.camera.field_of_view = data["Camera"]["FieldOfView"];
	scene.camera.focal_length = data["Camera"]["FocalLength"];
	scene.camera.lens_radius = data["Camera"]["LensRadius"];

	// Read materials
	for (const auto &entry : data["Materials"]) {
		Material mat;
		mat.ambient_color = read_vec3(entry["Ambient"]);
		mat.diffuse_color = read_vec3(entry["Diffuse"]);
		mat.specular_color = read_vec3(entry["Specular"]);
		mat.reflection_color = read_vec3(entry["Mirror"]);
		mat.refraction_color = read_vec3(entry["Refraction"]);
		mat.refraction_index = entry["RefractionIndex"];
		mat.specular_exponent = entry["Shininess"];
		scene.materials.push_back(mat);
	}

	// Read lights
	for (const auto &entry : data["Lights"]) {
		Light light;
		light.position = read_vec3(entry["Position"]);
		light.intensity = read_vec3(entry["Color"]);
		scene.lights.push_back(light);
	}

	// Read objects
	for (const auto &entry : data["Objects"]) {
		ObjectPtr object;
		if (entry["Type"] == "Sphere") {
			auto sphere = std::make_shared<Sphere>();
			sphere->position = read_vec3(entry["Position"]);
			sphere->radius = entry["Radius"];
			object = sphere;
		} else if (entry["Type"] == "Parallelogram") {
			// TODO
		} else if (entry["Type"] == "Mesh") {
			// Load mesh from a file
			
			std::string filename = std::string(DATA_DIR) + entry["Path"].get<std::string>();
			object = std::make_shared<Mesh>(filename);
			// meshObj = Mesh(filename);
			// std::cout << object.facets << std::endl;

			std::cout << "Reading from " << filename << std::endl;

		}
		object->material = scene.materials[entry["Material"]];
		scene.objects.push_back(object);
	}

	return scene;
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " scene.json" << std::endl;
		return 1;
	}
	
	Scene scene = load_scene(argv[1]);
	render_scene(scene);

	std::cout << "Found left box " << leftHitBox << std::endl;
	std::cout << "Found right box " << rightHitTri << std::endl;
	std::cout << "Found left tri " << leftHitTri << std::endl;
	std::cout << "Found right tri " << rightHitTri << std::endl;

	// std::shared_ptr<Mesh> sceneMesh = std::dynamic_pointer_cast<Mesh>(scene.objects.at(0));
	
	// std::cout << sceneMesh->bvh.nodes[0].bbox.min() << std::endl;
	// std::cout << sceneMesh->bvh.nodes[0].bbox.max() << "\n" << std::endl;

	// std::cout << sceneMesh->bvh.nodes[1].bbox.min() << std::endl;
	// std::cout << sceneMesh->bvh.nodes[1].bbox.max() << "\n" << std::endl;

	// std::cout << sceneMesh->bvh.nodes[2].bbox.min() << std::endl;
	// std::cout << sceneMesh->bvh.nodes[2].bbox.max() << "\n" << std::endl;


	// Vector3d a(0.25,0.1,5);
	// Vector3d b(0.25,0.1,4);

	// Ray ray(a,b);
	// Intersection intersect;

	// std::cout << sceneMesh->intersect(ray,intersect) << std::endl;

	// std::cout << intersect.position << std::endl;

	

	return 0;
}

// Vector3d a(0,0,0);
	// Vector3d b(1,0,5);
	// Vector3d c(0,1,1);
	// AlignedBox3d test = bbox_triangle(a,b,c);

	// std::cout << "Max Corner: " << std::endl << test.max() << std::endl;

	// std::cout << "Bottom Left Floor: " << std::endl << test.corner(test.BottomLeftFloor) << std::endl;
	// std::cout << "Bottom Right Floor: " << std::endl << test.corner(test.BottomRightFloor) << std::endl;
	// std::cout << "Top Left Floor: " << std::endl << test.corner(test.TopLeftFloor) << std::endl;
	// std::cout << "Top Right Floor: " << std::endl << test.corner(test.TopRightFloor) << std::endl;
	// std::cout << "Bottom Left Ceil: " << std::endl << test.corner(test.BottomLeftCeil) << std::endl;
	// std::cout << "Bottom Right Ceil: " << std::endl << test.corner(test.BottomRightCeil) << std::endl;
	// std::cout << "Top Left Ceil: " << std::endl << test.corner(test.TopLeftCeil) << std::endl;
	// std::cout << "Top Right Ceil: " << std::endl << test.corner(test.TopRightCeil) << std::endl;
