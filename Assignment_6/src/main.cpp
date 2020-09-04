// This example is heavily based on the tutorial at https://open.gl

////////////////////////////////////////////////////////////////////////////////
// OpenGL Helpers to reduce the clutter
#include "helpers.h"
// GLFW is necessary to handle the OpenGL context
#include <GLFW/glfw3.h>
// Linear Algebra Library
#include <Eigen/Dense>
#include <Eigen/Geometry>
// STL headers
#include <chrono>
#include <iostream>
#include <fstream>

using namespace Eigen;
using namespace std;

#define PI 3.14159265
////////////////////////////////////////////////////////////////////////////////
struct Ray {
	Vector3f origin;
	Vector3f direction;
	Ray() { }
	Ray(Vector3f o, Vector3f d) : origin(o), direction(d) { }
};

////////////////////////////////////////////////////////////////////////////////
bool checkIntersectTri(Ray ray, Vector3f a, Vector3f b, Vector3f c)
{
	Vector3f A = a;
	Vector3f B = b;
	Vector3f C = c;

	Matrix3f matrixVar;

	// Vector3d ray_direction = ray.direction - ray.origin;
	Vector3f ray_direction = ray.direction;

	matrixVar << A(0)-C(0),A(0)-B(0),ray_direction(0),A(1)-C(1),A(1)-B(1),ray_direction(1),A(2)-C(2),A(2)-B(2),ray_direction(2);
	
	Vector3f solution;

	solution << A(0)-ray.origin(0), A(1)-ray.origin(1), A(2)-ray.origin(2);

	Vector3f result = matrixVar.colPivHouseholderQr().solve(solution);

	if(result(0) >= 0 && result(1) >= 0 && result(0)+result(1) <= 1 && result(2) >= 0)
	{
		cout << "Found intersect" << endl;
		return true;
	}
	return false;
}
////////////////////////////////////////////////////////////////////////////////
// Ray struct

// Mesh object, with both CPU data (Eigen::Matrix) and GPU data (the VBOs)
struct Mesh {
	Eigen::MatrixXf V; // mesh vertices [3 x n]
	Eigen::MatrixXi F; // mesh triangles [3 x m]

	// VBO storing vertex position attributes
	VertexBufferObject V_vbo;

	// VBO storing vertex indices (element buffer)
	VertexBufferObject F_vbo;

	// VAO storing the layout of the shader program for the object 'bunny'
	VertexArrayObject vao;

	Vector3f centroid;

	int startIndexV;
	int endIndexV;

	int startIndexF;
	int endIndexF;

	double minY;
	double maxY;
	double maxX;
	double minX;
	double minZ;
	double maxZ;
	

	void calculateCentroid();	
	void calculateMinMax();
	bool checkIntersect(double xpos, double ypos);

};

void Mesh::calculateCentroid()
{
	double xComp = 0;
	double yComp = 0;
	double zComp = 0;

	for(int i = 0; i < V.cols(); i++)
	{
		xComp += V.col(i)(0);
		yComp += V.col(i)(1);
		zComp += V.col(i)(2);
	}

	centroid << xComp/V.cols(), yComp/V.cols(), zComp/V.cols();
}

void Mesh::calculateMinMax()
{
	double minYL = INT_MAX;
	double maxYL = -INT_MAX;
	double maxXL = -INT_MAX;
	double minXL = INT_MAX;
	double maxZL = -INT_MAX;
	double minZL = INT_MAX;

	for(int i = 0; i < V.cols(); i++)
	{
		if(minYL > V.col(i)(1))
		{
			minYL = V.col(i)(1);
		}

		if(minXL > V.col(i)(0))
		{
			minXL = V.col(i)(0);
		}

		if(maxXL < V.col(i)(0))
		{
			maxXL = V.col(i)(0);
		}

		if(maxYL < V.col(i)(1))
		{
			maxYL = V.col(i)(1);
		}

		if(maxZL < V.col(i)(2))
		{
			maxZL = V.col(i)(2);
		}

		if(minZL > V.col(i)(2))
		{
			minZL = V.col(i)(2);
		}

		minX = minXL;
		minY = minYL;
		maxX = maxXL;
		maxY = maxYL;

		minZ = minZL;
		maxZ = maxZL;
	}
}

bool Mesh::checkIntersect(double xpos, double ypos)
{
	Vector3f origin(xpos,ypos,1);
	Vector3f direction(0,0,-0.3);
	Ray ray(origin,direction);

	for(int i = 0; i < F.cols(); i++)
	{
		Vector3f a = V.col(F.col(i)(0));
		Vector3f b = V.col(F.col(i)(1));
		Vector3f c = V.col(F.col(i)(2));

		if(checkIntersectTri(ray,a,b,c))
		{
			return true;
		}
	}

	return false;
}

////////////////////////////////////////////////////////////////////////////////
void addNewMesh(Mesh &mesh);
void scaleMesh(MatrixXf &V, MatrixXi &F, double scaleFactor);
void printMeshInfo(Mesh mesh);
void printMeshVector();
void translateMesh(double x, double y);
double diffuseShading(double diffuse_coefficent, double light_intensity, Vector3f normal, Vector3f ray_to_light);
double specularShading(double specular_coefficient, double light_intensity, double phong_exponent, Vector3d normal, Vector3d view_ray, Vector3d ray_to_light);
Vector3f getNormal(int i);
void runPhong();
Vector3f getNormalVertex(int a, int b, int c);
void scaleTranslateMesh(Mesh &mesh);
void rotateMesh(Mesh &mesh);
void clickDrag();
void scaleMesh();

Mesh bunny;
////////////////////////////////////////////////////////////////////////////////
Eigen::MatrixXf V; 
Eigen::MatrixXi F;
Eigen::MatrixXf N;
Eigen::MatrixXf C;

VertexBufferObject V_vbo;
VertexBufferObject F_vbo;
VertexBufferObject Normal_vbo;
VertexBufferObject Color_vbo;
VertexArrayObject vao;

vector<Mesh> meshList;
////////////////////////////////////////////////////////////////////////////////
//Select fields
int indexSelect = -1;

// For click translate
MatrixXf tempMeshV;
double xMousePos;
double yMousePos;
double xMouseClick;
double yMouseClick;
bool drag = false;

//Render mode
int mode = 0;

// Light fields
Vector3f light(0,0,1);
////////////////////////////////////////////////////////////////////////////////

// Read a triangle mesh from an off file
void load_off(const std::string &filename, Eigen::MatrixXf &meshV, Eigen::MatrixXi &meshF) {
	std::ifstream in(filename);
	std::string token;
	in >> token;
	int nv, nf, ne;
	in >> nv >> nf >> ne;
	meshV.resize(3, nv);
	meshF.resize(3, nf);
	for (int i = 0; i < nv; ++i) {
		in >> meshV(0, i) >> meshV(1, i) >> meshV(2, i);
	}
	for (int i = 0; i < nf; ++i) {
		int s;
		in >> s >> meshF(0, i) >> meshF(1, i) >> meshF(2, i);
		assert(s == 3);
	}
}

////////////////////////////////////////////////////////////////////////////////

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	// Get viewport size (canvas in number of pixels)
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);

	// Get the size of the window (may be different than the canvas size on retina displays)
	int width_window, height_window;
	glfwGetWindowSize(window, &width_window, &height_window);

	// Get the position of the mouse in the window
	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);

	// Deduce position of the mouse in the viewport
	double highdpi = (double) width / (double) width_window;
	xpos *= highdpi;
	ypos *= highdpi;

	// Convert screen position to the canonical viewing volume
	double xcan = ((xpos/double(width))*2)-1;
	double ycan = (((height-1-ypos)/double(height))*2)-1; // NOTE: y axis is flipped in glfw

	// TODO: Ray-casting for object selection (Ex.3)
	if(action == 0)
	{
		bool found = true;
		cout << xcan << " " << ycan << endl;
		for(int i = meshList.size() - 1; i >= 0; i--)
		{
			if(meshList.at(i).checkIntersect(xcan,ycan))
			{
				Mesh currentMesh = meshList.at(i);
				found = false;
				indexSelect = i;
				cout << "Found intersection at Mesh " << i << endl; 
			
				double vertexLength = currentMesh.endIndexV - currentMesh.startIndexV + 1;
				tempMeshV.resize(3,vertexLength);

				int vindex = currentMesh.startIndexV;
				for(int i = 0; i < vertexLength; i++)
				{
					tempMeshV.col(i) << V.col(vindex);
					vindex++;
				}
			}
		}

		if(found)
			indexSelect = -1;
	}

	if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if(GLFW_PRESS == action)
		{
			xMouseClick = xcan;
			yMouseClick = ycan;
			std::cout << xMouseClick << " " << yMouseClick << std::endl;
            drag = true;
		}
        else if(GLFW_RELEASE == action)
		{
            drag = false;
		}
    }

}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	// Update the position of the first vertex if the keys 1,2, or 3 are pressed
	switch (key) {
		case GLFW_KEY_1:
			if(action == 0)
			{
				Mesh newMesh;
				load_off(DATA_DIR "cube.off", newMesh.V, newMesh.F);
				newMesh.calculateCentroid();
				newMesh.calculateMinMax();
				addNewMesh(newMesh);
				scaleTranslateMesh(meshList.at(meshList.size()-1));
			}
			break;
		case GLFW_KEY_2:
			if(action == 0)
			{
				Mesh newMesh;
				load_off(DATA_DIR "bumpy_cube.off", newMesh.V, newMesh.F);
				newMesh.calculateCentroid();
				newMesh.calculateMinMax();
				addNewMesh(newMesh);
				scaleTranslateMesh(meshList.at(meshList.size()-1));
			}
			break;
		case GLFW_KEY_3:
			//196 190 304
			if(action == 0)
			{
				Mesh newMesh;
				load_off(DATA_DIR "bunny.off", newMesh.V, newMesh.F);
				
				// scaleMesh(newMesh.V, newMesh.F, 4);
				newMesh.calculateCentroid();
				newMesh.calculateMinMax();
				
				addNewMesh(newMesh);
				scaleTranslateMesh(meshList.at(meshList.size()-1));
				
			}
			break;
		case GLFW_KEY_0:
			if(action == 0)
			{
				printMeshVector();
			}
			break;
		case GLFW_KEY_M:
			if(action == 0)
			{
				translateMesh(0.3,0);
			}
		break;
		case GLFW_KEY_R:
			if(action == 0 && indexSelect != -1)
			{
				Mesh something = meshList.at(indexSelect);
				cout << "Rotate Mesh " << indexSelect << endl; 
				rotateMesh(meshList.at(0));
			}
		break;
		case GLFW_KEY_O:
			if(action == 0)
			{
				cout << "Selected Index " << indexSelect << endl;
			}
		break;
		case GLFW_KEY_S:
			if(action == 0 && indexSelect != -1)
			{
				scaleMesh();
			}		
		break;
		case GLFW_KEY_P:
			if(action == 0)
			{
				if(mode == 0)
				{
					cout << "Switch to 1" << endl;
					mode = 1;
				}
				else if(mode == 1)
				{
					cout << "Switch to 2" << endl;
					mode = 2;
				}
				else if(mode == 2)
				{
					cout << "Switch to 0" << endl;
					mode = 0;
				}
				
			}
		default:
			break;
	}
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);

	// Get the size of the window (may be different than the canvas size on retina displays)
	int width_window, height_window;
	glfwGetWindowSize(window, &width_window, &height_window);

	// Deduce position of the mouse in the viewport
	double highdpi = (double) width / (double) width_window;
	xpos *= highdpi;
	ypos *= highdpi;

	// Convert screen position to world coordinates
	double xworld = ((xpos/double(width))*2)-1;
	double yworld = (((height-1-ypos)/double(height))*2)-1; // NOTE: y axis is flipped in glfw
	
	xMousePos = xworld;
	yMousePos = yworld;

	// std::cout << xMousePos << " " << yMousePos << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
//User code
void scaleMesh(MatrixXf &V, MatrixXi &F, double scaleFactor)
{
	Matrix4d scale;
	scale << scaleFactor,0,0,0,  0,scaleFactor, 0, 0,  0,0,scaleFactor,0  ,0,0,0,1;
	for(int i = 0; i < V.cols(); i++)
	{
		Vector4d vertex(V.col(i)(0), V.col(i)(1), V.col(i)(2), 1);
		Vector4d result = scale * vertex;

		V.col(i) << result(0), result(1), result(2);
	}

	V_vbo.update(V);
	F_vbo.update(F);

}

void translateMesh(double x, double y)
{
	Matrix4d translate;
	translate << 1,0,0,x, 0,1,0,y, 0,0,1,0, 0,0,0,1;
	for(int i = 0; i < V.cols(); i++)
	{
		Vector4d vertex(V.col(i)(0), V.col(i)(1), V.col(i)(2), 1);
		Vector4d result = translate * vertex;

		V.col(i) << result(0), result(1), result(2);
	}

	V_vbo.update(V);
	F_vbo.update(F);
}

void addNewMesh(Mesh &mesh)
{
	if(meshList.size() == 0)
	{
		//Nothing in list
		V.resize(3, mesh.V.cols());
		F.resize(3, mesh.F.cols());
		C.resize(3, mesh.V.cols());
		C.setZero();

		V << mesh.V;
		F << mesh.F;

		mesh.startIndexV = 0;
		mesh.endIndexV = mesh.V.cols() - 1;

		mesh.startIndexF = 0;
		mesh.endIndexF = mesh.F.cols() - 1; 

		V_vbo.update(V);
		F_vbo.update(F);
		Color_vbo.update(C);

		meshList.push_back(mesh);
	}
	else
	{
		int sizeV = V.cols();
		int sizeF = F.cols();

		cout << V.size() << endl;
		cout << F.size() << endl;

		MatrixXf tempV(V.rows(), V.cols() + mesh.V.cols());
		MatrixXi tempF(F.rows(), F.cols() + mesh.F.cols());

		tempV << V,mesh.V;
		
		for(int i = 0; i < tempF.cols(); i++)
		{
			if(i < sizeF)
			{
				tempF.col(i) << F.col(i);
			}
			else 
			{
				// cout << i - sizeF << endl;
				tempF.col(i)(0) = mesh.F.col(i - sizeF)(0) + sizeV;
				tempF.col(i)(1) = mesh.F.col(i - sizeF)(1) + sizeV;
				tempF.col(i)(2) = mesh.F.col(i - sizeF)(2) + sizeV;
			}
		}

		V.conservativeResize(tempV.rows(), tempV.cols());
		F.conservativeResize(tempV.rows(), tempF.cols());
		C.conservativeResize(tempV.rows(), tempV.cols());

		C.setZero();

		V << tempV;
		F << tempF;

		mesh.startIndexV = sizeV;
		mesh.endIndexV = mesh.V.cols() + sizeV - 1;

		mesh.startIndexF = sizeF;
		mesh.endIndexF = mesh.F.cols() + sizeF - 1;

		V_vbo.update(V);
		F_vbo.update(F);

		meshList.push_back(mesh);
	}
	
}

void scaleTranslateMesh(Mesh &mesh)
{
	// Scale Mesh to fit unit one

	double diffX = mesh.maxX - mesh.minX;
	double diffY = mesh.maxY - mesh.minY;
	double diffZ = mesh.maxZ - mesh.minZ;

	double xScale = 0;
	double yScale = 0;
	double zScale = 0;

	xScale = 1/diffX;
	yScale = 1/diffY;
	zScale = 1/diffZ;

	Matrix4d scale(4,4);
	scale << xScale,0,0,0,  0,yScale,0,0,  0,0,zScale,0  ,0,0,0,1;

	// Scale every shape
	for(int i = 0; i < mesh.V.cols(); i++)
	{
		Vector4d vertex(mesh.V.col(i)(0), mesh.V.col(i)(1), mesh.V.col(i)(2), 1);
		Vector4d result = scale * vertex;

		mesh.V.col(i) << result(0), result(1), result(2);
	}

	mesh.calculateCentroid();
	mesh.calculateMinMax();

	// Translate every shape
	double xTrans = 0 - mesh.centroid(0);
	double yTrans = 0 - mesh.centroid(1);
	double zTrans = 0 - mesh.centroid(2);

	Matrix4f transMatrix(4,4);
	transMatrix << 1,0,0,xTrans, 0,1,0,yTrans, 0,0,1,zTrans, 0,0,0,1;

	for(int i = 0; i < mesh.V.cols(); i++)
	{
		Vector4f vertex(mesh.V.col(i)(0), mesh.V.col(i)(1), mesh.V.col(i)(2), 1);
		Vector4f result = transMatrix * vertex;

		mesh.V.col(i) << result(0), result(1), result(2);
	}

	// Update the matrix
	int colIndex = 0;
	for(int i = mesh.startIndexV; i < mesh.endIndexV + 1; i++)
	{
		V.col(i) << mesh.V.col(colIndex);
		colIndex++;
	}

	V_vbo.update(V);
	F_vbo.update(F);

	mesh.calculateCentroid();
	mesh.calculateMinMax();
}

void clickDrag()
{
	if(drag && indexSelect != -1)
	{
		Mesh curr = meshList.at(indexSelect);
		double diffX = (xMousePos - xMouseClick);
		double diffY = (yMousePos - yMouseClick);

		Matrix4f transMat(4,4);
		transMat << 1,0,0,diffX, 0,1,0,diffY, 0,0,1,0, 0,0,0,1;

		MatrixXf resultMat(4,tempMeshV.cols());

		//Do result
		for(int i = 0; i < tempMeshV.cols(); i++)
		{
			Vector4f vertex;
			vertex(0) = tempMeshV.col(i)(0);
			vertex(1) = tempMeshV.col(i)(1);
			vertex(2) = tempMeshV.col(i)(2);
			vertex(3) = 1;

			resultMat.col(i) << transMat * vertex;
		}

		// Update V
		int vindex = meshList.at(indexSelect).startIndexV;
		for(int i = 0; i < resultMat.cols(); i++)
		{
			V.col(vindex)(0) = resultMat.col(i)(0);
			V.col(vindex)(1) = resultMat.col(i)(1);
			V.col(vindex)(2) = resultMat.col(i)(2);
			vindex++;
		}

		//Update meshList
		for(int i = 0; i < resultMat.cols(); i++)
		{
			meshList.at(indexSelect).V.col(i)(0) = resultMat.col(i)(0);
			meshList.at(indexSelect).V.col(i)(1) = resultMat.col(i)(1);
			meshList.at(indexSelect).V.col(i)(2) = resultMat.col(i)(2);
		}

		meshList.at(indexSelect).calculateCentroid();
		meshList.at(indexSelect).calculateMinMax();
		V_vbo.update(V);
	}
}

void rotateMesh(Mesh &mesh)
{
	double rotDeg = (10*PI/180);

	double xtrans = 0 - mesh.centroid(0);
	double ytrans = 0 - mesh.centroid(1);
	double ztrans = 0 - mesh.centroid(2);
	
	Matrix4Xf rotateMat(4,4);
	Matrix4Xf transMat(4,4);
	Matrix4Xf transBackMat(4,4);
	rotateMat << cos(rotDeg),0,sin(rotDeg),0, 0,1,0,0, -sin(rotDeg),0,cos(rotDeg),0, 0,0,0,1;
	transMat << 1,0,0,xtrans, 0,1,0,ytrans, 0,0,1,ztrans, 0,0,0,1;
	transBackMat << 1,0,0,-xtrans, 0,1,0,-ytrans, 0,0,1,-ztrans, 0,0,0,1;
	
	// transMat << 0,0,

	// Matrix4f transMatrix;
	// transMatrix << 1,0,0,0.2, 0,1,0,0.2, 0,0,1,0, 0,0,0,1;

	for(int i = 0; i < meshList.at(indexSelect).V.cols(); i++)
	{
		Vector4f vertex(meshList.at(indexSelect).V.col(i)(0), meshList.at(indexSelect).V.col(i)(1), meshList.at(indexSelect).V.col(i)(2), 1);
		Vector4f result = transBackMat * rotateMat * transMat * vertex;

		meshList.at(indexSelect).V.col(i) << result(0), result(1), result(2);
	}

	// Update the matrix
	int colIndex = 0;
	for(int i = meshList.at(indexSelect).startIndexV; i < meshList.at(indexSelect).endIndexV + 1; i++)
	{
		V.col(i) << meshList.at(indexSelect).V.col(colIndex);
		colIndex++;
	}



	V_vbo.update(V);
	F_vbo.update(F);

	mesh.calculateCentroid();
	mesh.calculateMinMax();
}

void scaleMesh()
{
	double xtrans = 0 - meshList.at(indexSelect).centroid(0);
	double ytrans = 0 - meshList.at(indexSelect).centroid(1);
	double ztrans = 0 - meshList.at(indexSelect).centroid(2);

	double xScale = 1.1;
	double yScale = 1.1;
	double zScale = 1.1;
	
	Matrix4Xf scale(4,4);
	Matrix4Xf transMat(4,4);
	Matrix4Xf transBackMat(4,4);
	// rotateMat << cos(rotDeg),0,sin(rotDeg),0, 0,1,0,0, -sin(rotDeg),0,cos(rotDeg),0, 0,0,0,1;
	scale << xScale,0,0,0,  0,yScale,0,0,  0,0,zScale,0  ,0,0,0,1;
	transMat << 1,0,0,xtrans, 0,1,0,ytrans, 0,0,1,ztrans, 0,0,0,1;
	transBackMat << 1,0,0,-xtrans, 0,1,0,-ytrans, 0,0,1,-ztrans, 0,0,0,1;
	
	// transMat << 0,0,

	// Matrix4f transMatrix;
	// transMatrix << 1,0,0,0.2, 0,1,0,0.2, 0,0,1,0, 0,0,0,1;

	for(int i = 0; i < meshList.at(indexSelect).V.cols(); i++)
	{
		Vector4f vertex(meshList.at(indexSelect).V.col(i)(0), meshList.at(indexSelect).V.col(i)(1), meshList.at(indexSelect).V.col(i)(2), 1);
		Vector4f result = transBackMat * scale * transMat * vertex;

		meshList.at(indexSelect).V.col(i) << result(0), result(1), result(2);
	}

	// Update the matrix
	int colIndex = 0;
	for(int i = meshList.at(indexSelect).startIndexV; i < meshList.at(indexSelect).endIndexV + 1; i++)
	{
		V.col(i) << meshList.at(indexSelect).V.col(colIndex);
		colIndex++;
	}

	// //UPdate 
	// for(int i = 0; i < meshList.at(i).V.cols(); i++)
	// {
	// 	meshList.at(indexSelect).V.col(i) << mesh.V.col(i)(0), mesh.V.col(i)(1), mesh.V.col(i)(2);
	// }

	V_vbo.update(V);
	F_vbo.update(F);

 	meshList.at(indexSelect).calculateCentroid();
	meshList.at(indexSelect).calculateMinMax();
}

void printMeshInfo(Mesh mesh)
{
	cout << "V Size: " << mesh.V.cols() << endl;
	cout << "F Size: " << mesh.F.cols() << endl;
	cout << "V s/e: " << mesh.startIndexV << " " << mesh.endIndexV << endl;
	cout << "F s/e: " << mesh.startIndexF << " " << mesh.endIndexF << endl;
	cout << "MinX: " << mesh.minX << " MaxX: " << mesh.maxX << " MinY: " << mesh.minY << " MaxY: " << mesh.maxY << endl;
	cout << "Centroid: " << mesh.centroid(0) << " " << mesh.centroid(1) << " " << mesh.centroid(2);
	cout << endl;
}

void printMeshVector()
{
	for(int i = 0; i < meshList.size(); i++)
	{
		printMeshInfo(meshList.at(i));
	}
}

double diffuseShading(double diffuse_coefficent, double light_intensity, Vector3f normal, Vector3f ray_to_light)
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

Vector3f getNormal(int i)
{
	Vector3f a(V.col(F.col(i)(0))(0),V.col(F.col(i)(0))(1),V.col(F.col(i)(0))(2));
	Vector3f b(V.col(F.col(i)(1))(0),V.col(F.col(i)(1))(1),V.col(F.col(i)(1))(2));
	Vector3f c(V.col(F.col(i)(2))(0),V.col(F.col(i)(2))(1),V.col(F.col(i)(2))(2));

	Vector3f ab;
	Vector3f cb;

	ab = b - a;
	cb = b - c;

	Vector3f normal = ab.cross(cb);

	return normal;
}

Vector3f getNormalVertex(int a, int b, int c)
{
	Vector3f v1(V.col(a)(0),V.col(a)(1),V.col(a)(2));
	Vector3f v2(V.col(b)(0),V.col(b)(1),V.col(b)(2));
	Vector3f v3(V.col(c)(0),V.col(c)(1),V.col(c)(2));

	Vector3f v12;
	Vector3f v13;

	v12 = v1 - v2;
	v13 = v1 - v3;

	Vector3f normal = v12.cross(v13);
	return normal;
}

void runPhong()
{
	MatrixXf normal_sum;
	normal_sum.resize(4,V.cols());
	normal_sum.setZero();

	for(int i = 0; i < F.cols(); i++)
	{
		int v1 = F.col(i)(0);
		int v2 = F.col(i)(1);
		int v3 = F.col(i)(2);

		Vector3f v1Normal = getNormalVertex(v1,v2,v3).normalized();
		Vector3f v2Normal = getNormalVertex(v2,v1,v3).normalized();
		Vector3f v3Normal = getNormalVertex(v3,v1,v2).normalized();

		normal_sum.col(v1) << v1Normal, normal_sum.col(v1)(3) + 1;
		normal_sum.col(v2) << v1Normal, normal_sum.col(v2)(3) + 1;
		normal_sum.col(v3) << v1Normal, normal_sum.col(v3)(3) + 1;

	}

	//Divide by number of references
	for(int i = 0; i < normal_sum.cols(); i++)
	{
		normal_sum.col(i)(0) = normal_sum.col(i)(0)/(normal_sum.col(i)(3));
		normal_sum.col(i)(1) = normal_sum.col(i)(1)/(normal_sum.col(i)(3));
		normal_sum.col(i)(2) = normal_sum.col(i)(2)/(normal_sum.col(i)(3));
	}

	for(int i = 0; i < V.cols(); i++)
	{
		Vector3f normal = -normal_sum.col(i);
		normal = normal.normalized();
		// cout << normal << endl;
		Vector3f point(V.col(i)(0), V.col(i)(1), V.col(i)(2));
		Vector3f ray_to_light = light - point;

		double diffuseValue = diffuseShading(0.5,0.1, normal,ray_to_light);
		// cout << diffuseValue << endl;

		C.col(i) << diffuseValue * 40, diffuseValue* 40, diffuseValue * 40;
	}
}

////////////////////////////////////////////////////////////////////////////////

int main(void) {
	// Initialize the GLFW library
	if (!glfwInit()) {
		return -1;
	}

	// Activate supersampling
	glfwWindowHint(GLFW_SAMPLES, 8);

	// Ensure that we get at least a 3.2 context
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);

	// On apple we have to load a core profile with forward compatibility
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// Create a windowed mode window and its OpenGL context
	GLFWwindow * window = glfwCreateWindow(640, 640, "[Float] Hello World", NULL, NULL);
	if (!window) {
		glfwTerminate();
		return -1;
	}

	// Make the window's context current
	glfwMakeContextCurrent(window);

	// Load OpenGL and its extensions
	if (!gladLoadGL()) {
		printf("Failed to load OpenGL and its extensions");
		return(-1);
	}
	printf("OpenGL Version %d.%d loaded", GLVersion.major, GLVersion.minor);

	int major, minor, rev;
	major = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
	minor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
	rev = glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION);
	printf("OpenGL version recieved: %d.%d.%d\n", major, minor, rev);
	printf("Supported OpenGL is %s\n", (const char*)glGetString(GL_VERSION));
	printf("Supported GLSL is %s\n", (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

	// Initialize the OpenGL Program
	// A program controls the OpenGL pipeline and it must contains
	// at least a vertex shader and a fragment shader to be valid
	Program program;
	const GLchar* vertex_shader = R"(
		#version 150 core

		uniform mat4 model;
		uniform mat4 view;
		uniform mat4 proj;

		in vec3 position;
		in vec3 normal;
		in vec3 color;

		out vec3 Color;

		void main() {
			Color = color;
			gl_Position = proj * view * model * vec4(position, 1.0);
		}
	)";

	const GLchar* fragment_shader = R"(
		#version 150 core

		uniform vec3 triangleColor;
		uniform int p;
		in vec3 Color;
		out vec4 outColor;

		void main() {
			outColor = vec4(Color, 1.0);
		}
	)";

	// Compile the two shaders and upload the binary to the GPU
	// Note that we have to explicitly specify that the output "slot" called outColor
	// is the one that we want in the fragment buffer (and thus on screen)
	program.init(vertex_shader, fragment_shader, "outColor");

	// Prepare a dummy bunny object
	// We need to initialize and fill the two VBO (vertex positions + indices),
	// and use a VAO to store their layout when we use our shader program later.
	{
		// Initialize the VBOs
		V_vbo.init(GL_FLOAT, GL_ARRAY_BUFFER);
		F_vbo.init(GL_UNSIGNED_INT, GL_ELEMENT_ARRAY_BUFFER);
		Normal_vbo.init(GL_FLOAT, GL_ARRAY_BUFFER);
		Color_vbo.init(GL_FLOAT, GL_ARRAY_BUFFER);

		Vector3f a(0, 0.5, -0.9);
		Vector3f b(0.5,-0.5,-0.9);
		Vector3f c(-0.5,-0.5, -0.9);

		//Normals

		// Vertex positions
		V.resize(3, 3);
		// V <<
		// 	0, 0.5, -0.5,
		// 	0.5, -0.5, -0.5,
		// 	-0.9, -0.9, -0.9;
		V.col(0) = a;
		V.col(1) = b;
		V.col(2) = c;
		V_vbo.update(V);

		// Triangle indices
		F.resize(3, 1);
		F << 0, 1, 2;
		F_vbo.update(F);

		// Create a new VAO for the bunny. and bind it
		vao.init();
		vao.bind();

		// Bind the element buffer, this information will be stored in the current VAO
		F_vbo.bind();

		C.resize(3,3);
		C <<
		1,1,1,
		1,1,1,
		1,1,1;

		Color_vbo.update(C);

		// The vertex shader wants the position of the vertices as an input.
		// The following line connects the VBO we defined above with the position "slot"
		// in the vertex shader
		program.bindVertexAttribArray("position", V_vbo);
		program.bindVertexAttribArray("normal", Normal_vbo);
		program.bindVertexAttribArray("color", Color_vbo);

		// Unbind the VAO when I am done
		vao.unbind();
	}

	// For the first exercises, 'view' and 'proj' will be the identity matrices
	// However, the 'model' matrix must change for each model in the scene
	Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
	program.bind();
	glUniformMatrix4fv(program.uniform("view"), 1, GL_FALSE, I.data());
	glUniformMatrix4fv(program.uniform("proj"), 1, GL_FALSE, I.data());

	// Save the current time --- it will be used to dynamically change the triangle color
	auto t_start = std::chrono::high_resolution_clock::now();

	// Register the keyboard callback
	glfwSetKeyCallback(window, key_callback);

	// Register the mouse callback
	glfwSetMouseButtonCallback(window, mouse_button_callback);

	// Register drag callback
	glfwSetCursorPosCallback(window, cursor_position_callback);

	// Loop until the user closes the window
	while (!glfwWindowShouldClose(window)) {
		// Set the size of the viewport (canvas) to the size of the application window (framebuffer)
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		glViewport(0, 0, width, height);

		// Clear the framebuffer
		glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glEnable(GL_DEPTH_TEST);  

		// Bind your program
		program.bind();

		{
			// Bind the VAO for the bunny
			vao.bind();

			// Model matrix for the bunny
			glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, I.data());

			// Set the uniform value depending on the time difference
			auto t_now = std::chrono::high_resolution_clock::now();
			float time = std::chrono::duration_cast<std::chrono::duration<float>>(t_now - t_start).count();
			// glUniform3f(program.uniform("triangleColor"), (float)(sin(time * 4.0f) + 1.0f) / 2.0f, 0.0f, 0.0f);
			glUniform3f(program.uniform("triangleColor"), 1,1,1);
			glUniform1f(program.uniform("p"), 0);
			// Draw the triangles
			// GL_TRIANGLES
			// GL_LINE_LOOP
			// cout << bunny.F_vbo.scalar_type << endl;
			// cout << F_vbo.scalar_type << endl;

			if(mode == -1)
			{
				C.setConstant(0.8);
				Color_vbo.update(C);
				F_vbo.update(F);
				glDrawElements(GL_TRIANGLES, 3 * F.cols(), F_vbo.scalar_type, 0);
			}
			else if(mode == 0)
			{
				//Need loop to draw every 3.
				for(int i = 0; i < F.cols(); i++)
				{
					Eigen::MatrixXi temp(3,1);
					temp << F.col(i)(0), F.col(i)(1), F.col(i)(2);
					//&indices[0]

					F_vbo.update(temp);


					glDrawElements(GL_LINE_LOOP, 3, F_vbo.scalar_type, 0);
				}
				C.setZero();
				Color_vbo.update(C);
			}
			else if(mode == 1)
			{
				glUniform1f(program.uniform("p"), 1);
				for(int i = 0; i < F.cols(); i++)
				{
					Eigen::MatrixXi temp(3,1);
					// Eigen::MatrixXi tempColor(3,3);

					temp << F.col(i)(0), F.col(i)(1), F.col(i)(2);

					Vector3f point(V.col(F.col(i)(0))(0),V.col(F.col(i)(0))(1),V.col(F.col(i)(0))(2));

					double z1 = V.col(F.col(i)(0))(2);
					double z2 = V.col(F.col(i)(1))(2);
					double z3 = V.col(F.col(i)(2))(2);

					Vector3f normal = getNormal(i).normalized();
					Vector3f ray_to_light = (light - point).normalized();
					double diffuseValue = diffuseShading(0.1, 0.1, normal, ray_to_light);
					
					F_vbo.update(temp);

					C.col(temp(0)) << diffuseValue * 50, diffuseValue * 50, diffuseValue * 50;
					C.col(temp(1)) << diffuseValue * 50, diffuseValue * 50, diffuseValue * 50;
					C.col(temp(2)) << diffuseValue * 50, diffuseValue * 50, diffuseValue * 50;

					Color_vbo.update(C);

					
					glDrawElements(GL_TRIANGLES, 3, F_vbo.scalar_type, 0);
					C.setZero();
					Color_vbo.update(C);
					glDrawElements(GL_LINE_LOOP, 3, F_vbo.scalar_type, 0);
					
				}
			}
			else if(mode == 2)
			{
				runPhong();
				Color_vbo.update(C);
				V_vbo.update(V);
				F_vbo.update(F);
				glDrawElements(GL_TRIANGLES, 3 * F.cols(), F_vbo.scalar_type, 0);
			}

			clickDrag();
		}

		// Swap front and back buffers
		glfwSwapBuffers(window);

		// Poll for and process events
		glfwPollEvents();

		// exit(0);
	}

	// Deallocate opengl memory
	program.free();
	vao.free();
	V_vbo.free();
	F_vbo.free();

	// Deallocate glfw internals
	glfwTerminate();
	return 0;
}
