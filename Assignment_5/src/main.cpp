// This example is heavily based on the tutorial at https://open.gl

////////////////////////////////////////////////////////////////////////////////
// OpenGL Helpers to reduce the clutter
#include "helpers.h"
// GLFW is necessary to handle the OpenGL context
#include <GLFW/glfw3.h>
// Linear Algebra Library
#include <Eigen/Dense>
// Timer
#include <chrono>

#include <iostream>

#include <math.h>

#define PI 3.14159265

using namespace Eigen;
////////////////////////////////////////////////////////////////////////////////

//Declare functions
void drawTriangle(double xworld, double yworld,int button, int action);
void drawLines();
bool checkTriangle(Vector2d &a, Vector2d &b, Vector2d &c, Vector2d &sol);
int selectTriangle(Vector2d mouse);
void clickDrag();
void deleteTriangle();
void selectColor(Program program, float time);
void rotateTriangle(int direction);
void scaleTri(int i);
void changeVertexColor(double r, double g, double b);
void deselectColor();

//Triangle class for keeping track of each triangle position
struct Triangle
{
	Vector2d A;
	Vector2d B;
	Vector2d C;

	//Color
	MatrixXf color;

	//Centroid of the triangle
	Vector2d centroid;

	//Tells where this triangle is relational to first vertex in the V matrix.
	int indexCol = 0;

	// The id of the triangle. From ascending order of lowest to highest
	int id = -1;

	Triangle() = default;
	Triangle(const Vector2d &a, const Vector2d &b, const Vector2d &c);
	void updateCentroid();
};

Triangle::Triangle(const Vector2d &a, const Vector2d &b, const Vector2d &c)
{
	A = a;
	B = b;
	C = c;

	Vector2d temp((A(0)+B(0)+C(0))/3, (A(1)+B(1)+C(1))/3);
	centroid = temp;

	color.resize(3,3);

	color << 1,1,1, 1,1,1, 1,1,1;
}

void Triangle::updateCentroid()
{
	Vector2d temp((A(0)+B(0)+C(0))/3, (A(1)+B(1)+C(1))/3);
	centroid = temp;
}

//Vector of Triangles
std::vector<Triangle> triangleList;

// VertexBufferObject wrapper
VertexBufferObject VBO;
// VertexBufferObject line_VBO;

// Store color array
VertexBufferObject VBO_color;

// Contains the vertex positions
Eigen::MatrixXf V(2,3);

//Contains color of each vertex
Eigen::MatrixXf C(3,3);

// Flag for insertion
bool insert = false;
int drawStatus = 0;
int triangleVPos = 0;
int triangleNum = 0;
int pointOneIndex = 0;

Eigen::MatrixXf tempMatrix(2,3);

// Fields for selecting and dragging
int indexSelect = -1;
bool translate = false;
bool drag = false;

// Pos for mouse
double xMousePos = 0;
double yMousePos = 0;

// origin mouse click
double xMouseClick = -1;
double yMouseClick = -1;

// vertex origin
Vector3d origin_a;
Vector3d origin_b;
Vector3d origin_c;

//Original centroid
Vector2d origin_centroid;

// Fields for rotating triangles
bool rotate = false;

//Fields for color
bool color = false;
int colorVertex = -1;
int colorTriangleIndex = -1;
Vector3d defaultColor(1.0f, 1.0f, 1.0f);

void findVertex(double xpos, double ypos)
{
	if(color)
	{

		double distance = INT_MAX;
		int index = -1;

		for(int i = 0; i < V.cols(); i++)
		{
			double currDis = sqrt(pow(V.col(i)(0)-xpos, 2) + pow(V.col(i)(1)-ypos,2));

			if(currDis < distance)
			{
				distance = currDis;
				index = i;
			}
		}

		colorTriangleIndex = index/3;

		std::cout << "Select vertex " << index << ". In triangle " << colorTriangleIndex << std::endl;
		colorVertex = index;
	}	
}

void selectVertex(Program program)
{
	if(color && colorVertex != -1)
	{
		glDrawArrays(GL_POINTS, colorVertex, 1);
	}
}

void changeVertexColor(double r, double g, double b)
{

	if(colorVertex != -1)
	{
		int triangleIndexLocal = colorVertex/3;
		int indexLocal = colorVertex%3;

		std::cout<< "Triangle Index " << triangleIndexLocal << " Index Local" << indexLocal << std::endl;

		triangleList.at(triangleIndexLocal).color.col(indexLocal)(0) = r;
		triangleList.at(triangleIndexLocal).color.col(indexLocal)(1) = g;
		triangleList.at(triangleIndexLocal).color.col(indexLocal)(2) = b;

		C.col(colorVertex) << r, g, b;	
	}

	VBO_color.update(C);
}

void deselectColor()
{
	//Deselect from transition 
	for(int i = 0; i < C.cols(); i++)
	{
		C.col(i) << defaultColor(0), defaultColor(1), defaultColor(2);
		VBO_color.update(C);
	
	}

	VBO_color.update(C);
}

void selectColor(Program program, float time)
{
	if(indexSelect != -1)
	{
		
		for(int i = 0; i < C.cols(); i++)
		{
			// Turn the previous selected original color
			int triangleIndexLocal = i/3;
			int indexLocal = i%3;

			C.col(i)(0) = triangleList.at(triangleIndexLocal).color.col(indexLocal)(0);
			C.col(i)(1) = triangleList.at(triangleIndexLocal).color.col(indexLocal)(1);
			C.col(i)(2) = triangleList.at(triangleIndexLocal).color.col(indexLocal)(2);

			// C.col(i)(0) = 1;
			// C.col(i)(1) = 1;
			// C.col(i)(2) = 1;
		}

		C.col(indexSelect * 3) << 0.0f, 0.0f, 0.0f;
		C.col(indexSelect * 3 + 1) << 0.0f, 0.0f, 0.0f;
		C.col(indexSelect * 3 + 2) << 0.0f, 0.0f, 0.0f;
		VBO_color.update(C);
	}
	else
	{
		// std::cout << "In else for select" << std::endl;
		for(int i = 0; i < C.cols() && !insert; i++)
		{
			int triangleIndexLocal = i/3;
			int indexLocal = i%3;

			if(triangleList.size() != 0)
			{
				C.col(i)(0) = triangleList.at(triangleIndexLocal).color.col(indexLocal)(0);
				C.col(i)(1) = triangleList.at(triangleIndexLocal).color.col(indexLocal)(1);
				C.col(i)(2) = triangleList.at(triangleIndexLocal).color.col(indexLocal)(2);
			}

			// C.col(i)(0) = 1;
			// C.col(i)(1) = 1;
			// C.col(i)(2) = 1;
		}
		
		VBO_color.update(C);

		// glDrawArrays(GL_TRIANGLES, 0, triangleNum * 3);
	}
	
}

void drawTriangle(double xworld, double yworld,int button, int action)
{
	if(insert && button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
	{
		if(drawStatus == 0)
		{
			//Increment triangle num
			triangleNum++;

			//Resize V matrix
			V.conservativeResize(2,triangleNum * 3);	
			C.conservativeResize(3,triangleNum * 3);

			// std::cout << "First point\n" << xworld << yworld << std::endl;
			tempMatrix.col(0) << xworld, yworld;
			V.col(triangleVPos) = tempMatrix.col(0);
			V.col(triangleVPos +1) = tempMatrix.col(0);
			V.col(triangleVPos +2) = tempMatrix.col(0);

			// std::cout << "V matrix " << V << std::endl;
			drawStatus++;

			pointOneIndex = triangleVPos;
		}
		else if(drawStatus == 1)
		{
			// std::cout << "Second point\n"  << xworld << yworld << std::endl;

			tempMatrix.col(1) << xworld, yworld;
			V.col(triangleVPos) = tempMatrix.col(0);
			V.col(triangleVPos + 1) = tempMatrix.col(1);
			V.col(triangleVPos + 2) = tempMatrix.col(1);

			// std::cout << "V matrix " << V << std::endl;
			drawStatus++;
		}
		else if(drawStatus == 2)
		{
			// std::cout << "Third point\n"  << xworld << yworld << std::endl;

			tempMatrix.col(2) << xworld, yworld;
			V.col(triangleVPos) = tempMatrix.col(0);
			V.col(triangleVPos + 1) = tempMatrix.col(1);
			V.col(triangleVPos + 2) = tempMatrix.col(2);

			C.col(triangleVPos) << 1,1,1;
			C.col(triangleVPos + 1) << 1,1,1;
			C.col(triangleVPos + 2) << 1,1,1;

			VBO_color.update(C);

			// std::cout << V << std::endl;

			// Work in progress
			Vector2d a(tempMatrix.col(0)(0), tempMatrix.col(0)(1));
			Vector2d b(tempMatrix.col(1)(0), tempMatrix.col(1)(1));
			Vector2d c(tempMatrix.col(2)(0), tempMatrix.col(2)(1));

			// std::cout << a << std::endl << b << std::endl << c << std::endl;

			Triangle newTri(a,b,c);
			
			newTri.indexCol = triangleVPos;
			newTri.id = triangleNum;
			newTri.color << 1,1,1,1,1,1,1,1,1;
			triangleList.push_back(newTri);

			//Reset draw status to first point
			drawStatus = 0;

			//Move onto next triangle
			triangleVPos += 3;		
		}
	}
}

void drawLines()
{
	if(drawStatus == 1)
	{
		// std::cout << xMousePos << " " << yMousePos << std::endl;
		V.col(triangleVPos+1) << xMousePos, yMousePos;
		VBO.update(V);
	}
	else if(drawStatus == 2)
	{
		V.col(triangleVPos+2) << xMousePos, yMousePos;
		VBO.update(V);
	}
}

int selectTriangle(Vector2d mouse)
{
	//Start from last triangle drawn because it will select the first triangle
	for(int i = triangleList.size() - 1; i >= 0; i--)
	{
		// std::cout << i << std::endl;
		Triangle curr = triangleList.at(i);

		if(checkTriangle(curr.A,curr.B,curr.C, mouse))
		{
			return i;
		}
	}

	// std::cout << " Cannot find triangle" << std::endl;
	return -1;
}

bool checkTriangle(Vector2d &a, Vector2d &b, Vector2d &c, Vector2d &sol)
{
	
	Vector3d ab(b(0) - a(0), b(1) - a(1), 0);
	Vector3d bc(c(0) - b(0), c(1) - b(1), 0);
	Vector3d ca(a(0) - c(0), a(1) - c(1), 0);
	
	Vector3d solA(sol(0) - a(0), sol(1) - a(1),0);
	Vector3d solB(sol(0) - b(0), sol(1) - b(1),0);
	Vector3d solC(sol(0) - c(0), sol(1) - c(1),0);

	int signA = ab.cross(solA)(2) > 0 ? 1 : 0;
	int signB = bc.cross(solB)(2) > 0 ? 1 : 0;
	int signC = ca.cross(solC)(2) > 0 ? 1 : 0;

	if(signA == signB && signA == signC && signB == signC && signC == signA)
	{
		return true;
	}

	return false;
}

void clickDrag()
{
	if(drag && translate && indexSelect != -1)
	{
		double diffX = (xMousePos - xMouseClick);
		double diffY = (yMousePos - yMouseClick);

		// std::cout << diffX << " " << diffY << std::endl;

		// std::cout << "In main loop: " << indexSelect << std::endl;

		Matrix3d transitionMat;
		transitionMat <<  1, 0, diffX, 0, 1, diffY, 0, 0, 1;

		//Update V
		Vector3d aPos(V.col(indexSelect*3)(0), V.col(indexSelect*3)(1), 1);
		Vector3d bPos(V.col((indexSelect*3) + 1)(0),V.col((indexSelect*3) + 1)(1), 1);
		Vector3d cPos(V.col((indexSelect*3) + 2)(0),V.col((indexSelect*3) + 2)(1), 1);

		Vector3d aRes = (transitionMat * origin_a);
		Vector3d bRes = (transitionMat * origin_b);
		Vector3d cRes = (transitionMat * origin_c);

		// std::cout << aRes << bRes << cRes << std::endl << std::endl;

		V.col(indexSelect*3) << aRes(0),aRes(1);
		V.col((indexSelect*3)+1) << bRes(0), bRes(1);
		V.col((indexSelect*3)+2) << cRes(0), cRes(1);

		// curr.A << V.col(indexSelect*3)(0) , V.col(indexSelect*3)(1);
		// curr.B << V.col((indexSelect*3) + 1)(0), V.col((indexSelect*3) + 1)(1);
		// curr.C << V.col((indexSelect * 3) + 2)(0), V.col((indexSelect * 3) + 2)(1);

		triangleList.at(indexSelect).A << aRes(0), aRes(1);
		triangleList.at(indexSelect).B << bRes(0), bRes(1);
		triangleList.at(indexSelect).C << cRes(0), cRes(1);

		triangleList.at(indexSelect).updateCentroid();

		VBO.update(V);
	}
}

void rotateTriangle(int direction)
{
	if(translate && indexSelect != -1)
	{
		double rotDeg = (10*PI/180);
		Matrix3d rotationMat;
		Matrix3d translateOriginMat;
		Matrix3d translateBackMat;

		//Set up translate matrix
		Vector2d centroid = triangleList.at(indexSelect).centroid;

		translateOriginMat << 1,0,-centroid(0),0,1,-centroid(1),0,0,1;
		translateBackMat << 1,0,centroid(0),0,1,centroid(1),0,0,1;

		//Set up rotation matrix
		if(direction == 1)
		{
			// std::cout << "Pressed key for clockwise rotation" << std::endl;
			rotationMat << cos(rotDeg), -sin(rotDeg), 0, sin(rotDeg), cos(rotDeg), 0, 0, 0, 1;
		}
		else
		{
			// std::cout << "Pressed key for counter rotation" << std::endl;
			rotationMat << cos(rotDeg), sin(rotDeg), 0, -sin(rotDeg), cos(rotDeg), 0, 0, 0, 1;
		}
		

		Vector3d aPos(V.col(indexSelect*3)(0), V.col(indexSelect*3)(1), 1);
		Vector3d bPos(V.col((indexSelect*3) + 1)(0),V.col((indexSelect*3) + 1)(1), 1);
		Vector3d cPos(V.col((indexSelect*3) + 2)(0),V.col((indexSelect*3) + 2)(1), 1);

		//Does matrix transformation
		//translateBackMat * rotationMat * 
		Vector3d aRes = (translateBackMat * rotationMat * translateOriginMat * aPos);
		Vector3d bRes = (translateBackMat * rotationMat * translateOriginMat * bPos);
		Vector3d cRes = (translateBackMat * rotationMat * translateOriginMat * cPos);

		// std::cout << "After Translate:\n" << aRes << "\n" << bRes << "\n" << cRes << "\n" << std::endl;
		
		V.col(indexSelect*3) << aRes(0),aRes(1);
		V.col((indexSelect*3)+1) << bRes(0), bRes(1);
		V.col((indexSelect*3)+2) << cRes(0), cRes(1);

		// origin_a << aRes(0), aRes(1), 1;
		// origin_b << bRes(0), bRes(1), 1;
		// origin_c << cRes(0), cRes(1), 1;

		//Change positions in the triangleList
		triangleList.at(indexSelect).A << aRes(0), aRes(1);
		triangleList.at(indexSelect).B << bRes(0), bRes(1);
		triangleList.at(indexSelect).C << cRes(0), cRes(1);

		triangleList.at(indexSelect).updateCentroid();

		VBO.update(V);

	}
}

void scaleTri(int i)
{
	if(translate && indexSelect != -1)
	{
		// Set up matrixes
		Matrix3d scaleMat;
		Matrix3d translateOriginMat;
		Matrix3d translateBackMat;

		Vector2d centroid = triangleList.at(indexSelect).centroid;
		translateOriginMat << 1,0,-centroid(0),0,1,-centroid(1),0,0,1;
		translateBackMat << 1,0,centroid(0),0,1,centroid(1),0,0,1;
		

		if(i == 1)
		{
			//Scale up
			scaleMat << 1.25, 0, 0, 0, 1.25, 0, 0, 0, 1;
		}
		else
		{
			scaleMat << 0.75, 0, 0, 0, 0.75, 0, 0, 0, 1;
		}

		Vector3d aPos(V.col(indexSelect*3)(0), V.col(indexSelect*3)(1), 1);
		Vector3d bPos(V.col((indexSelect*3) + 1)(0),V.col((indexSelect*3) + 1)(1), 1);
		Vector3d cPos(V.col((indexSelect*3) + 2)(0),V.col((indexSelect*3) + 2)(1), 1);

		//Does matrix transformation
		//translateBackMat * rotationMat * 
		Vector3d aRes = (translateBackMat * scaleMat * translateOriginMat * aPos);
		Vector3d bRes = (translateBackMat * scaleMat * translateOriginMat * bPos);
		Vector3d cRes = (translateBackMat * scaleMat * translateOriginMat * cPos);

		// origin_a << aRes(0), aRes(1), 1;
		// origin_b << bRes(0), bRes(1), 1;
		// origin_c << cRes(0), cRes(1), 1;

		V.col(indexSelect*3) << aRes(0),aRes(1);
		V.col((indexSelect*3)+1) << bRes(0), bRes(1);
		V.col((indexSelect*3)+2) << cRes(0), cRes(1);

		//Change positions in the triangleList
		triangleList.at(indexSelect).A << aRes(0), aRes(1);
		triangleList.at(indexSelect).B << bRes(0), bRes(1);
		triangleList.at(indexSelect).C << cRes(0), cRes(1);

		//	Get new centroid
		triangleList.at(indexSelect).updateCentroid();

		VBO.update(V);
	}
}

void deleteTriangle()
{
	int columnNum = -1;
	if(V.cols() == 3)
	{
		// std::cout << "Only one triangle left" << std::endl;
		columnNum = 3;
	}
	else
	{
		columnNum = V.cols()-3;
	}

	MatrixXf temp(2,columnNum);
	temp.col(0) << 0,0;
	temp.col(1) << 0,0;
	temp.col(2) << 0,0;
	
	//Change V
	int vIndex = indexSelect*3;
	int i = 0;
	int j = 0;
	// std::cout << "V index " << vIndex << std::endl;

	// std::cout << V.cols() << std::endl;
	while(i < V.cols())
	{
		if((vIndex) == i || (vIndex) + 1 == i || (vIndex) + 2 == i)
		{
			//Do nothing
			// std::cout << "Delete " << i << std::endl;
			i++;
		}
		else
		{
			// std::cout << "Use: " << i << std::endl;
			temp.col(j) = V.col(i);
			j++;
			i++;
		}	
	}

	// std::cout << temp << std::endl;

	V = temp;
	
	VBO.update(V);

	// std::cout << V.cols() << std::endl;

	//Change triangleList vector

	std::vector<Triangle> triangleListTemp;

	for(int i = 0; i < triangleList.size(); i++)
	{
		if(i == indexSelect)
		{
			//Do nothing
		}
		else
		{
			triangleListTemp.push_back(triangleList.at(i));
			triangleListTemp.at(triangleListTemp.size()-1).id = i;
		}
		
	}

	triangleList = triangleListTemp;

	//Update values
	triangleNum -= 1;
	indexSelect = -1;
	triangleVPos -= 3;
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

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	// Update the position of the first vertex if the keys 1,2, or 3 are pressed
	switch (key) {
		case GLFW_KEY_I:
			if(action == 0)
			{
				if(insert)
				{
					insert = false;
					translate = false;
					color = false;
					std::cout << "Switch out of insert" << std::endl;
				}
				else
				{
					insert = true;
					translate = false;
					color = false;
					std::cout << "Switch into insert" << std::endl;
				}
			}
			break;
		case GLFW_KEY_O:
			if(action == 0)
			{
				if(translate)
				{
					translate = false;
					color = false;
					insert = false;
					std::cout << "Switch out transition" << std::endl;
				}
				else
				{
					translate = true;
					insert = false;
					color = false;
					std::cout << "Switch into transition" << std::endl;
				}
			}
			break;
		case GLFW_KEY_P:
			if(action == 0)
			{
				Vector2d mouse(xMousePos,yMousePos);
				indexSelect = selectTriangle(mouse);

				if(indexSelect != -1)
				{
					deleteTriangle();
				}

			}
			
			// std::cout << "Delete Triangle" << std::endl;
			break;
		case GLFW_KEY_V:	
			if(action == 0)
			{
				std::cout << V << std::endl;

				for(int i = 0; i < triangleList.size(); i++)
				{
					std::cout << triangleList.at(i).id << std::endl;
				}
			}
			break;
		case GLFW_KEY_H:
			if(action == 0)
			{
				rotateTriangle(1);
				
			}
			
			break;
		case GLFW_KEY_J:
			if(action == 0)
			{
				rotateTriangle(2);
			}
			break;
		case GLFW_KEY_X:
			if(action == 0)
			{
				std::cout << V << std::endl;
				// if(indexSelect != -1)
				// 	std::cout << triangleList.at(indexSelect).centroid << std::endl;
			}
			break;
		case GLFW_KEY_K:
			if(action == 0)
			{
				std::cout << "In scale up" << std::endl;
				scaleTri(1);
			}
			break;
		case GLFW_KEY_L:
			if(action == 0)
			{
				std::cout << "In scale down" << std::endl;
				scaleTri(2);
			}
			break;
		case GLFW_KEY_C:
			if(action == 0)
			{
				if(color)
				{
					color = false;
					std::cout << "Out color mode" << std::endl;
					insert = false;
					translate = false;

					colorVertex = -1;
				}
				else
				{
					color = true;
					std::cout << "In color mode" << std::endl;
					// deselectColor();
					insert = false;
					translate = false;
				}
				
				
			}
			break;
		case GLFW_KEY_0:
			if(action == 0)
			{
				changeVertexColor(0.1,0.2,0.3);
			}
			break;
		case GLFW_KEY_1:
			if(action == 0)
			{
				changeVertexColor(0.0,0.1,0.3);
			}
			break;
		case GLFW_KEY_2:
			if(action == 0)
			{
				changeVertexColor(0.1,0.0,0.2);
			}
			break;
		case GLFW_KEY_3:
			if(action == 0)
			{
				changeVertexColor(0.7,0.0,1.0);
			}
			break;
		case GLFW_KEY_4:
			if(action == 0)
			{
				changeVertexColor(0.2,0.1,0.95);
			}
			break;
		case GLFW_KEY_5:
			if(action == 0)
			{
				changeVertexColor(0.7,0.0,1.0);
			}
			break;
		case GLFW_KEY_6:
			if(action == 0)
			{
				changeVertexColor(0.0,1,0.0);
			}
			break;
		case GLFW_KEY_7:
			if(action == 0)
			{
				changeVertexColor(0.5,0.2,0.8);
			}
			break;
		case GLFW_KEY_8:
			if(action == 0)
			{
				changeVertexColor(0.7,0.5,0.2);
			}
			break;
		case GLFW_KEY_9:
			if(action == 0)
			{
				changeVertexColor(0.7,0.1,0.1);
			}
			break;
		default:
			break;
	}

	// Upload the change to the GPU
	VBO.update(V);
}

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

	// Convert screen position to world coordinates
	double xworld = ((xpos/double(width))*2)-1;
	double yworld = (((height-1-ypos)/double(height))*2)-1; // NOTE: y axis is flipped in glfw

	// std::cout << "In Click " << xworld << " " << yworld << std::endl;

	// Update the position of the first vertex if the left button is pressed
	// if (!insert && button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
		
	// 	V.col(0) << xworld, yworld;
	// }

	

	//Add to the index of the triangles
	drawTriangle(xworld, yworld, button, action);

	if(translate)
	{
		Vector2d mouse(xworld, yworld);
		// std::cout << "Click: " << mouse << std::endl;

		indexSelect = selectTriangle(mouse);
		// std::cout << "Select " << indexSelect << std::endl;

		if(indexSelect != -1)
		{
			origin_centroid = triangleList.at(indexSelect).centroid;
			origin_a << V.col(indexSelect*3)(0), V.col(indexSelect*3)(1), 1;
			origin_b << V.col(indexSelect*3 + 1)(0), V.col(indexSelect*3+1)(1), 1;
			origin_c << V.col(indexSelect*3 + 2)(0), V.col(indexSelect*3+2)(1), 1;
		}
	}

	if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if(GLFW_PRESS == action)
		{
			xMouseClick = xworld;
			yMouseClick = yworld;
			// std::cout << xMouseClick << " " << yMouseClick << std::endl;
            drag = true;
		}
        else if(GLFW_RELEASE == action)
		{
            drag = false;
		}


    }

	if(button == GLFW_MOUSE_BUTTON_LEFT)
	{
		if(action == 0)
		{
			findVertex(xworld,yworld);
		}
	}

	// Upload the change to the GPU
	VBO.update(V);
}

int main(void) {

	// Matrix3d test;
	// test << 1,0,2,0,1,2,0,0,1;

	// Vector3d a(0,0,1);

	// Vector3d res = test*a;
	// std::cout << res << std::endl;

	// exit(0);

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
	GLFWwindow * window = glfwCreateWindow(640, 480, "[Float] Hello World", NULL, NULL);
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

	// Initialize the VAO
	// A Vertex Array Object (or VAO) is an object that describes how the vertex
	// attributes are stored in a Vertex Buffer Object (or VBO). This means that
	// the VAO is not the actual object storing the vertex data,
	// but the descriptor of the vertex data.
	VertexArrayObject VAO;
	VAO.init();
	VAO.bind();

	// Initialize the VBO with the vertices data
	// A VBO is a data container that lives in the GPU memory
	VBO.init();

	// Initialize the VBO lines
	// line_VBO.init();

	//VBO matrix that lives in GPU.
	//VAO is pipe between the CPU and GPU

	V.resize(2,3);
	// V << 0,  0.5, -0.5, 0.5, -0.5, -0.5;
	V << 0,  0, 0, 0, 0, 0;

	VBO.update(V);

		VBO_color.init();
		C.resize(3,3);
		C << 1, 1, 1,
			 1, 1, 1,
			 1, 1, 1;


		VBO_color.update(C);

	// Initialize the OpenGL Program
	// A program controls the OpenGL pipeline and it must contains
	// at least a vertex shader and a fragment shader to be valid
	Program program;

	// in vec3 color;
	// out vec3 o_color;
	const GLchar* vertex_shader = R"(
		#version 150 core

		in vec2 position;
		// uniform mat4 view;
		in vec3 color;

		out vec3 Color;

		void main() {
			Color = color;
			gl_Position = vec4(position, 0.0, 1.0);
		}
	)";

	// uniform mat4 view

	//in vec3 o_color;
	//o_color = color;

	const GLchar* fragment_shader = R"(
		#version 150 core

		uniform vec3 triangleColor;
		in vec3 Color;

		out vec4 outColor;

		void main() {
		   outColor = vec4(Color, 1.0);
		}
	)";

	//triangleColor

	// uniform float sizex;
	// uniform float sizey;

	//Using gl_FragCoord 

	// This can change the color
	// outColor = vec4(triangleColor, 1.0);
	// outColor = vec4(gl_FragCoord.x/sizex, gl_FragCoord.y/sizey, 0.0, 1.0);

	// Compile the two shaders and upload the binary to the GPU
	// Note that we have to explicitly specify that the output "slot" called outColor
	// is the one that we want in the fragment buffer (and thus on screen)
	program.init(vertex_shader, fragment_shader, "outColor");
	program.bind();

	// program.bindVertexAttribArray("color", VBO_color);

	// Save the current time --- it will be used to dynamically change the triangle color
	auto t_start = std::chrono::high_resolution_clock::now();

	// Register the keyboard callback
	glfwSetKeyCallback(window, key_callback);

	// Register the mouse callback
	glfwSetMouseButtonCallback(window, mouse_button_callback);

	// Register the position of the mouse when it is moving
    glfwSetCursorPosCallback(window, cursor_position_callback);

	// Loop until the user closes the window
	while (!glfwWindowShouldClose(window)) {

		// Set the size of the viewport (canvas) to the size of the application window (framebuffer)
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		glViewport(0, 0, width, height);

		// Bind your VAO (not necessary if you have only one)
		VAO.bind();

		// Bind your program
		program.bind();

		// The vertex shader wants the position of the vertices as an input.
		// The following line connects the VBO we defined above with the position "slot"
		// in the vertex shader
		program.bindVertexAttribArray("position", VBO);
		program.bindVertexAttribArray("color", VBO_color);

		// Set the uniform value depending on the time difference
		auto t_now = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration_cast<std::chrono::duration<float>>(t_now - t_start).count();
		glUniform3f(program.uniform("triangleColor"), (float)(sin(time * 4.0f) + 1.0f) / 2.0f, 0.0f, 0.0f);

		//Change values at the end to modify the colors

		// glUniform1f(program.uniform("sizex", width);
		// glUniform1f(program.uniform("sizey", height);

		// Clear the framebuffer
		glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

				// Resize the view
				// float ar = float(height)/float(width);
				// // std::cout << ar << std::endl;

				// //Construct transformation matrix
				// Matrix4f view;
				// view << ar, 0, 0, 0,
				// 		0, ar, 0, 0,
				// 		0, 0, 1, 0,
				// 		0, 0, 0, 1;

				// //view.data() does pointer to where matrix is
				// glUniformMatrix4fv(program.uniform("view"), 1, GL_FALSE, view.data());

		// std::cerr << view << std::endl << std::endl;

		// Draw a triangle
		// This get's constantly called

		// glUniform3f(program.uniform("triangleColor"), 0, 0, 0);

		//Selects the color and draws the triangle
		selectColor(program, time);

		glDrawArrays(GL_TRIANGLES, 0, triangleNum * 3);

		// Draw border for triangle
		glDrawArrays(GL_LINE_LOOP, pointOneIndex, 3);

		// Draw lines
		drawLines();

		// Click and drag shape
		clickDrag();

		// Rotate 
		// rotateTriangle(0);

		//Select vertex for color
		selectVertex(program);

		// Swap front and back buffers
		glfwSwapBuffers(window);

		// Poll for and process events
		glfwPollEvents();

		// exit(0);
	}

	// Deallocate opengl memory
	program.free();
	VAO.free();
	VBO.free();

	// Deallocate glfw internals
	glfwTerminate();
	return 0;
}

