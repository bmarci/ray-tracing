//=============================================================================================
// Framework for the ray tracing homework
// ---------------------------------------------------------------------------------------------
// Author    : Marci Blum
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;
const int MAX_DEPTH = 3;
const bool SMOOTH_MATERIAL = true;
const bool ROUGH_MATERIAL = false;
const bool REFLECTIVE = true;
const bool NOT_REFLECTIVE = false;
const bool REFRACTIVE = true;
const bool NOT_REFRACTIVE = false;
const float TRASHOLD = 0.0001f;


// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

struct vec3 {
    float x, y, z;

    vec3(float x0 = 0, float y0 = 0, float z0 = 0) { x = x0; y = y0; z = z0; }

    vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }

    vec3 operator/(float a) const { return vec3(x / a, y / a, z / a); }

    vec3 operator+(const vec3& v) const {
        return vec3(x + v.x, y + v.y, z + v.z);
    }
    vec3 operator-(const vec3& v) const {
        return vec3(x - v.x, y - v.y, z - v.z);
    }
    vec3 operator*(const vec3& v) const {
        return vec3(x * v.x, y * v.y, z * v.z);
    }
    vec3 operator-() const {
        return vec3(-x, -y, -z);
    }
    vec3 normalize() const {
        return (*this) * (1 / (Length() + 0.000001));
    }
    float Length() const { return sqrtf(x * x + y * y + z * z); }

    operator float*() { return &x; }
};

float dot(const vec3& v1, const vec3& v2) {
    return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}

vec3 cross(const vec3& v1, const vec3& v2) {
    return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

vec3 normalColor(const vec3& color){
    return color / 255.f;
}

float signum(float x){
    return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
}


// row-major matrix 4x4
struct mat4 {
    float m[4][4];
public:
    mat4() {}
    mat4(float m00, float m01, float m02, float m03,
         float m10, float m11, float m12, float m13,
         float m20, float m21, float m22, float m23,
         float m30, float m31, float m32, float m33) {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
        m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
    }

    mat4 operator*(const mat4& right) {
        mat4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
            }
        }
        return result;
    }
    operator float*() { return &m[0][0]; }
};

mat4 Translate(float tx, float ty, float tz) {
    return mat4(1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                tx, ty, tz, 1);
}


void getErrorInfo(unsigned int handle) {
    int logLen;
    glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
    if (logLen > 0) {
        char * log = new char[logLen];
        int written;
        glGetShaderInfoLog(handle, logLen, &written, log);
        printf("Shader log:\n%s", log);
        delete log;
    }
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
    int OK;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
    if (!OK) {
        printf("%s!\n", message);
        getErrorInfo(shader);
    }
}

// check if shader could be linked
void checkLinking(unsigned int program) {
    int OK;
    glGetProgramiv(program, GL_LINK_STATUS, &OK);
    if (!OK) {
        printf("Failed to link shader program!\n");
        getErrorInfo(program);
    }
}

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0

	out vec2 texcoord;

	void main() {
		texcoord = (vertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates

	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

vec3 background[windowWidth * windowHeight];	// The image, which stores the ray tracing result

struct vec4 {
    float v[4];

    vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
        v[0] = x; v[1] = y; v[2] = z; v[3] = w;
    }
};

class Material {
    bool smooth;
    bool reflective;
    bool refractive;
public:
    Material(bool isSmooth, bool aReflective, bool aRefractive){
        smooth = isSmooth;
        reflective = aReflective;
        refractive = aRefractive;
    }
    vec3 F0;
    float n;
    vec3 ka, kd, ks;
    float shininess;
    bool isSmooth(){
        return smooth;
    }
    bool isReflective(){
        return reflective;
    }
    bool isRefractive(){
        return refractive;
    }
    virtual vec3 reflect(vec3 direction, vec3 normal){
        return vec3(0,0,0); //ugly hack
    }
    virtual vec3 refract(vec3 direction, vec3 normal){
        return vec3(0,0,0); //ugly hack
    }
    virtual vec3 fresnel(vec3 direction, vec3 normal){
        return vec3(0,0,0);
    }
    virtual vec3 shade(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 inRad){
        return vec3(0,0,0);
    }
};

class SmoothMaterial: public Material{
    vec3 calculateF0(const vec3& n, const vec3& k){
        vec3 tmpF0;
        tmpF0.x = (pow((n.x - 1),2) + pow(k.x, 2)) / (pow((n.x+1), 2) + pow(k.x, 2));
        tmpF0.y = (pow((n.y - 1),2) + pow(k.y, 2)) / (pow((n.y+1), 2) + pow(k.y, 2));
        tmpF0.z = (pow((n.z - 1),2) + pow(k.z, 2)) / (pow((n.z+1), 2) + pow(k.z, 2));
        return tmpF0;
    }
public:
    SmoothMaterial(vec3 aN, vec3 aK, bool aReflective, bool aRefractive): Material(SMOOTH_MATERIAL, aReflective, aRefractive){
        n = aN.x;
        F0 = calculateF0(aN, aK);
    }
    vec3 reflect(vec3 direction, vec3 normal){
        return direction - normal * dot(normal, direction) * 2.0f;
    }
    vec3 refract(vec3 direction, vec3 normal){
        direction = direction.normalize();
        normal = normal.normalize();
        float ior = n; //toresmutato
        float cosa = -dot(normal, direction);
        if(cosa < 0.f){
            cosa = -cosa;
            normal = normal * -1.f;
            ior = 1.f/n;
        }
        float disc = 1.f - (1.f - pow(cosa, 2)) / pow(ior,2);
        if(disc < 0.f){
            return reflect(direction, normal);
        }
        vec3 ret = direction / ior + normal * (cosa/ior - sqrt(disc));
        return (direction / ior) + (normal * (cosa/ior - sqrt(disc)));
    }
    vec3 fresnel(vec3 direction, vec3 normal){
        float cosa = fabs(dot(normal, direction));
        return F0 + (vec3(1, 1, 1) - F0) * pow(1.f-cosa, 5);
    }
};

class RoughMaterial: public Material{
public:
    RoughMaterial(vec3 aKa, vec3 aKd, vec3 aKs, float aShininess):Material(ROUGH_MATERIAL, NOT_REFLECTIVE, NOT_REFRACTIVE){
        ka = aKa;
        kd = aKd;
        ks = aKs;
        shininess = aShininess;
    }
    vec3 shade(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 inRad){
        vec3 reflRad(0, 0, 0);
        float cosTheta = dot(normal, lightDir);
        if(cosTheta < 0){
            return reflRad;
        }
        reflRad = inRad * kd * cosTheta;
        vec3 halfway = (viewDir + lightDir).normalize();
        float cosDelta = dot(normal, halfway);
        if(cosDelta < 0 ){
            return reflRad;
        }
        vec3 newColor = reflRad + inRad * ks * pow(cosDelta, shininess);
        return reflRad + inRad * ks * pow(cosDelta, shininess);
    }
};

struct Hit {
    float t;
    vec3 position;
    vec3 normal;
    Material* material;
    Hit() {
        t = -1;
    }
};

struct Ray {
public:
    vec3 p0;
    vec3 v;
    Ray(vec3 aP0, vec3 aV) {
        p0 = aP0;
        v = aV;
    }
};

struct Camera {
public:
    vec3 wEye, wLookat, wVup, wVright; //wEye: (0, 0, 3), wLookat(0,0,0), wVup(0,1,0)
    Camera() { //default values
        wEye = vec3(0, 0, 0);
        wLookat = vec3(0, 0, 1);
        wVup = vec3(0, 3, 0);
        wVright = vec3(3, 0, 0);
    }
    Ray getRay(int x, int y) {
        vec3 p = wLookat + wVright * (2*x / (float)windowWidth - 1)  + wVup * (2*y / (float)windowHeight - 1);
        vec3 r = (p - wEye).normalize();
        return Ray(wEye, r);
    }
};

struct Light {
    vec3 p0;
    vec3 color;
    Light(vec3 aP0, vec3 aColor) {
        p0 = aP0;
        color = aColor;
    }
    vec3 getLightDir(vec3 p){
        return p - p0;
    }
    vec3 getP0(){
        return p0;
    }
    vec3 getColor(){
        return color;
    }
};

class Intersectable {
    Material* material;
public:
    Intersectable(Material* aMaterial){
        material = aMaterial;
    }
    Material* getMaterial(){
        return material;
    }
    virtual Hit intersect(const Ray& ray) = 0;
    ~Intersectable(){
        delete material;
    }
};

class Cylinder : public Intersectable {
    vec3 center;
    vec3 v;
    float radius;
public:
    Cylinder(vec3 aCenter, vec3 aV, float aRadius, Material* aMaterial):Intersectable(aMaterial){
        center = aCenter;
        v = aV;
        radius = aRadius;
    }
    virtual Hit intersect(const Ray& ray) {
        Hit hit = Hit(); //TODO
        vec3 deltaP = ray.p0 - center;
        vec3 elso = ray.v - v*(dot(ray.v, v));
        vec3 masodik = deltaP - v*(dot(deltaP, v));
        vec3 elsoDeriv = vec3(1, 1, 1) - (v * v);
        vec3 masodikDeriv = vec3(1, 1, 1) - (v * v);

        float a = dot(elso, elso);
        float b = 2 * (dot(elso, masodik));
        float c = dot(masodik, masodik) - pow(radius, 2);
        float d = pow(b, 2) - 4 * a * c;
        if(d < 0){ // nem talalta el
            return hit;
        }
        else{
            float t1 = (-b + sqrt(d)) / (2 * a );
            if(d > 0){ //minimum kell
                float t2 = (-b - sqrt(d)) / (2 * a);
                (t2 < t1 && t2 > 0) ? t1 = t2 : t1=t1;
                if(t1 < 0){
                    return hit;
                }

            }
            hit.t = t1;
        }
        vec3 deriv = (elso * elsoDeriv) * 2 * pow(hit.t, 2) + ((elsoDeriv * masodik) + (masodikDeriv * elso)) * 2 * hit.t + (masodik * masodikDeriv) * 2;
        hit.position = ray.p0 + ray.v * hit.t;
        hit.normal = deriv.normalize();
        hit.material = getMaterial();
        return hit;
    }
};

class Round : public Intersectable {
    vec3 center;
    float radius;
public:
    Round(vec3 aCenter, float aRadius, Material* aMaterial): Intersectable(aMaterial) {
        center = aCenter;
        radius = aRadius;
    }
    virtual Hit intersect(const Ray& ray){
        Hit hit = Hit();
        float cx = ray.p0.x - center.x;
        float cy = ray.p0.y - center.y;
        float cz = ray.p0.z - center.z;
        float a = pow(ray.v.x, 2) + pow(ray.v.y, 2) + pow(ray.v.z, 2);
        float b = 2 * (ray.v.x * cx + ray.v.y * cy + ray.v.z * cz);
        float c = pow(cx, 2) + pow(cy, 2) + pow(cz, 2) - pow(radius, 2);
        float d = pow(b, 2) - 4 * a * c;
        if(d < 0){ // nem talalta el
            return hit;
        }
        else{
            float t1 = (-b + sqrt(d)) / (2 * a );
            if(d > 0){ //minimum kell
                float t2 = (-b - sqrt(d)) / (2 * a);
                (t2 < t1 && t2 > 0.0) ? t1 = t2 : t1=t1;
                if(t1 < 0.0){
                    return hit; //csak pozitivakat nezek
                }

            }
            hit.t = t1;
        }
        hit.position = ray.p0 + ray.v * hit.t;
        hit.normal = (hit.position - center).normalize();
        hit.material = getMaterial();
        return hit;
    }
};

class Surface: public Intersectable{
    vec3 p0;
    vec3 normal;
public:
    Surface(vec3 aP0, vec3 aNormal, Material* aMaterial): Intersectable(aMaterial){
        p0 = aP0;
        normal = aNormal;
    }
    virtual Hit intersect(const Ray& ray){
        Hit hit = Hit();
        float a = normal.x;
        float b = normal.y;
        float c = normal.z;
        float dx = ray.p0.x - p0.x;
        float dy = ray.p0.y - p0.y;
        float dz = ray.p0.z - p0.z;
        float vx = ray.v.x;
        float vy = ray.v.y;
        float vz = ray.v.z;
        float t = -(a * dx + b * dy + c * dz) / (a * vx + b * vy + c * vz);
        if(t < 0){
            return hit;
        }
        hit.t = t;
        vec3 deriv = (normal * t + normal).normalize();
        hit.position = ray.p0 + ray.v * hit.t;
        hit.normal = deriv;
        hit.material = getMaterial();
        return hit;
    }
};

class Scene {
    static const int NUMBER_OF_OBJECTS = 10;
    static const int NUMBER_OF_LIGHTS = 10;
    int sceneObjectLength = 0;
    int sceneLightLength = 0;
    vec3 La;
    Camera cam;
    Light* lights[NUMBER_OF_LIGHTS];
    Intersectable* sceneObjects[NUMBER_OF_OBJECTS];

public:
    void build() {
        La = vec3(0.8f, 0.8f, 0.8f);
        vec3 n = vec3(0.17, 0.35, 1.5);
        vec3 k = vec3(3.1, 2.7, 1.9);
        SmoothMaterial* gold = new SmoothMaterial(n, k, REFLECTIVE, NOT_REFRACTIVE);
        Cylinder *henger = new Cylinder(vec3(0, 0, -3.f), vec3(0, 0, 1).normalize(), 3.f, gold);
        sceneObjects[sceneObjectLength++] = henger;

        vec3 kaTop = vec3(0.5,0.3,0.4);
        vec3 kdTop = vec3(0, 1, 0);
        vec3 ksTop = vec3(0.5, 0.5, 0.5);
        RoughMaterial* topMaterial = new RoughMaterial(kaTop, kdTop, ksTop, 5.f);
        vec3 topP0 = vec3(0, 2.5f, 0);
        vec3 topN = vec3(0, 1, 0);
        Surface* top = new Surface(topP0, topN, topMaterial);
        sceneObjects[sceneObjectLength++] = top;

        vec3 kaBottom = vec3(0.6, 0.5, 0.5);
        vec3 kdBottom = vec3(0, 1, 1);
        vec3 ksBottom = vec3(1, 1, 1);
        RoughMaterial* bottomMaterial = new RoughMaterial(kaBottom, kdBottom, ksBottom, 5.f);
        vec3 bottomP0 = vec3(0, -2.5f, 0);
        vec3 bottomN = vec3(0, 1, 0);
        Surface* bottom = new Surface(bottomP0, bottomN, bottomMaterial);
        sceneObjects[sceneObjectLength++] = bottom;


        n = vec3(0.14, 0.16, 0.13);
        k = vec3(4.1, 2.3, 3.1);
        SmoothMaterial* silver = new SmoothMaterial(n, k, REFLECTIVE, NOT_REFRACTIVE);
        Cylinder *hengerKetto = new Cylinder(vec3(-1.f, 0.1f, 3.f), vec3(0.16, 1.f, 0.1).normalize(), 0.5f, silver);
        sceneObjects[sceneObjectLength++] = hengerKetto;



        n = vec3(1.5, 1.5, 1.5);
        k = vec3(0,0,0);
        SmoothMaterial* glass = new SmoothMaterial(n, k, REFLECTIVE, REFRACTIVE);
        Cylinder *hengerHarom = new Cylinder(vec3(-1.f,-1.f, 0.3f), vec3(-0.2, 1, 0).normalize(), 0.5f, glass);
        sceneObjects[sceneObjectLength++] = hengerHarom;

/*			n = vec3(1.33f, 1.33f, 1.33f);
			k = vec3(0,0,0);
			SmoothMaterial* beer = new SmoothMaterial(n, k, REFLECTIVE, REFRACTIVE);
			Cylinder *hengerNegy = new Cylinder(vec3(1.5f,-1.f,2.f), vec3(-0.2, 0.8f, 0.2).normalize(), 0.8f, beer);
			sceneObjects[sceneObjectLength++] = hengerNegy;*/

        n = vec3(0.17, 0.35, 1.5);
        k = vec3(3.1, 2.7, 1.9);
        SmoothMaterial* gold2 = new SmoothMaterial(n, k, REFLECTIVE, NOT_REFRACTIVE);
        Round *gombBub = new Round(vec3(1.f, 0,2.f), 0.3f, gold2);
        sceneObjects[sceneObjectLength++] = gombBub;

        n = vec3(0.14, 0.16, 0.13);
        k = vec3(4.1, 2.3, 3.1);
        SmoothMaterial* silver2 = new SmoothMaterial(n, k, REFLECTIVE, NOT_REFRACTIVE);
        Round *gombBub2 = new Round(vec3(-1.3f,1.f,2.f), 0.3f, silver2);
        sceneObjects[sceneObjectLength++] = gombBub2;

        /*vec3 ka = vec3 (0.8, 0.5, 1);
        vec3 kd = vec3(0.1f, 1.f, 0.5f);
        vec3 ks = vec3(1,1,1);
        RoughMaterial* difspec = new RoughMaterial(ka, kd, ks, 10.f);
        Cylinder *hengerOt = new Cylinder(vec3(1.f, -1.f, 3.f), vec3(1, 0, 0.5).normalize(), 0.5f, difspec);
        sceneObjects[sceneObjectLength++] = hengerOt;*/

        vec3 lightOneP0 = vec3(-1.5f,2.f, 2.f);
        vec3 lightOneColor = vec3(1, 0, 0);
        Light *lightOne = new Light(lightOneP0, lightOneColor);
        lights[sceneLightLength++] = lightOne;

        vec3 lightTwoP0 = vec3(1.5f, 2.f, 2.f);
        vec3 lightTwoColor = vec3(0, 1, 0);
        Light *lightTwo = new Light(lightTwoP0, lightTwoColor);
        lights[sceneLightLength++] = lightTwo;
/*
			vec3 lightThreeP0 = vec3(-3,2,8.f);
			vec3 lightThreeColor = vec3(0, 0, 1);
			Light *lightThree = new Light(lightThreeP0, lightThreeColor);
			lights[sceneLightLength++] = lightThree; */
    }
    void render() {
        for (int x = 0; x < windowWidth; x++) {
            for (int y = 0; y < windowHeight; y++) {
                Ray r = cam.getRay(x,y);
                vec3 color = trace(r, 0);
                background[y * windowWidth + x] = color;
            }
        }
    }
    Hit firstIntersect(const Ray& ray) {
        Hit bestHit;
        Hit tryHit;
        for (int i = 0; i < sceneObjectLength; i++) {
            tryHit = sceneObjects[i]->intersect(ray);
            if(tryHit.t > 0 && ((bestHit.t < 0) || (tryHit.t < bestHit.t))){
                bestHit = tryHit;
            }
        }
        return bestHit;
    }

    vec3 trace(const Ray& ray, int depth) {
        vec3 outRadiance = vec3(0,0,0);
        if(depth > MAX_DEPTH){
            return La;
        }
        Hit hit = firstIntersect(ray);
        if(hit.t < 0){
            return La;
        }

        if(!hit.material->isSmooth()){
            outRadiance = hit.material->ka * La;
            for(int i = 0; i < sceneLightLength; i++){
                vec3 hitPosition = hit.position + hit.normal * TRASHOLD * signum(dot(hit.normal, ray.v));
                vec3 lightDir = lights[i]->getLightDir(hitPosition).normalize();
                vec3 lightColor = lights[i]->getColor();
                Ray shadowRay = Ray(hitPosition, lightDir);
                Hit shadowHit = firstIntersect(shadowRay);
                if(!((shadowHit.t) < 0 || (shadowHit.t > (hitPosition - lights[i]->getP0()).Length()))){
                    outRadiance = outRadiance + hit.material->shade(hit.normal, ray.v, lightDir, lightColor);
                }
            }
        }

        if(hit.material->isReflective()){
            vec3 reflectionDir = hit.material->reflect(ray.v, hit.normal).normalize();
            Ray reflectedRay(hit.position + (hit.normal * TRASHOLD * signum(dot(hit.normal, ray.v))), reflectionDir);
            outRadiance = outRadiance + trace(reflectedRay, depth+1) * hit.material->fresnel(ray.v, hit.normal);
        }
        if(hit.material->isRefractive()){
            vec3 refractionDir = hit.material->refract(ray.v, hit.normal).normalize();
            Ray refractedRay(hit.position - hit.normal * TRASHOLD * signum(dot(hit.normal, ray.v)), refractionDir);
            outRadiance = outRadiance + trace(refractedRay, depth+1) * (vec3(1,1,1) - hit.material->fresnel(ray.v, hit.normal));
        }
        return outRadiance;
    }
    ~Scene(){
        //objects
        for(int i = 0; i < sceneObjectLength; i++){
            delete sceneObjects[i];
        }
        //lights
        for(int i = 0; i < sceneLightLength; i++){
            delete lights[i];
        }
    }
};

// handle of the shader program
unsigned int shaderProgram;

class FullScreenTexturedQuad {
    unsigned int vao, textureId;	// vertex array object id and texture id
public:
    void Create(vec3 image[windowWidth * windowHeight]) {
        glGenVertexArrays(1, &vao);	// create 1 vertex array object
        glBindVertexArray(vao);		// make it active

        unsigned int vbo;		// vertex buffer objects
        glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

        // vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        static float vertexCoords[] = { -1, -1, 1, -1, -1, 1,
                                        1, -1, 1, 1, -1, 1 };	// two triangles forming a quad
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
        // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

        // Create objects by setting up their vertex data on the GPU
        glGenTextures(1, &textureId);  				// id generation
        glBindTexture(GL_TEXTURE_2D, textureId);    // binding

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGB, GL_FLOAT, image); // To GPU
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // sampling
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }

    void Draw() {
        glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
        int location = glGetUniformLocation(shaderProgram, "textureUnit");
        if (location >= 0) {
            glUniform1i(location, 0);		// texture sampling unit is TEXTURE0
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, textureId);	// connect the texture to the sampler
        }
        glDrawArrays(GL_TRIANGLES, 0, 6);	// draw two triangles forming a quad
    }
};

// The virtual world: single quad
FullScreenTexturedQuad fullScreenTexturedQuad;


// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    Scene scene = Scene();
    // Ray tracing fills the image called background
    scene.build();
    scene.render();

    fullScreenTexturedQuad.Create(background);

    // Create vertex shader from string
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    if (!vertexShader) {
        printf("Error in vertex shader creation\n");
        exit(1);
    }
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);
    checkShader(vertexShader, "Vertex shader error");

    // Create fragment shader from string
    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    if (!fragmentShader) {
        printf("Error in fragment shader creation\n");
        exit(1);
    }
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);
    checkShader(fragmentShader, "Fragment shader error");

    // Attach shaders to a single program
    shaderProgram = glCreateProgram();
    if (!shaderProgram) {
        printf("Error in shader program creation\n");
        exit(1);
    }
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);

    // Connect the fragmentColor to the frame buffer memory
    glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

    // program packaging
    glLinkProgram(shaderProgram);
    checkLinking(shaderProgram);
    // make this program run
    glUseProgram(shaderProgram);
}

void onExit() {
    glDeleteProgram(shaderProgram);
    printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
    fullScreenTexturedQuad.Draw();
    glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
    }
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
}

int main(int argc, char * argv[]) {
    glutInit(&argc, argv);
#if !defined(__APPLE__)
    glutInitContextVersion(majorVersion, minorVersion);
#endif
    glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
    glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
    glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
    glewExperimental = true;	// magic
    glewInit();
#endif

    printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
    printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
    printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
    glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
    glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
    printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
    printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

    onInitialization();

    glutDisplayFunc(onDisplay);                // Register event handlers
    glutMouseFunc(onMouse);
    glutIdleFunc(onIdle);
    glutKeyboardFunc(onKeyboard);
    glutKeyboardUpFunc(onKeyboardUp);
    glutMotionFunc(onMouseMotion);

    glutMainLoop();
    onExit();
    return 1;
}
