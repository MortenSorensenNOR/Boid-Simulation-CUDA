#include <vector>
#include <chrono>
#include <cassert>
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"

#define OLC_PGEX_TRANSFORMEDVIEW
#include "olcPGEX_TransformedView.h"
using namespace std;
using namespace std::chrono;

constexpr float PI = 3.14159265358979;
constexpr float WORLDWIDTH = 3500;  // 3500
constexpr float WORLDHEIGHT = 3500; // 3500

constexpr int N = 1 << 14;
constexpr size_t bytes = sizeof(float2) * N;

float PERCEPTION_RADIUS = 225.0f;
float MAX_FORCE = 15.0f;
float MAX_VEL = 75.0f;

__global__ void updateVel(float2* vel, float2* acc, float dt, int n)
{
    int tID = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tID < n)
    {
        vel[tID].x += acc[tID].x * dt;
        vel[tID].y += acc[tID].y * dt;
    }
}

__global__ void updatePos(float2* pos, float2* vel, float dt, int n)
{
    int tID = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tID < n)
    {
        pos[tID].x += vel[tID].x * dt;
        pos[tID].y += vel[tID].y * dt;

        pos[tID].x = fmod(pos[tID].x + WORLDWIDTH, WORLDWIDTH);
        pos[tID].y = fmod(pos[tID].y + WORLDHEIGHT, WORLDHEIGHT);
    }
}

__global__ void calcSteer(float2* pos, float2* vel, float2* acc, int n, float PERCEPTION_RADIUS, float MAX_FORCE, float MAX_VEL)
{
    int tID = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tID < n)
    {
        int total = 0;
        float2 align{ 0.0f, 0.0f };
        float2 cohesion{ 0.0f, 0.0f };
        float2 seperation{ 0.0f, 0.0f };

        float distance = 0.0f;
        for (int i = 0; i < n; i++)
        {
            if (tID != i)
            {
                distance = (pos[tID].x - pos[i].x) * (pos[tID].x - pos[i].x) + (pos[tID].y - pos[i].y) * (pos[tID].y - pos[i].y);
                if (distance < PERCEPTION_RADIUS * PERCEPTION_RADIUS)
                {
                    float2 diff = { (pos[tID].x - pos[i].x) / distance, (pos[tID].y - pos[i].y) / distance };
                    align.x += vel[i].x;
                    align.y += vel[i].y;

                    cohesion.x += pos[i].x;
                    cohesion.y += pos[i].y;

                    seperation.x += diff.x;
                    seperation.y += diff.y;
                    total++;
                }
            }
        }

        if (total)
        {
            align.x /= total;
            align.y /= total;

            cohesion.x /= total;
            cohesion.y /= total;

            seperation.x /= total;
            seperation.y /= total;

            // Align
            float length_align = sqrtf(align.x * align.x + align.y * align.y);
            align.x = align.x / length_align * MAX_VEL;
            align.y = align.y / length_align * MAX_VEL;

            align.x -= vel[tID].x;
            align.y -= vel[tID].y;

            length_align = sqrtf(align.x * align.x + align.y * align.y);
            if (length_align > MAX_FORCE)
            {
                align.x = align.x / length_align * MAX_FORCE;
                align.y = align.y / length_align * MAX_FORCE;
            }

            // Cohesion
            cohesion.x -= pos[tID].x;
            cohesion.y -= pos[tID].y;

            float length_cohesion = sqrtf(cohesion.x * cohesion.x + cohesion.y * cohesion.y);
            cohesion.x = cohesion.x / length_cohesion * MAX_VEL;
            cohesion.y = cohesion.y / length_cohesion * MAX_VEL;

            cohesion.x -= vel[tID].x;
            cohesion.y -= vel[tID].y;
            length_cohesion = sqrtf(cohesion.x * cohesion.x + cohesion.y * cohesion.y);
            if (length_cohesion > MAX_FORCE)
            {
                cohesion.x = cohesion.x / length_cohesion * MAX_FORCE;
                cohesion.y = cohesion.y / length_cohesion * MAX_FORCE;
            }

            // Seperation
            float length_seperation = sqrtf(seperation.x * seperation.x + seperation.y * seperation.y);
            seperation.x = seperation.x / length_seperation * MAX_VEL;
            seperation.y = seperation.y / length_seperation * MAX_VEL;

            seperation.x -= vel[tID].x;
            seperation.y -= vel[tID].y;

            length_seperation = sqrtf(seperation.x * seperation.x + seperation.y * seperation.y);
            if (length_seperation > MAX_FORCE)
            {
                seperation.x = seperation.x / length_seperation * MAX_FORCE;
                seperation.y = seperation.y / length_seperation * MAX_FORCE;
            }
        }

        acc[tID] = align;
        acc[tID].x += cohesion.x;
        acc[tID].y += cohesion.y;
        acc[tID].x += seperation.x;
        acc[tID].y += seperation.y;
    }

}

__global__ void avoidBoundaries(float2* pos, float2* vel, float dt, int n)
{
    int tID = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tID < n)
    {
        float steerAngle = 0.0;
        float boidRelDirection;
        float length = sqrtf(vel[tID].x * vel[tID].x + vel[tID].y * vel[tID].y);
        float2 velNorm = { vel[tID].x / length, vel[tID].y / length };

        boidRelDirection = velNorm.y;
        if (boidRelDirection > 0.0)
        {
            if ((WORLDHEIGHT - pos[tID].y) < 100.0)
            {
                steerAngle -= ((90.0f * 3.14 / 180.0f) - acos(boidRelDirection)) / 10.0f * dt;
            }
        }

        boidRelDirection = velNorm.x;
        if (boidRelDirection > 0.0)
        {
            if ((WORLDWIDTH - pos[tID].x) < 100.0)
            {
                steerAngle -= ((90.0f * 3.14 / 180.0f) - acos(boidRelDirection)) / 10.0f * dt;
            }
        }

        boidRelDirection = -1*velNorm.x;
        //boid in moving towards it, if dot product is greater than 0
        if (boidRelDirection > 0.0)
        {
            //check is the boid is coming close
            if (pos[tID].x < 100.0)
            {
                //steer the boid away from the boundary
                steerAngle -= ((90 * 3.14 / 180) - acos(boidRelDirection)) / 10 * dt;
            }
        }

        boidRelDirection = -1*velNorm.y;
        //boid in moving towards it, if dot product is greater than 0
        if (boidRelDirection > 0.0)
        {
            //check is the boid is coming close
            if (pos[tID].y < 100.0)
            {
                //steer the boid away from the boundary
                steerAngle -= ((90 * 3.14 / 180) - acos(boidRelDirection)) / 10 * dt;
            }
        }

        float2 result;
        result.x = vel[tID].x * cos(steerAngle) + vel[tID].y * sin(steerAngle);
        result.y = -1 * vel[tID].x * sin(steerAngle) + vel[tID].y * cos(steerAngle);
        vel[tID] = result;
    }
}

float dot(float2 vec1, float2 vec2)
{
    return (vec1.x * vec2.x + vec1.y * vec2.y);
}

float2 normalize(float2 a)
{
    float length = sqrtf(a.x * a.x + a.y * a.y);
    return { a.x / length, a.y / length };
}

class Simulation : public olc::PixelGameEngine
{
public:
    Simulation()
    {
        sAppName = "Simulation";
    }

    ~Simulation()
    {
        free(&pos);
        free(&vel);
        free(&acc);

        cudaFree(d_pos);
        cudaFree(d_vel);
        cudaFree(d_acc);
    }

protected:
    float sWidth;
    float sHeight;

    olc::TransformedView tv;

    std::vector<float2> pos;
    std::vector<float2> vel;
    std::vector<float2> acc;

    int NUM_THREADS;
    int NUM_BLOCKS;

    float2* d_pos, * d_vel, * d_acc;
public:
    bool OnUserCreate() override
    {
        // Called once at the start, so create things here
        sWidth = float(ScreenWidth());
        sHeight = float(ScreenHeight());

        tv.Initialise({ int(WORLDWIDTH), int(WORLDHEIGHT) });
        tv.SetWorldOffset({ WORLDWIDTH / 2.0f - sWidth / 2.0f, WORLDHEIGHT / 2.0f - sHeight / 2.0f });
        tv.SetZoom(sHeight / WORLDHEIGHT, { sWidth / 2.0f, sHeight / 2.0f });

        pos.reserve(N);
        vel.reserve(N);
        acc.reserve(N);

        for (int i = 0; i < N; i++)
        {
            pos.push_back({ float(rand()) / float(RAND_MAX) * WORLDWIDTH, float(rand()) / float(RAND_MAX) * WORLDHEIGHT });
            vel.push_back({ float(rand()) / float(RAND_MAX) * 2 * MAX_VEL - MAX_VEL, float(rand()) / float(RAND_MAX) * 2 * MAX_VEL - MAX_VEL });
            acc.push_back({ 0.0f, 0.0f });
        }

        NUM_THREADS = 1 << 10;
        NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

        cudaMalloc(&d_pos, bytes);
        cudaMalloc(&d_vel, bytes);
        cudaMalloc(&d_acc, bytes);

        cudaMemcpy(d_pos, pos.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vel, vel.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_acc, acc.data(), bytes, cudaMemcpyHostToDevice);

        return true;
    }

    void boidAvoidBoundies(float dt)
    {
        float steerAngle = 0.0;
        float boidRelDirection;

        boidRelDirection = dot(normalize(vel[0]), { 0.0f, 1.0f });
        if (boidRelDirection > 0.0)
        {
            if ((WORLDHEIGHT - pos[0].y) < 100.0)
            {
                steerAngle -= ((90.0f * 3.14 / 180.0f) - acos(boidRelDirection)) / 10.0f * dt;
            }
        }

        boidRelDirection = dot(normalize(vel[0]), { 1.0f, 0.0f });
        if (boidRelDirection > 0.0)
        {
            if ((WORLDWIDTH - pos[0].x) < 100.0)
            {
                steerAngle -= ((90.0f * 3.14 / 180.0f) - acos(boidRelDirection)) / 10.0f * dt;
            }
        }

        boidRelDirection = dot(normalize(vel[0]), { -1.0, 0.0 });
        //boid in moving towards it, if dot product is greater than 0
        if (boidRelDirection > 0.0)
        {
            //check is the boid is coming close
            if (fabs(0 - pos[0].x) < 100.0)
            {
                //steer the boid away from the boundary
                steerAngle -= ((90 * 3.14 / 180) - acos(boidRelDirection)) / 10 * dt;
            }
        }

        boidRelDirection = dot(normalize(vel[0]), { 0.0f, -1.0f });
        //boid in moving towards it, if dot product is greater than 0
        if (boidRelDirection > 0.0)
        {
            //check is the boid is coming close
            if (fabs(0 - pos[0].y) < 100.0)
            {
                //steer the boid away from the boundary
                steerAngle -= ((90 * 3.14 / 180) - acos(boidRelDirection)) / 10 * dt;
            }
        }

        float2 result;
        result.x = vel[0].x * cos(steerAngle) + vel[0].y * sin(steerAngle);
        result.y = -1 * vel[0].x * sin(steerAngle) + vel[0].y * cos(steerAngle);
        vel[0] = result;
    }

    bool OnUserUpdate(float fElapsedTime) override
    {
        // called once per frame
        tv.HandlePanAndZoom(0);

        if (GetKey(olc::Key::ESCAPE).bPressed)
            return false;
        if (GetKey(olc::Key::W).bPressed)
            MAX_FORCE += 5;
        if (GetKey(olc::Key::S).bPressed)
            MAX_FORCE -= 5;
        if (GetKey(olc::Key::UP).bPressed)
            MAX_VEL += 5;
        if (GetKey(olc::Key::DOWN).bPressed)
            MAX_VEL -= 5;
        if (GetKey(olc::Key::RIGHT).bPressed)
            PERCEPTION_RADIUS += 5;
        if (GetKey(olc::Key::LEFT).bPressed)
            PERCEPTION_RADIUS -= 5;

        if (MAX_FORCE > 100.0f)
            MAX_FORCE = 100.0f;
        else if (MAX_FORCE < 0.1f)
            MAX_FORCE = 0.1f;

        if (MAX_VEL > 1000.0f)
            MAX_VEL = 1000.0f;
        else if (MAX_VEL < 1.0f)
            MAX_VEL = 1.0f;

        if (PERCEPTION_RADIUS > 1000.0f)
            PERCEPTION_RADIUS = 1000.0f;
        else if (PERCEPTION_RADIUS < 1.0f)
            PERCEPTION_RADIUS = 1.0f;

        auto start = high_resolution_clock::now();

        // Calculate Boid Steer
        calcSteer<<<NUM_BLOCKS, NUM_THREADS>>>(d_pos, d_vel, d_acc, N, PERCEPTION_RADIUS, MAX_FORCE, MAX_VEL);
        cudaDeviceSynchronize();

        // Update boid velocity
        updateVel<<<NUM_BLOCKS, NUM_THREADS>>>(d_vel, d_acc, fElapsedTime, N);
        cudaDeviceSynchronize();

        //cudaMemcpy(pos.data(), d_pos, bytes, cudaMemcpyDeviceToHost);
        //cudaMemcpy(vel.data(), d_vel, bytes, cudaMemcpyDeviceToHost);
        //boidAvoidBoundies();
        //cudaMemcpy(d_pos, pos.data(), bytes, cudaMemcpyHostToDevice);
        //cudaMemcpy(d_vel, vel.data(), bytes, cudaMemcpyHostToDevice);

        avoidBoundaries<<<NUM_BLOCKS, NUM_THREADS>>>(d_pos, d_vel, fElapsedTime, N);
        cudaDeviceSynchronize();

        // Update boid position
        updatePos<<<NUM_BLOCKS, NUM_THREADS>>>(d_pos, d_vel, fElapsedTime, N);
        cudaDeviceSynchronize();

        cudaMemcpy(pos.data(), d_pos, bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(vel.data(), d_vel, bytes, cudaMemcpyDeviceToHost);

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);

        Clear(olc::BLACK);

        tv.FillRect({ 0.0f, 0.0f }, { WORLDWIDTH, WORLDHEIGHT }, olc::Pixel(32, 32, 32));
        tv.FillCircle({ pos[0].x, pos[0].y }, PERCEPTION_RADIUS, olc::Pixel(255, 0, 0, 128));

        //float headlen = 3;
        //float vectorlen = 10;
        for (int i = 0; i < N; i++)
        {
            //float fromx = pos[i].x;
            //float fromy = pos[i].y;

            //float velocityLength = sqrtf(vel[i].x * vel[i].x + vel[i].y * vel[i].y);
            //float2 temp = { vel[i].x / velocityLength * vectorlen, vel[i].y / velocityLength * vectorlen };
            //float tox = pos[i].x + temp.x;
            //float toy = pos[i].y + temp.y;

            //float dx = tox - fromx;
            //float dy = toy - fromy;
            //float angle = atan2(dy, dx);

            //olc::Pixel color = olc::WHITE;

            //tv.DrawLineDecal(olc::vi2d(int(fromx), int(fromy)), olc::vi2d(int(tox), int(toy)), color);
            //tv.DrawLineDecal(olc::vi2d(int(tox), int(toy)), olc::vi2d(int(tox - headlen * cosf(angle - PI / 6)), int(toy - headlen * sinf(angle - PI / 6))), color);
            //tv.DrawLineDecal(olc::vi2d(int(tox), int(toy)), olc::vi2d(int(tox - headlen * cosf(angle + PI / 6)), int(toy - headlen * sinf(angle + PI / 6))), color);
            tv.FillRectDecal({ pos[i].x, pos[i].y }, { 2.0f, 2.0f }, olc::WHITE);
        }

        std::string frametime = std::to_string(fElapsedTime * 1000);
        std::string rounded_frametime = frametime.substr(0, frametime.find(".") + 4);
        std::string framerate = std::to_string(1.0f / fElapsedTime);
        std::string rounded_framerate = framerate.substr(0, framerate.find(".") + 1);
        std::string execution_time = std::to_string(duration.count());
        std::string rounded_execution_time = execution_time.substr(0, execution_time.find("."));
        DrawString(olc::vi2d(0, 20), "FT: " + rounded_frametime + "ms", olc::WHITE, 2);
        DrawString(olc::vi2d(0, 40), "FR: " + rounded_framerate + "fps", olc::WHITE, 2);
        DrawString(olc::vi2d(0, 60), "CT: " + execution_time + "ms", olc::WHITE, 2);
        DrawString(olc::vi2d(0, 80), "BOIDCOUNT: " + std::to_string(N), olc::WHITE, 2);
        DrawString(olc::vi2d(0, 110), "MAX_FORCE: " + std::to_string(MAX_FORCE), olc::WHITE, 2);
        DrawString(olc::vi2d(0, 130), "MAX_VEL: " + std::to_string(MAX_VEL), olc::WHITE, 2);
        DrawString(olc::vi2d(0, 150), "RADIUS: " + std::to_string(PERCEPTION_RADIUS), olc::WHITE, 2);

        DrawString(olc::vi2d(0, 180), "MAX_FORCE: W/S", olc::WHITE, 2);
        DrawString(olc::vi2d(0, 200), "MAX_VEL: UP/DOWN", olc::WHITE, 2);
        DrawString(olc::vi2d(0, 220), "RADIUS: RIGHT/LEFT", olc::WHITE, 2);

        return true;
    }
};


int main()
{
    Simulation demo;
    if (demo.Construct(1920, 1080, 1, 1, 1))
        demo.Start();

    return 0;
}
