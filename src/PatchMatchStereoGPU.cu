#include "PatchMatchStereoGPU.h"
#include <stdio.h>
#include <unistd.h>
#include <iostream>
#include <sys/time.h>
#include <string.h>
#include <assert.h>
#include <sstream>
#include <stdlib.h>
#include "lodepng.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <omp.h>
#include <pthread.h>
#include <queue>
#include <random>
#include <chrono>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

//#include<lua.hpp> 

//graph-based segmentation to multiple minimum spanning trees
#include "segment-graph.h"
typedef struct { uchar r, g, b; } rgb;

#include <set>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/config.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>

// setS as edge list enforces no parallel edges, takes extra time to look up though
typedef boost::adjacency_list < boost::setS, boost::vecS, boost::undirectedS> tree_graph_t;
typedef tree_graph_t::vertex_descriptor tree_vertex_descriptor;
typedef tree_graph_t::edge_descriptor tree_edge_descriptor;
typedef std::pair<int, int> tree_edge;

#define THREADS 12

struct EdgeWeight
{
	double weight;
	double weight2;
};

struct VertexProperties
{
	int parent_idx;
	std::vector<int> children_indices;
};

typedef boost::adjacency_list < boost::vecS, boost::vecS, boost::undirectedS, VertexProperties/*boost::no_property*/, EdgeWeight> mst_graph_t;
typedef mst_graph_t::vertex_descriptor mst_vertex_descriptor;
typedef mst_graph_t::edge_descriptor mst_edge_descriptor;
typedef std::pair<int, int> mst_edge;

int devCount;
double* init_time;
double* main_time;
double* post_time; 
double average = 0.0;
double sd = 0.0;

// for cost volume guided filter
texture<float, 2> tex;
texture<float, 2> tex_left_guide;
texture<float, 2> tex_right_guide;
texture<float, 2> tex_left_mean_guide;
texture<float, 2> tex_right_mean_guide;
texture<float, 2> tex_left_var_g;
texture<float, 2> tex_right_var_g;
texture<float, 2> tex_lh;
texture<float, 2> tex_rh;
texture<uchar4, 2, cudaReadModeNormalizedFloat> rgbaTex;
texture<float, cudaTextureType3D, cudaReadModeElementType> tex_right_cost_vol;
texture<float, cudaTextureType3D, cudaReadModeElementType> tex_left_cost_vol;
cudaArray *cudaray_right_cost_vol, *cudaray_left_cost_vol;
cudaArray *d_array, *d_array_lh, *d_array_rh;
cudaArray *d_array_left_guide, *d_array_left_mean_guide, *d_array_left_var_g;
cudaArray *d_array_right_guide, *d_array_right_mean_guide, *d_array_right_var_g;

float* d_left_gray = NULL;
float* d_right_gray = NULL;
float* d_left_cost_vol = NULL;
float* d_right_cost_vol = NULL;

float* d_tmp_global = NULL;
float* d_mean_guide_global = NULL;
float* d_corr_g_global = NULL;
float* d_var_g_global = NULL;
float* d_mean_input_global = NULL;
float* d_corr_gi_global = NULL;
float* d_a_global = NULL;
float* d_b_global = NULL;
float* d_guide_global = NULL;
float* d_input_global = NULL;
const int num_threads = 32;	//CUDA thread block size

//color version guided filter
float *d_b = NULL;
float *d_g = NULL;
float *d_r = NULL;
float *d_mean_b = NULL;
float *d_mean_g = NULL;
float *d_mean_r = NULL;
float *d_var_rr = NULL;
float *d_var_rg = NULL;
float *d_var_rb = NULL;
float *d_var_gg = NULL;
float *d_var_gb = NULL;
float *d_var_bb = NULL;
float *d_inv_rr = NULL;
float *d_inv_rg = NULL;
float *d_inv_rb = NULL;
float *d_inv_gg = NULL;
float *d_inv_gb = NULL;
float *d_inv_bb = NULL;
float *d_cov_det = NULL;
float *d_mean_I_r = NULL;
float *d_mean_I_g = NULL;
float *d_mean_I_b = NULL;
float *d_cov_I_r = NULL;
float *d_cov_I_g = NULL;
float *d_cov_I_b = NULL;
float *d_a_r = NULL;
float *d_a_g = NULL;
float *d_a_b = NULL;

struct HostThreadData
{
	int id, cols, rows, img_size_pad_rows, num_disp, win_rad_bf;
	float eps;
	float *d_guide, *d_tmp, *d_mean_guide, *d_corr_g, *d_var_g;
	float *d_input, *d_mean_input, *d_corr_gi, *d_a, *d_b;
};


#if USE_PCL
bool show_normals = true;
struct callback_args{
  // structure used to pass arguments to the callback function
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
	pcl::PointCloud<pcl::Normal>::Ptr normals;
	pcl::visualization::PCLVisualizer::Ptr viewerPtr;
};

void pp_callback (const pcl::visualization::PointPickingEvent& event, void* args)
{
	struct callback_args* data = (struct callback_args *)args;
	pcl::PointCloud<pcl::Normal>::Ptr normals = data->normals;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = data->cloud;

	if(show_normals) data->viewerPtr->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals, 5, 10.0f, "normals");
	else data->viewerPtr->removePointCloud("normals");
	show_normals = !show_normals;
}
#endif

#define POST_PROCESSING 0

// CPU timing
struct timeval timerStart;
//void StartTimer();
//double GetTimer();

__device__ float theta_sigma_d = 0.0f;
__device__ float theta_sigma_n = 0.0f;


// GPU timing
void StartTimer_GPU(cudaEvent_t* start, cudaEvent_t* stop);
float GetTimer_GPU(cudaEvent_t* start, cudaEvent_t* stop);

void loadPNG(float* img_ptr, float* R, float* G, float* B, std::string file_name, int* cols, int* rows);
void savePNG(unsigned char* disp, std::string fileName, int cols, int rows);
void timingStat(double* time, int nt, double* average, double* sd);
int imgCharToFloat(unsigned char* imgCharPtr, float* imgFloatPtr, bool reverse, unsigned int imgSize, float scale);

void costVolumeGuidedFilterOMP(const cv::Mat & guide, float* d_cost_vol, const int cols, const int rows, const int img_size_pad_rows, 
				const int num_disp, double eps, const int gfsize);

cv::Mat guidedFilter(cv::Mat input, cv::Mat guide, int gfsize, double reg)
{
	cv::Mat meanguide, meaninput;
	cv::blur(input, meaninput, cv::Size(gfsize,gfsize));
	cv::blur(guide, meanguide, cv::Size(gfsize,gfsize));

	cv::Mat corr_g, corr_gi;
	cv::multiply(guide,guide,corr_g);
	cv::blur(corr_g,corr_g, cv::Size(gfsize,gfsize));

	cv::multiply(input, guide, corr_gi);
	cv::blur(corr_gi, corr_gi, cv::Size(gfsize,gfsize));
	//2
	cv::Mat var_g;
	cv::multiply(meanguide, meanguide,var_g);
	var_g = var_g*(-1) + corr_g;
	cv::Mat cov_gi;
	cv::multiply(meanguide, meaninput, cov_gi);
	cov_gi = cov_gi*(-1) + corr_gi;
	//3
	cv::Mat a = cov_gi/(var_g+reg);
	cv::Mat b;
	cv::multiply(a, meanguide, b);
	b = b*(-1) + meaninput;
	//4
	cv::blur(a, a, cv::Size(gfsize,gfsize));
	cv::blur(b, b, cv::Size(gfsize,gfsize));
	//5
	cv::multiply(a, guide, input);
	input += b;

	return input;
}

__global__ void boxFilterCUDABaseline(float* d_src, float* d_dst, const int cols, const int rows, const int rad)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x>=cols || y>=rows) return;

	float N = (float)((rad*2+1)*(rad*2+1));
	float t = 0.0f;
	int h, w;

	const int idx = y*cols+x;

	for(h=-rad; h<=rad; h++)
		for(w=-rad; w<=rad; w++)
			if(h+y>=0 && h+y<rows && w+x>=0 && w+x<=cols) t += d_src[(y+h)*cols+x+w];
	
	d_dst[idx] = t/N;
}

// each thread process one row
__global__ void rowSumIntegral(float* d_img, float* d_out, const int rows, const int cols)
{
	const int y = threadIdx.x;

	if(y >= rows) return;

	int row_start = y*cols;

	for(int x=1; x<cols; x++) d_out[row_start+x] += d_img[row_start+x-1];
}

__global__ void transposeImage(float* d_in, const int in_rows, const int in_cols, float* d_out)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= in_cols || y >= in_rows) return;
	
	const int in_idx = y*in_cols + x;
	const int out_idx = x*in_rows + y;

	d_out[out_idx] = d_in[in_idx];
}

__global__ void boxFilterGPU(float* integral_img, float* out, const int rows, const int cols, const int win_rad)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x>=cols || y>=rows) return;

	const int pixel_idx = y*cols+x;

	if(x-win_rad<0 || y-win_rad<0 || x+win_rad>=cols || y+win_rad>=rows) 
	{
		out[pixel_idx] = 0.0f;
		return;
	}

	out[pixel_idx] = integral_img[pixel_idx + win_rad + win_rad*cols] 
			+ integral_img[pixel_idx - win_rad - win_rad*cols]
			- integral_img[pixel_idx + win_rad - win_rad*cols]
			- integral_img[pixel_idx - win_rad + win_rad*cols];

	out[pixel_idx] /= (float)( (2*win_rad+1)*(2*win_rad+1) );
}


// process row
__device__ void
d_boxfilter_x(float *id, float *od, int w, int h, int r)
{
    float scale = 1.0f / (float)((r << 1) + 1);

    float t;
    // do left edge
    t = id[0] * r;

    for (int x = 0; x < (r + 1); x++)
    {
        t += id[x];
    }

    od[0] = t * scale;

    for (int x = 1; x < (r + 1); x++)
    {
        t += id[x + r];
        t -= id[0];
        od[x] = t * scale;
    }

    // main loop
    for (int x = (r + 1); x < w - r; x++)
    {
        t += id[x + r];
        t -= id[x - r - 1];
        od[x] = t * scale;
    }

    // do right edge
    for (int x = w - r; x < w; x++)
    {
        t += id[w - 1];
        t -= id[x - r - 1];
        od[x] = t * scale;
    }
}

// process column
__device__ void
d_boxfilter_y(float *id, float *od, int w, int h, int r)
{
    float scale = 1.0f / (float)((r << 1) + 1);

    float t;
    // do left edge
    t = id[0] * r;

    for (int y = 0; y < (r + 1); y++)
    {
        t += id[y * w];
    }

    od[0] = t * scale;

    for (int y = 1; y < (r + 1); y++)
    {
        t += id[(y + r) * w];
        t -= id[0];
        od[y * w] = t * scale;
    }

    // main loop
    for (int y = (r + 1); y < (h - r); y++)
    {
        t += id[(y + r) * w];
        t -= id[((y - r) * w) - w];
        od[y * w] = t * scale;
    }

    // do right edge
    for (int y = h - r; y < h; y++)
    {
        t += id[(h-1) * w];
        t -= id[((y - r) * w) - w];
        od[y * w] = t * scale;
    }
}

// texture version
// texture fetches automatically clamp to edge of image
__global__ void
d_boxfilter_x_tex(float *od, int w, int h, int r)
{
	float scale = 1.0f / (float)((r*2) + 1);
	unsigned int y = blockIdx.x*blockDim.x + threadIdx.x;
	if(y >= h) return; 

	float t = 0.0f;

	for (int x =- r; x <= r; x++)
	{
		t += tex2D(tex, x, y);
	}

	od[y * w] = t * scale;

	for (int x = 1; x < w; x++)
	{
		t += tex2D(tex, x + r, y);
		t -= tex2D(tex, x - r - 1, y);
		od[y * w + x] = t * scale;
	}
}

__global__ void boxFilter_x_tex(float* d_out, const int rows, const int cols, const int r)
{
	int x = blockIdx.x*blockDim.x;
	int y = blockIdx.y*blockDim.x + threadIdx.x;
	if(y>=rows) return;

	float t = 0.0f;
	float scale = 1.0f / (float)((r<<1) + 1);

	for(int i=x-r; i<=x+r; i++) t += tex2D(tex, i, y);

	d_out[y*cols+x] = t*scale;

	int end_idx = cols-x < blockDim.x ? cols : x+blockDim.x;


	for(int i=x+1; i<end_idx; i++)
	{
		t += tex2D(tex, i+r, y);
		t -= tex2D(tex, i-r-1, y);
		d_out[y*cols + i] = t*scale;
	}
}


__global__ void boxFilter_y_tex(float* d_out, const int rows, const int cols, const int r)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.x;
	if(x>=cols) return;

	float t = 0.0f;
	float scale = 1.0f / (float)((r<<1) + 1);

	for(int i=y-r; i<=y+r; i++) t += tex2D(tex, x, i);

	d_out[y*cols+x] = t*scale;

	int end_idx = rows-y < blockDim.x ? rows : y+blockDim.x;


	for(int i=y+1; i<end_idx; i++)
	{
		t += tex2D(tex, x, i+r);
		t -= tex2D(tex, x, i-r-1);
		d_out[i*cols + x] = t*scale;
	}
}


void boxFilterSlideWindow_tex(float* d_src, float* d_temp, float* d_dest, const int rows, const int cols, const int radius, const int num_threads)
{
	cudaMemcpyToArray(d_array, 0, 0, d_src, rows*cols*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaBindTextureToArray(tex, d_array);
	dim3 grid_size( (cols+num_threads-1)/num_threads, (rows+num_threads-1)/num_threads);
	boxFilter_y_tex<<<grid_size, num_threads>>>(d_dest, rows, cols, radius);
	//cudaDeviceSynchronize();

/*	dim3 block_size_t(16,16);
	dim3 grid_size_t( (cols+block_size_t.x-1)/block_size_t.x, (rows+block_size_t.y-1)/block_size_t.y);
	transposeImage<<<grid_size_t, block_size_t>>>(d_dest, rows, cols, d_temp);
*/

/*	dim3 dimBlock(TILE_DIM, BLOCK_ROWS);
	dim3 dimGrid((cols+dimBlock.x-1)/dimBlock.x, (rows+dimBlock.y-1)/dimBlock.y);

	transposeSC<float, false><<<dimGrid, dimBlock>>>(d_temp, d_dest, cols, rows);
	cudaDeviceSynchronize();

 
	cudaMemcpyToArray(d_array_T, 0, 0, d_temp, rows*cols*sizeof(float), cudaMemcpyDeviceToDevice);	
	cudaBindTextureToArray(tex, d_array_T);
	dim3 grid_size_tb( (rows+num_threads-1)/num_threads, (cols+num_threads-1)/num_threads);
	boxFilter_y_tex<<<grid_size_tb, num_threads>>>(d_temp, cols, rows, radius);
	cudaDeviceSynchronize();

	dim3 dimGridT((rows+dimBlock.x-1)/dimBlock.x, (cols+dimBlock.y-1)/dimBlock.y);
	transposeSC<float, false><<<dimGridT, dimBlock>>>(d_dest, d_temp, rows, cols);
	cudaDeviceSynchronize();
*/
	
//	transposeImage<<<grid_size_tb, block_size_t>>>(d_temp, cols, rows, d_dest);
	

	cudaMemcpyToArray(d_array, 0, 0, d_dest, rows*cols*sizeof(float), cudaMemcpyDeviceToDevice);
	boxFilter_x_tex<<<grid_size, num_threads>>>(d_dest, rows, cols, radius);
	//cudaDeviceSynchronize();
}


__global__ void boxFilter_x_global_shared(float* d_in, float* d_out, const int rows, const int cols, const int r)
{
	extern __shared__ float temp[];

	int x = blockIdx.x*blockDim.x;
	int y = blockIdx.y*blockDim.x + threadIdx.x;
	const int store_start_pitch = threadIdx.x*(blockDim.x+1);
	const int end_idx = cols-x < blockDim.x ? cols : x+blockDim.x;

	float t = 0.0f;
	float scale = 1.0f / (float)((r<<1) + 1);
	
	for(int i=x-r; i<=x+r; i++) t += (i<0||i>=cols) ? 0.0f : d_in[y*cols+i];

	temp[store_start_pitch] = t*scale;

	int li = 1;
	for(int i=x+1; i<end_idx; i++, li++)
	{
		t += (i+r>=cols) ? 0.0f : d_in[y*cols+i+r]; 
		t -= (i-r-1<0) ? 0.0f : d_in[y*cols+i-r-1];
		temp[store_start_pitch+li] = t*scale;
	}

	__syncthreads();

	// local row id
	for(int i=0; i<blockDim.x; i++)
			if(threadIdx.x+x<cols)
				d_out[(blockIdx.y*blockDim.x+i)*cols+x+threadIdx.x] = temp[i*(blockDim.x+1)+threadIdx.x];
}


__global__ void boxFilter_x_global(float* d_in, float* d_out, const int rows, const int cols, const int r)
{
	int x = blockIdx.x*blockDim.x;
	int y = blockIdx.y*blockDim.x + threadIdx.x;
	if(y>=rows) return;

	float t = 0.0f;
	float scale = 1.0f / (float)((r<<1) + 1);

	for(int i=x-r; i<=x+r; i++) t += (i<0||i>=cols) ? 0.0f : d_in[y*cols+i];

	d_out[y*cols+x] = t*scale;

	int end_idx = cols-x < blockDim.x ? cols : x+blockDim.x;

	for(int i=x+1; i<end_idx; i++)
	{
		t += (i+r>=cols) ? 0.0f : d_in[y*cols+i+r]; 
		t -= (i-r-1<0) ? 0.0f : d_in[y*cols+i-r-1];
		d_out[y*cols + i] = t*scale;
	}
}


__global__ void boxFilter_y_global(float* d_in, float* d_out, const int rows, const int cols, const int r)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.x;
	if(x>=cols) return;

	float t = 0.0f;
	float scale = 1.0f / (float)((r<<1) + 1);

	for(int i=y-r; i<=y+r; i++) t += (i<0||i>=rows) ? 0.0f : d_in[i*cols+x];

	d_out[y*cols+x] = t*scale;

	int end_idx = rows-y < blockDim.x ? rows : y+blockDim.x;

	for(int i=y+1; i<end_idx; i++)
	{
		t += (i+r>=rows) ? 0.0f : d_in[(i+r)*cols+x];
		t -= (i-r-1<0) ? 0.0f : d_in[(i-r-1)*cols+x];
		d_out[i*cols + x] = t*scale;
	}
}

__global__ void boxFilterPlusPointWiseMul_y_global(float* d_in, float* d_out, const int rows, const int cols, const int r, float* d_in_pm, float* d_out_pm)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.x;
	if(x>=cols) return;

	float t = 0.0f;
	float scale = 1.0f / (float)((r<<1) + 1);

	for(int i=y-r; i<=y+r; i++) t += (i<0||i>=rows) ? 0.0f : d_in[i*cols+x];

	int idx = y*cols+x;
	d_out[idx] = t*scale;

	d_out_pm[idx] = d_in[idx]*d_in_pm[idx];

	int end_idx = rows-y < blockDim.x ? rows : y+blockDim.x;

	for(int i=y+1; i<end_idx; i++)
	{
		t += (i+r>=rows) ? 0.0f : d_in[(i+r)*cols+x];
		t -= (i-r-1<0) ? 0.0f : d_in[(i-r-1)*cols+x];
		idx = i*cols+x;
		d_out[idx] = t*scale;
		d_out_pm[idx] = d_in[idx]*d_in_pm[idx];
	}
}

void boxFilterSlideWindow_global(float* d_src, float* d_temp, float* d_dest, const int rows, const int cols, const int radius, const int num_threads)
{
	dim3 grid_size( (cols+num_threads-1)/num_threads, (rows+num_threads-1)/num_threads);
	//store result in shared memory and then coelesced write to global memory
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float)>>>(d_src, d_temp, rows, cols, radius);
//	boxFilter_x_global<<<grid_size, num_threads>>>(d_temp, d_dest, rows, cols, radius);
	

	//coelesced read and write of global memory
	boxFilter_y_global<<<grid_size, num_threads>>>(d_temp, d_dest, rows, cols, radius);
}

__global__ void boxFilterVol_x_global(float* d_in, float* d_out, const int rows, const int cols, const int r)
{
	int x = blockIdx.x*blockDim.x;
	int y = blockIdx.y*blockDim.x + threadIdx.x;
	int z = blockIdx.z;
	if(y>=rows) return;

	float t = 0.0f;
	float scale = 1.0f / (float)((r<<1) + 1);
	const int z_pitch = z*cols*rows;

	for(int i=x-r; i<=x+r; i++) t += (i<0||i>=cols) ? 0.0f : d_in[z_pitch + y*cols+i];

	d_out[z_pitch + y*cols+x] = t*scale;

	int end_idx = cols-x < blockDim.x ? cols : x+blockDim.x;

	for(int i=x+1; i<end_idx; i++)
	{
		t += (i+r>=cols) ? 0.0f : d_in[z_pitch + y*cols+i+r]; //tex2D(tex, i+r, y);
		t -= (i-r-1<0) ? 0.0f : d_in[z_pitch + y*cols+i-r-1];//tex2D(tex, i-r-1, y);
		d_out[z_pitch + y*cols + i] = t*scale;
	}
}


__global__ void boxFilterVol_y_global(float* d_in, float* d_out, const int rows, const int cols, const int r)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.x;
	int z = blockIdx.z;
	if(x>=cols) return;

	float t = 0.0f;
	float scale = 1.0f / (float)((r<<1) + 1);
	const int z_pitch = z*cols*rows;

	for(int i=y-r; i<=y+r; i++) t += (i<0||i>=rows) ? 0.0f : d_in[z_pitch +i*cols+x];//tex2D(tex, x, i);

	d_out[z_pitch +y*cols+x] = t*scale;

	int end_idx = rows-y < blockDim.x ? rows : y+blockDim.x;

	for(int i=y+1; i<end_idx; i++)
	{
		t += (i+r>=rows) ? 0.0f : d_in[z_pitch + (i+r)*cols+x];//tex2D(tex, x, i+r);
		t -= (i-r-1<0) ? 0.0f : d_in[z_pitch + (i-r-1)*cols+x];//tex2D(tex, x, i-r-1);
		d_out[z_pitch +i*cols + x] = t*scale;
	}
}

void boxFilterSlideWindowVol_global(float* d_src, float* d_temp, float* d_dest, const int rows, const int cols, const int depth, const int radius, const int num_threads)
{
	dim3 grid_size( (cols+num_threads-1)/num_threads, (rows+num_threads-1)/num_threads, depth);
	boxFilterVol_y_global<<<grid_size, num_threads>>>(d_src, d_temp, rows, cols, radius);

	boxFilterVol_x_global<<<grid_size, num_threads>>>(d_temp, d_dest, rows, cols, radius);
}


__global__ void
d_boxfilter_y_tex(float *od, int w, int h, int r)
{
	float scale = 1.0f / (float)((r*2) + 1);
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	if(x>=w) return;

	float t = 0.0f;

	for (int y = -r; y <= r; y++)
	{
		t += tex2D(tex, x, y);
	}

	od[x] = t * scale;

	for (int y = 1; y < h; y++)
	{
		t += tex2D(tex, x, y + r);
		t -= tex2D(tex, x, y - r - 1);
		od[y * w + x] = t * scale;
	}
}

__global__ void
d_boxfilter_x_global(float *id, float *od, int w, int h, int r)
{
	int y = blockIdx.x*blockDim.x + threadIdx.x;
	if(y>=h) return;
	d_boxfilter_x(&id[y * w], &od[y * w], w, h, r);
}

__global__ void
d_boxfilter_y_global(float *id, float *od, int w, int h, int r)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	if(x>=w) return;
	d_boxfilter_y(&id[x], &od[x], w, h, r);
}


void freeTextures()
{
    cudaFreeArray(d_array_lh);
    cudaFreeArray(d_array_rh);
    cudaFreeArray(d_array);
    cudaFreeArray(d_array_left_guide);
    cudaFreeArray(d_array_left_mean_guide);
    cudaFreeArray(d_array_left_var_g);
    cudaFreeArray(d_array_right_guide);
    cudaFreeArray(d_array_right_mean_guide);
    cudaFreeArray(d_array_right_var_g);
}

void boxFilterCUDA(float* d_src, float *d_temp, float *d_dest, int width, int height, int radius, int nthreads)
{
	cudaMemcpyToArray(d_array, 0, 0, d_src, width*height*sizeof(float), cudaMemcpyDeviceToDevice);
	
	cudaBindTextureToArray(tex, d_array);

	// use texture for horizontal pass
	d_boxfilter_x_tex<<< (height+nthreads-1)/nthreads, nthreads, 0 >>>(d_temp, width, height, radius);
	d_boxfilter_y_global<<< (width+nthreads-1) / nthreads, nthreads, 0 >>>(d_temp, d_dest, width, height, radius);

//	d_boxfilter_y_tex<<< (width+nthreads-1) / nthreads, nthreads, 0 >>>(d_temp, width, height, radius);
//	d_boxfilter_x_global<<< (height+nthreads-1)/ nthreads, nthreads, 0 >>>(d_temp, d_dest, width, height, radius);
}

__global__ void boxFilter_tex(float* d_dest, const int rows, const int cols, const int radius)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x>=cols || y>=rows) return;

	float mean = 0.0f;

	for(int h=-radius; h<=radius; h++)
	{	
		for(int w=-radius; w<=radius; w++)
		{
			mean += tex2D(tex, x+w, y+h);
		}
	}

	int len = radius<<1+1;

	d_dest[y*cols+x] = mean/(float)(len*len);
}

__global__ void pointWiseMulY_global(float* d_img1, float* d_img2, float* d_dest, const int rows, const int cols)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	if(x>=cols) return;

	for(int y=0; y<rows; y++)
	{
		int idx = y*cols + x;
		d_dest[idx] = d_img1[idx]*d_img2[idx];
	}	
}

__global__ void pointWiseMul_global(float* d_img1, float* d_img2, float* d_dest, const int rows, const int cols)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if(x>=cols || y>=rows) return;
	
	const int idx = y*cols + x;
	d_dest[idx] = d_img1[idx]*d_img2[idx];
}


__global__ void pointWiseMul_tex(float* d_dest, const int rows, const int cols)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if(x>=cols || y>=rows) return;
	
	d_dest[y*cols+x] = tex2D(tex_lh, x, y)*tex2D(tex_rh, x, y);
}


void pointWiseMultiplyCUDA(float* d_img1, float* d_img2, float* d_dest, const int rows, const int cols, const int threads=64)
{
	// coalesce
	pointWiseMulY_global<<< (cols+threads-1)/threads, threads>>>(d_img1, d_img2, d_dest, rows, cols);
}

__global__ void pointWiseSub_global(float* d_left, float* d_right, float* d_dest, const int rows, const int cols, bool add=false)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x>=cols || y>=rows) return;

	const int idx = y*cols + x;

	d_dest[idx] = add ? d_left[idx] + d_right[idx] : d_left[idx] - d_right[idx];
}

__global__ void pointWiseSubY_global(float* d_left, float* d_right, float* d_dest, const int rows, const int cols, bool add=false)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	if(x>=cols) return;

	if(!add)
	for(int y=0; y<rows; y++)
	{
		int idx = y*cols + x;
		d_dest[idx] = d_left[idx] - d_right[idx];
	}
	else	
	for(int y=0; y<rows; y++)
	{
		int idx = y*cols + x;
		d_dest[idx] = d_left[idx] + d_right[idx];
	}	
}

__global__ void pointWise_A_div_Beps_Y_global(float* d_A, float* d_B, float eps, float* d_dest, const int rows, const int cols)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	if(x>=cols) return;

	for(int y=0; y<rows; y++)
	{
		int idx = y*cols + x;
		d_dest[idx] = d_A[idx]/(d_B[idx] + eps);
	}	
}

__global__ void pointWise_A_div_Beps_global(float* d_A, float* d_B, float eps, float* d_dest, const int rows, const int cols)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if(x>=cols || y>=rows) return;

	const int idx = y*cols + x;
	d_dest[idx] = d_A[idx]/(d_B[idx] + eps);
}


// u, v already +0.5f
// kernels
// evaluate window-based disimilarity unary cost
__device__ float evaluateCost(cudaTextureObject_t lR_to, cudaTextureObject_t lG_to,  cudaTextureObject_t lB_to,
				cudaTextureObject_t lGrad_to, cudaTextureObject_t rR_to,  cudaTextureObject_t rG_to,
				cudaTextureObject_t rB_to,  cudaTextureObject_t rGrad_to,
				cudaTextureObject_t lGray_to,  cudaTextureObject_t rGray_to,
				const float &u, const float &v, const int &x, const int &y, const float &disp, const int &cols, const int &rows, const int &min_disp, const int &max_disp,
                                const int &winRadius, const float &nx, const float &ny, const float &nz, const int &base) // base 0 left, 1 right
{
        float cost = 0.0f;
        float weight;
        float af, bf, cf;

        float xf = (float)x;
        float yf = (float)y;
	const float du = 1.0f/(float)cols;
	const float dv = 1.0f/(float)rows;

	const float alpha_c = 0.1f;
	const float alpha_g = 1.0f - alpha_c;
	const float weight_c = 1.0f/10.f;
	const float color_truncation = 10.0f;
	const float gradient_truncation = 2.0f;
	const float bad_cost = alpha_c*color_truncation + alpha_g*gradient_truncation;
	

//	nx = 0.f; ny = 0.f; nz = 1.f;

        // af = -nx/nz, bf = -ny/nz, cf = (nx*x+ny*y+nz*disp)/nz
        af = nx/nz*(-1.0f);
        bf = ny/nz*(-1.0f);
        cf = (nx*xf + ny*yf + nz*disp)/nz;



        if( isinf(af)!=0 || isinf(bf)!=0 || isinf(cf)!=0) return bad_cost;

        float weight_sum = 0.0f;

        float tmp_disp, r, g, b, color_dist, wn, hn, sign;

	int h, w;

	cudaTextureObject_t* baseR;
	cudaTextureObject_t* baseG;
	cudaTextureObject_t* baseB;
	cudaTextureObject_t* matchR;
	cudaTextureObject_t* matchG;
	cudaTextureObject_t* matchB;
	cudaTextureObject_t* baseGrad;
	cudaTextureObject_t* matchGrad;
	cudaTextureObject_t* baseGray;
	cudaTextureObject_t* matchGray;

	if(base == 0) // left base
	{
		sign = -1.0f;
		baseR = &lR_to; baseG = &lG_to;	baseB = &lB_to;
		matchR = &rR_to; matchG = &rG_to; matchB = &rB_to;
		baseGrad = &lGrad_to;	
		matchGrad = &rGrad_to; 
		baseGray = &lGray_to;
		matchGray = &rGray_to;

	}
	else	// right base
	{
		sign = 1.0f;
		baseR = &rR_to; baseG = &rG_to; baseB = &rB_to;
		matchR = &lR_to; matchG = &lG_to; matchB = &lB_to;
		baseGrad = &rGrad_to;
		matchGrad = &lGrad_to; 
		baseGray = &rGray_to;
		matchGray = &lGray_to;
	}

	

	for(h=-winRadius; h<=winRadius; h+=1)
        {
                for(w=-winRadius; w<=winRadius; w+=1)
                {
                        tmp_disp = (af*(xf+(float)w) + bf*(yf+(float)h) + cf)*sign;
			const float match_center_disp = (af*xf + bf*yf + cf)*sign;

			if(tmp_disp*sign>=min_disp && tmp_disp*sign<=max_disp && tmp_disp+x>=0 && tmp_disp+x<cols
				&& match_center_disp*sign>=min_disp && match_center_disp*sign<=max_disp && match_center_disp+x>=0 && match_center_disp+x<cols
					)
			{
		                tmp_disp = tmp_disp*du;

		                wn = (float)w*du;
		                hn = (float)h*dv;

		                r = (tex2D<float>(*baseR, u, v)-tex2D<float>(*baseR, u+wn, v+hn));
		                g = (tex2D<float>(*baseG, u, v)-tex2D<float>(*baseG, u+wn, v+hn));
		                b = (tex2D<float>(*baseB, u, v)-tex2D<float>(*baseB, u+wn, v+hn));

	                        weight = expf(-(fabsf(r)+fabsf(b)+fabsf(g))*weight_c);
				//weight = expf(-(fabsf(r)+fabsf(b)+fabsf(g))*0.333333333f*weight_c);

		                weight_sum += weight;

#if 0				//census

				r = r*(tex2D<float>(*matchR, u + match_center_disp, v)-tex2D<float>(*matchR, u + tmp_disp + wn, v + hn)) < 0.0f ? 0.9f : 0.0f;
				g = g*(tex2D<float>(*matchG, u + match_center_disp, v)-tex2D<float>(*matchG, u + tmp_disp + wn, v + hn)) < 0.0f ? 0.9f : 0.0f;
				b = b*(tex2D<float>(*matchB, u + match_center_disp, v)-tex2D<float>(*matchB, u + tmp_disp + wn, v + hn)) < 0.0f ? 0.9f : 0.0f;
				color_dist = (r+g+b);
				cost += weight*color_dist;
				//cost += weight*(alpha_c*min(color_dist, color_truncation)+alpha_g*min(fabsf(tex2D<float>(*baseGrad, u + wn, v + hn)-tex2D<float>(*matchGrad, u + tmp_disp + wn, v + hn)), gradient_truncation));
#endif
	

#if 1
/*
		                r = fabsf( tex2D<float>(*baseR, u + wn, v + hn) - tex2D<float>(*matchR, u + tmp_disp + wn, v + hn));
		                g = fabsf( tex2D<float>(*baseG, u + wn, v + hn) - tex2D<float>(*matchG, u + tmp_disp + wn, v + hn));
		                b = fabsf( tex2D<float>(*baseB, u + wn, v + hn) - tex2D<float>(*matchB, u + tmp_disp + wn, v + hn));

		                color_dist = (r+g+b)*0.33333333333f;*/
				//color_dist = (r+g+b);
color_dist = (tex2D<float>(*baseGray, u, v)-tex2D<float>(*baseGray, u+wn, v+hn))*(tex2D<float>(*matchGray, u + match_center_disp*du, v)-tex2D<float>(*matchGray, u+wn+tmp_disp, v+hn))< 0.0f ? 0.1f : 0.0f;

				cost += weight*color_dist;
		                //cost += weight*(alpha_c*min(color_dist, color_truncation)+alpha_g*min(fabsf(tex2D<float>(*baseGrad, u + wn, v + hn)-tex2D<float>(*matchGrad, u + tmp_disp + wn, v + hn)), gradient_truncation));
#endif
			}
			else
			{
				//cost += bad_cost;
				//cost += 1.0f;
				cost += 10.0f;
				weight_sum += 1.0f;
			}
                }
        }
	
	return cost/weight_sum;
}


__global__ void perPixelWeightPlusNormalizeImg(float* d_gray, float* d_weight, const float alpha, const float beta, const int rows, const int cols,  
						float* d_disp, float* d_denoise, const int min_disp, const int max_disp)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
        const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x>=cols || y>= rows) return;

	const int idx = y*cols + x;

	float dx = x == cols-1 ? 0 : (d_gray[idx+1] - d_gray[idx])/255.f;
	float dy = y == rows-1 ? 0 : (d_gray[idx+cols] - d_gray[idx])/255.f;
	
	d_weight[idx] = expf(-alpha*powf(fabsf(sqrtf(dx*dx+dy*dy)), beta));

//	d_weight[idx] = 1.0f;

        d_disp[idx] = (d_disp[idx] - (float)min_disp)/(float)(max_disp-min_disp);	//scale dispariyt to [0,1] 

//	d_disp[idx] = (1.0f/d_disp[idx] - 1.0f/(float)max_disp)/(1.0f/(float)min_disp - 1.0f/(float)max_disp);	//depth

	d_denoise[idx] = d_disp[idx];
}

__global__ void scaleDisparityBack(float* d_disp, const int min_disp, const int max_disp, const int rows, const int cols)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
        const int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x>=cols || y>= rows) return;

	const int idx = y*cols + x;

	//d_disp[idx] = 1.f/d_disp[idx];

	d_disp[idx] =  d_disp[idx]*((float)(max_disp-min_disp)) + (float)min_disp;	//disparity

//	d_disp[idx] = 1.0f/(d_disp[idx]*(1.0f/(float)min_disp-1.0f/(float)max_disp) + 1.0f/(float)max_disp);	//depth

}

__global__ void weightedHuberDenoiseDualUpdate(float* d_img, float* d_denoise/*primal*/, float* d_weight, float* d_data_dual, 
						float* d_regu_dual_x, float* d_regu_dual_y, const int rows, const int cols, 
						const float lambda, const float delta/*regu*/, const float gamma/*data*/, const float sigma)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
        const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	
	if(x>=cols || y>=rows) return;

	const int idx = y*cols + x;

	//primal gradient
	float dprimal_dx = x == cols-1 ? 0.0f : d_denoise[idx+1] - d_denoise[idx];
	float dprimal_dy = y == rows-1 ? 0.0f : d_denoise[idx+cols] - d_denoise[idx];


	//ROF model test, handa's paper. Newcombe added lambda, which is wrong
/*	float new_regu_dual_x = (d_regu_dual_x[idx] + sigma*dprimal_dx);
	float new_regu_dual_y = (d_regu_dual_y[idx] + sigma*dprimal_dy);
	float len = fmaxf(1.0f, sqrtf(new_regu_dual_x*new_regu_dual_x+new_regu_dual_y*new_regu_dual_y));	
	d_regu_dual_x[idx] = new_regu_dual_x/len;
	d_regu_dual_y[idx] = new_regu_dual_y/len;
*/
	//Huber ROF model handa's paper
/*	float new_regu_dual_x = (d_regu_dual_x[idx] + sigma*dprimal_dx)/(1.0f+sigma*delta);
	float new_regu_dual_y = (d_regu_dual_y[idx] + sigma*dprimal_dy)/(1.0f+sigma*delta);
	float len = fmaxf(1.f, sqrtf(new_regu_dual_x*new_regu_dual_x+new_regu_dual_y*new_regu_dual_y)/d_weight[idx]);	
	d_regu_dual_x[idx] = new_regu_dual_x/len;
	d_regu_dual_y[idx] = new_regu_dual_y/len;
*/

	//TV-L1 test handa's paper
/*	float new_regu_dual_x = (d_regu_dual_x[idx] + sigma*dprimal_dx);
	float new_regu_dual_y = (d_regu_dual_y[idx] + sigma*dprimal_dy);
	float len = fmaxf(1.0f, sqrtf(new_regu_dual_x*new_regu_dual_x+new_regu_dual_y*new_regu_dual_y));	
	d_regu_dual_x[idx] = new_regu_dual_x/len;
	d_regu_dual_y[idx] = new_regu_dual_y/len;
	float new_data_dual = (d_data_dual[idx] + sigma*(d_denoise[idx] - d_img[idx]));
	d_data_dual[idx] = new_data_dual/fmaxf(1.0f, fabsf(new_data_dual)/lambda);
*/

	//weighted huber newcombe
	float new_data_dual = (d_data_dual[idx] + sigma*(d_denoise[idx] - d_img[idx]))/(1.0f+sigma*gamma);
	d_data_dual[idx] = new_data_dual / fmaxf( 1.0f, fabsf(new_data_dual)/lambda );
	float new_regu_dual_x = (d_regu_dual_x[idx] + sigma*dprimal_dx)/(1.0f+sigma*delta);
	float new_regu_dual_y = (d_regu_dual_y[idx] + sigma*dprimal_dy)/(1.0f+sigma*delta);
	float len = fmaxf(1.0f, sqrtf(new_regu_dual_x*new_regu_dual_x + new_regu_dual_y*new_regu_dual_y)/d_weight[idx]);
	d_regu_dual_x[idx] = new_regu_dual_x/len;
	d_regu_dual_y[idx] = new_regu_dual_y/len;	

}

__global__ void weightedHuberDenoisePrimalUpdate(float* d_img, float* d_denoise/*primal*/, float* d_weight, float* d_data_dual, float* d_regu_dual_x, float* d_regu_dual_y, 
							const int rows, const int cols, const float lambda, const float tau)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
        const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x>=cols || y>=rows) return;

	const int idx = y*cols + x;

	//dual divergence
	float div_x, div_y;
	
	if(x==0) div_x = d_regu_dual_x[idx];
	else if(x==cols-1) div_x = -d_regu_dual_x[idx-1];
	else div_x = d_regu_dual_x[idx]-d_regu_dual_x[idx-1];


	if(y==0) div_y = d_regu_dual_y[idx];
	else if(y==rows-1) div_y = -d_regu_dual_y[idx-cols];
	else div_y = d_regu_dual_y[idx]-d_regu_dual_y[idx-cols];

//	d_denoise[idx] = d_denoise[idx] + tau*((div_x+div_y) - lambda*d_data_dual[idx]);	//TV-L1

//	d_denoise[idx] = d_denoise[idx] + tau*((div_x+div_y) - lambda*d_data_dual[idx]);	// weighted Huber newcombe

	d_denoise[idx] = 2.0f*(d_denoise[idx] + tau*((div_x+div_y) - lambda*d_data_dual[idx])) - d_denoise[idx];	// weighted Huber newcombe plus extrapolation

//	d_denoise[idx] = (d_denoise[idx] + tau*((div_x+div_y) + lambda*d_img[idx]))/(1.0f+tau*lambda);	//ROF model

//	d_denoise[idx] = (d_denoise[idx] + tau*(div_x+div_y + lambda*d_img[idx]))/(1.0f+tau*lambda);	//Huber ROF
}

__global__ void handleOcclusionSharedMemory(float* d_left_disp, float* d_right_disp, const int rows, const int cols, const int min_disp, const int max_disp, 
						const float disp_thresh = 1.0f, bool remove_occlusion = false)
{
	const int x = threadIdx.x;
        const int y = blockIdx.x;

	extern __shared__ float s_disp_tile[];


	if(x>=cols) return;

	const int pixel_idx = y*cols + x;
	const int copy_pitch = 2*cols;
	const float invalid_disp = 1e18f;
	
	s_disp_tile[x] = d_left_disp[pixel_idx];

	s_disp_tile[x+cols] = d_right_disp[pixel_idx];

	s_disp_tile[x+copy_pitch] = s_disp_tile[x];

	s_disp_tile[x+copy_pitch+cols] = s_disp_tile[x+cols];

	__syncthreads();


	// mark occlusions in copy
	// left
	int right_x = x - s_disp_tile[x];
	int left_x = x;

	if((right_x >= 0 && fabsf(s_disp_tile[right_x+cols] - s_disp_tile[left_x]) > disp_thresh) || right_x < 0)
		s_disp_tile[left_x+copy_pitch] = invalid_disp;

	// right
	left_x = x + s_disp_tile[x+cols];
	right_x = x;

	if( (left_x < cols && fabsf(s_disp_tile[right_x+cols] - s_disp_tile[left_x]) > disp_thresh) || left_x >= cols)
		s_disp_tile[right_x+copy_pitch+cols] = invalid_disp;
	
	// fill occlusions
	if(remove_occlusion)
	{
		if(s_disp_tile[x+copy_pitch] == invalid_disp)
			d_left_disp[pixel_idx] = min_disp;

		if(s_disp_tile[x+cols+copy_pitch] == invalid_disp)
			d_right_disp[pixel_idx] = min_disp;
		return;
	}


	// left
	right_x = x - s_disp_tile[x];
	left_x = x;
	if(s_disp_tile[left_x+copy_pitch] == invalid_disp)
	{
		float left_search_result, right_search_result;
		//search left
		while(true)
		{
			left_x--;

			if( left_x >= 0)
			{
				if(s_disp_tile[left_x+copy_pitch] != invalid_disp)
				{	
					left_search_result = s_disp_tile[left_x];
					break;
				}
			}
			else
			{
				left_search_result = invalid_disp;
				break;		
			}
		}

		//search right
		left_x = x;
		while(true)
		{
			left_x++;

			if( left_x < cols )
			{
				if( s_disp_tile[left_x+copy_pitch] != invalid_disp )
				{	
					right_search_result = s_disp_tile[left_x];
					break;	
				}
			}
			else
			{
				right_search_result = invalid_disp;
				break;
			}
		}

		if(right_search_result == invalid_disp && left_search_result == invalid_disp)
			d_left_disp[pixel_idx] = 255.f;//0.0f;
		else
			d_left_disp[pixel_idx] = fminf(left_search_result, right_search_result);
	}


	// process right disparity map
	left_x = x + s_disp_tile[x+cols];
	right_x = x;

	if(s_disp_tile[right_x+cols+copy_pitch] == invalid_disp)
	{
		float left_search_result, right_search_result;
		//search left
		while(true)
		{
			right_x--;

			if( right_x >= 0 )
			{
				if( s_disp_tile[right_x+cols+copy_pitch] != invalid_disp ) 
				{
					left_search_result = s_disp_tile[right_x+cols];
					break;
				}
			}
			else
			{
				left_search_result = invalid_disp;
				break;
			}
		}

		//search right
		right_x = x;
		while(true)
		{
			right_x++;

			if(right_x < cols)
			{
				if(s_disp_tile[right_x+cols+copy_pitch] != invalid_disp)
				{
					right_search_result = s_disp_tile[right_x+cols];
					break;
				}
			}
			else
			{
				right_search_result = invalid_disp;
				break;
			}
		}

		if(right_search_result == invalid_disp && left_search_result == invalid_disp)
			d_right_disp[pixel_idx] = 255.f;//0.0f;
		else
			d_right_disp[pixel_idx] = fminf(left_search_result, right_search_result);
	}
}





#define ZNCC
//#define AD
//#define AGD

__global__ void buildCostVolumeSharedMemory(float* d_left_gray, float* d_right_gray, float* d_left_cost_vol, float* d_right_cost_vol, 
						const int rows, const int cols, const int min_disp, const int max_disp, const int win_rad,
						const int img_size_pad_rows)
{
	const int x = threadIdx.x;
        const int y = blockIdx.x;

#ifdef ZNCC
	const float  bad_cost = 0.0f;
#endif

#ifdef AD
	const float color_truncation = 7.0f;
	const float bad_cost = color_truncation + 0.5f;
#endif

#ifdef AGD
	const float color_truncation = 7.0f;
	const float gradient_truncation = 2.0f;
	const float bad_cost = 3.0f;
#endif

	extern __shared__ float s_gray_tile[];
	
//	if(x>=cols) return;

	const int pixel_idx = y*cols+x;
	const int tile_pitch = (win_rad*2+1)*cols;

	// move global image tile to shared memory
	for(int i=-win_rad; i<=win_rad; i++)
	{
		//if(y-win_rad >= 0 && y+win_rad < cols) 
		{
			s_gray_tile[x+(i+win_rad)*cols] = d_right_gray[pixel_idx + cols*i];
			s_gray_tile[x+tile_pitch+(i+win_rad)*cols] = d_left_gray[pixel_idx + cols*i];
		}
	}

/*	const int num_disp = max_disp - min_disp + 1;
	for(int d=0; d<num_disp; d++)
	{
		const int idx = d*img_size_pad_rows+y*cols+x;
		d_left_cost_vol[idx] = bad_cost;
		d_right_cost_vol[idx] = bad_cost;

	}
*/
	__syncthreads();


	if(x+win_rad>=cols || x-win_rad<0 || y+win_rad>=rows || y-win_rad<0) return;

	const float N = (float)((2*win_rad+1)*(2*win_rad+1));

	float cost = 0.f;
#ifdef ZNCC
	float ref_std = 0.f;
	float match_std = 0.f;
#endif

	int cv_idx, h, w, ref_nei_idx, match_nei_idx, d;

	// right image as reference first
	for(d=min_disp; d<=max_disp; d++)
	{
		//const int cv_idx = (d-min_disp)*cols*rows + y*cols + x;
		cv_idx = (d-min_disp)*img_size_pad_rows + y*cols + x;
		

		// "+1" is is for forward x gradient
		if(d + x + win_rad + 1 < cols) 
		{
			cost = 0.f;

#ifdef ZNCC
			if(d == min_disp) ref_std = 0.f;
			match_std = 0.f;
#endif

			for(h=-win_rad; h<=win_rad; h++)
			{
				for(w=-win_rad; w<=win_rad; w++)
				{
					ref_nei_idx = x + cols*(win_rad+h) + w;

					match_nei_idx = x + cols*(win_rad+h) + d + w + tile_pitch;
#ifdef AD
					cost += fminf(fabsf(s_gray_tile[ref_nei_idx] - s_gray_tile[match_nei_idx]), color_truncation);	//truncated AD 
#endif

#ifdef AGD
					cost += 0.1f*fminf(fabsf(s_gray_tile[ref_nei_idx] - s_gray_tile[match_nei_idx]), color_truncation)
					+ 0.9f*fmin(fabsf(s_gray_tile[ref_nei_idx+1]-s_gray_tile[ref_nei_idx]-(s_gray_tile[match_nei_idx+1]-s_gray_tile[match_nei_idx])), gradient_truncation);	
#endif

#ifdef ZNCC		
					cost += s_gray_tile[ref_nei_idx]*s_gray_tile[match_nei_idx];
					if(d == min_disp) ref_std += s_gray_tile[ref_nei_idx]*s_gray_tile[ref_nei_idx];
					match_std += s_gray_tile[match_nei_idx]*s_gray_tile[match_nei_idx];
#endif
				}
			}

#ifdef ZNCC
			cost = -cost/(N*sqrtf(ref_std/N)*sqrtf(match_std/N));
#endif
			d_right_cost_vol[cv_idx] = cost;
			d_left_cost_vol[cv_idx+d] = cost;
		}
		else
		{
			d_right_cost_vol[cv_idx] = N*bad_cost;
		}			
	}

#if 1
	for(d=min_disp; d<=max_disp; d++)
	{
		if( x-d < 0)
		{
			cv_idx = (d-min_disp)*img_size_pad_rows + y*cols + x;
			d_left_cost_vol[cv_idx] = N*bad_cost;
		}
	}
#endif


	// left image as reference
/*	for(d=min_disp; d<=max_disp; d++)
	{
		
		//const int cv_idx = (d-min_disp)*cols*rows + y*cols + x;
		cv_idx = (d-min_disp)*img_size_pad_rows + y*cols + x;

		if(x - d - win_rad >= 0) 
		{

			cost = 0.f;
	#ifdef ZNCC	
			if(d == min_disp) ref_std = 0.f;
			match_std = 0.f;
	#endif

			for(h=-win_rad; h<=win_rad; h++)
			{
				for(w=-win_rad; w<=win_rad; w++)
				{
					ref_nei_idx = x + cols*(win_rad+h) + w + tile_pitch;

					match_nei_idx = x + cols*(win_rad+h) - d + w;

	#ifdef AD
					cost += fminf(fabsf(s_gray_tile[ref_nei_idx] - s_gray_tile[match_nei_idx]), color_truncation);	//truncated AD
	#endif

	#ifdef AGD
					cost += 0.1f*fminf(fabsf(s_gray_tile[ref_nei_idx] - s_gray_tile[match_nei_idx]), color_truncation)
						+ 0.9f*fmin(fabsf(s_gray_tile[ref_nei_idx+1]-s_gray_tile[ref_nei_idx]-(s_gray_tile[match_nei_idx+1]-s_gray_tile[match_nei_idx])), gradient_truncation);	
	#endif

	#ifdef ZNCC							
					cost += s_gray_tile[ref_nei_idx]*s_gray_tile[match_nei_idx];
					if(d == min_disp) ref_std += s_gray_tile[ref_nei_idx]*s_gray_tile[ref_nei_idx];
					match_std += s_gray_tile[match_nei_idx]*s_gray_tile[match_nei_idx];
	#endif
				}
			}

	#ifdef ZNCC
			cost = -cost/(N*sqrtf(ref_std/N)*sqrtf(match_std/N));
	#endif

			d_left_cost_vol[cv_idx] = cost;	

		}
		else
		{
			d_left_cost_vol[cv_idx] = N*bad_cost;
		}
	}*/
}

//pixel wise BGR truncated AGD cost
__global__ void buildCostVolumeSharedMemoryBGR(float* d_left_bgr, float* d_right_bgr, float* d_left_cost_vol, float* d_right_cost_vol, 
						const int rows, const int cols, const int min_disp, const int max_disp, const int img_size_pad_rows)
{
	const int x = threadIdx.x;
        const int y = blockIdx.x;

	const float color_truncation = 7.0f;
	const float gradient_truncation = 2.0f;
	const float bad_cost = 3.0f;

	extern __shared__ float s_bgr_tile[];

	const int pixel_idx = 3*(y*cols+x);
	const int tile_pitch = 3*cols;

	int i, cv_idx, ref_nei_idx, match_nei_idx, d;

	// move global image tile to shared memory
	for(i=0; i<3; i++)	//3 channels
	{
		s_bgr_tile[3*x+i] = d_right_bgr[pixel_idx + i];
		s_bgr_tile[3*x+i+tile_pitch] = d_left_bgr[pixel_idx + i];
	}

	__syncthreads();


	float cost, color_l1, gray_gradient_l1, ref_gray, match_gray;

	// right image as reference first
	for(d=min_disp; d<=max_disp; d++)
	{
		//const int cv_idx = (d-min_disp)*cols*rows + y*cols + x;
		cv_idx = (d-min_disp)*img_size_pad_rows + y*cols + x;
		
		// "+1" is is for forward x gradient
		if(d + x + 1 < cols) 
		{
			cost = 0.f;
			
			ref_nei_idx = 3*x;

			match_nei_idx = 3*(x + d) + tile_pitch;

			color_l1 = 0.0f;

			for(i=0; i<3; i++) color_l1 += fabsf(s_bgr_tile[ref_nei_idx+i]-s_bgr_tile[match_nei_idx+i]);

			// current pixel
			ref_gray = 0.114f*s_bgr_tile[ref_nei_idx]+0.587f*s_bgr_tile[ref_nei_idx+1]+0.299f*s_bgr_tile[ref_nei_idx+2];
			match_gray = 0.114f*s_bgr_tile[match_nei_idx]+0.587f*s_bgr_tile[match_nei_idx+1]+0.299f*s_bgr_tile[match_nei_idx+2];
			gray_gradient_l1 = match_gray-ref_gray;

			// next pixel
			ref_gray = 0.114f*s_bgr_tile[ref_nei_idx+3]+0.587f*s_bgr_tile[ref_nei_idx+4]+0.299f*s_bgr_tile[ref_nei_idx+5];
			match_gray = 0.114f*s_bgr_tile[match_nei_idx+3]+0.587f*s_bgr_tile[match_nei_idx+4]+0.299f*s_bgr_tile[match_nei_idx+5];
			gray_gradient_l1 += ref_gray-match_gray;
		
			cost += 0.11f*fminf(color_l1*0.33333333333, color_truncation) + 0.89f*fmin(fabsf(gray_gradient_l1), gradient_truncation);	

			d_right_cost_vol[cv_idx] = cost;
			d_left_cost_vol[cv_idx+d] = cost;
		}
		else
			d_right_cost_vol[cv_idx] = bad_cost;

		if( x-d < 0 ) d_left_cost_vol[cv_idx] = bad_cost;			
	}
}


__global__ void buildCostVolume(float* d_left_gray, float* d_right_gray, float* d_left_cost_vol, float* d_right_cost_vol, const int rows, const int cols, 
				const int min_disp, const int max_disp, const int win_rad, const int img_size_pad_rows)
{

	const int x = blockIdx.x*blockDim.x + threadIdx.x;
        const int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int z = blockIdx.z;

	// ignore borders
	if(x+win_rad>=cols || x-win_rad<0 || y+win_rad>=rows || y-win_rad<0 ) return;

	// right image as reference first
	int match_row_idx = x + z;

	int cv_idx, ref_idx, match_idx, ref_nei_idx, match_nei_idx, h, w;

	float cost;

	const float truncate_value = 4.f;//0.01f;

	cv_idx = z*img_size_pad_rows+y*cols+x;

	if(match_row_idx < cols) 
	{
		if(match_row_idx+win_rad < cols) 
		{

			ref_idx = y*cols + x;

			match_idx = y*cols + match_row_idx;

			//aggregate cost
			cost = 0.f;
			for(h=-win_rad; h<=win_rad; h++)
			{
				for(w=-win_rad; w<=win_rad; w++)
				{
					ref_nei_idx = ref_idx+w+h*cols;

					match_nei_idx = match_idx+w+h*cols;

					cost += fminf(fabsf(d_right_gray[ref_nei_idx] - d_left_gray[match_nei_idx]), truncate_value);
					//cost += fabsf(d_right_gray[ref_nei_idx] - d_left_gray[match_nei_idx]);
				}
			}

			d_right_cost_vol[cv_idx] = cost;

			d_left_cost_vol[cv_idx+z-min_disp] = cost;
		}
	}




/*	

	match_row_idx = x - z;

	if(match_row_idx < 0) return;

	if(match_row_idx - win_rad < 0) return;

	match_idx = y*cols + match_row_idx;

	cost = 0.f;
	for(h=-win_rad; h<=win_rad; h++)
	{
		for(w=-win_rad; w<=win_rad; w++)
		{
			ref_nei_idx = ref_idx+w+h*cols;

			match_nei_idx = match_idx+w+h*cols;

			cost += fminf(fabsf(d_left_gray[ref_nei_idx] - d_right_gray[match_nei_idx]), truncate_value);
		}
	}

	d_left_cost_vol[cv_idx] = cost;*/
}

__global__ void selectDisparityTexture(const bool right_side, float* d_disp, const int rows, const int rows_pad, const int cols, const int min_disp, const int max_disp, const int img_size_pad_rows)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
        const int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int num_disp = max_disp - min_disp + 1;

	if(x>=cols || y>=rows) return;

	float x_f = (float)x;
	float y_f = (float)y;

	const int pixel_idx = y*cols + x;

	

	float min_cost = 1e10f;

	float cur_cost;

	int best_disp = -1;

	for(int d=0; d<num_disp; d++)
	{
		cur_cost = right_side ? tex3D(tex_right_cost_vol, x_f+0.5f, y_f+0.5f, (float)d+0.5f) : tex3D(tex_left_cost_vol, x_f+0.5f, y_f+0.5f, (float)d+0.5f);

		if(cur_cost < min_cost)	
		{
			min_cost = cur_cost;
			best_disp = d;
		}
	}

	if(best_disp == -1)
	{
		d_disp[pixel_idx] = 0.0f;
		return;
	}


/*
	const int d_pitch = img_size_pad_rows;//cols*rows;	
	const int idx = best_disp*d_pitch+pixel_idx;
	float pre_cost = best_disp == 0 ? 0.0f : (right_side ? tex3D(tex_right_cost_vol, x_f, y_f, (float)(best_disp-1)) : tex3D(tex_left_cost_vol, x_f, y_f, (float)(best_disp-1)));
	cur_cost = right_side ? tex3D(tex_right_cost_vol, x_f, y_f, (float)(best_disp)) : tex3D(tex_left_cost_vol, x_f, y_f, (float)(best_disp));
	float next_cost = best_disp == num_disp-1 ? 0.0f : (right_side ? tex3D(tex_right_cost_vol, x_f, y_f, (float)(best_disp+1)) : tex3D(tex_left_cost_vol, x_f, y_f, (float)(best_disp+1)));

	float subpixel_update = (next_cost-pre_cost)*0.5f/(next_cost-2.0f*cur_cost+pre_cost);

	if(fabsf(subpixel_update) < 1.0f)
		d_disp[pixel_idx] = (float)best_disp + min_disp - subpixel_update;	//minus if disparity, plus if depth
	else*/
		d_disp[pixel_idx] = (float)best_disp + min_disp;
}

__global__ void selectDisparity(float* d_cost_vol, float* d_disp, const int rows, const int cols, const int min_disp, const int max_disp, const int img_size_pad_rows)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
        const int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int num_disp = max_disp - min_disp + 1;

	if(x>=cols || y>=rows) return;

	const int pixel_idx = y*cols + x;

	const int d_pitch = img_size_pad_rows;//cols*rows;

	float min_cost = 1e10f;

	float cur_cost;

	int best_disp = -1;

	for(int d=0; d<num_disp; d++)
	{
		const int idx = d*d_pitch+pixel_idx;
		
		cur_cost = d_cost_vol[idx];

		if(cur_cost < min_cost)	
		{
			min_cost = cur_cost;
			best_disp = d;
		}
	}

	if(best_disp == -1)
	{
		d_disp[pixel_idx] = 0.0f;
		return;
	}


	const int idx = best_disp*d_pitch+pixel_idx;
	float pre_cost = best_disp == 0 ? 0.0f : d_cost_vol[idx-d_pitch];
	cur_cost = d_cost_vol[idx];
	float next_cost = best_disp == num_disp-1 ? 0.0f : d_cost_vol[idx+d_pitch];

	float subpixel_update = (next_cost-pre_cost)*0.5f/(next_cost-2.0f*cur_cost+pre_cost);

	if(fabsf(subpixel_update) < 1.0f)
		d_disp[pixel_idx] = (float)best_disp + min_disp - subpixel_update;	//minus if disparity, plus if depth
	else
		d_disp[pixel_idx] = (float)best_disp + min_disp;
}


__global__ void stereoMatching(float* dRDispPtr, float* dRPlanes, float* dLDispPtr, float* dLPlanes,
                                float* dLCost, float* dRCost, int cols, int rows, int winRadius,
                                curandState* states, int iteration, float maxDisp, 
				cudaTextureObject_t lR_to, cudaTextureObject_t lG_to, cudaTextureObject_t lB_to,
				cudaTextureObject_t lGray_to, cudaTextureObject_t lGrad_to, cudaTextureObject_t rR_to,
				cudaTextureObject_t rG_to, cudaTextureObject_t rB_to, cudaTextureObject_t rGray_to,
				cudaTextureObject_t rGrad_to, bool VIEW_PROPAGATION=true, bool PLANE_REFINE=true)
{
        const int x = blockIdx.x*blockDim.x + threadIdx.x;
        const int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x>=cols || y>=rows) return;

        float u = ((float)x+0.5f)/(float)(cols);
        float v = ((float)y+0.5f)/(float)(rows);

        const int idx = y*cols + x;

        // evaluate disparity of current pixel (based on right)
        float min_cost, cost, tmp_disp, delta_disp, cur_disp, s, nx, ny, nz, norm;
        int tmp_idx, i, j;
        int new_x;
	bool even_iteration = iteration%2==0 ? true : false;

	// even iteration process left then right, odd iteration opposite
	if(even_iteration) goto LEFT;
	
        //--------------------------------------------
RIGHT:
        min_cost = 1e10f;
        // spatial propagation
        for(i=-1; i<=1; i++)
        {
                for(j=-1; j<=1; j++)
                {
                        if(x+j<0 || x+j>=cols || y+i<0 || y+i>=rows) continue;

                        tmp_idx = idx + i*cols + j;

                        tmp_disp = dRDispPtr[tmp_idx];

                        tmp_idx *= 2;
		
			nx = dRPlanes[tmp_idx];	ny = dRPlanes[tmp_idx+1]; nz = sqrtf(1.0f-nx*nx-ny*ny);

                        cost =  evaluateCost(lR_to, lG_to, lB_to, lGrad_to, rR_to, rG_to, rB_to, rGrad_to, lGray_to, rGray_to,
						u, v, x, y, tmp_disp, cols, rows, 0, maxDisp, winRadius, nx, ny, nz, 1);

                        // base 0 left, 1 right
                        if(cost < min_cost)
                        {
                                min_cost = cost;
				dRDispPtr[idx] = tmp_disp;
				dRPlanes[idx*2] = nx;
				dRPlanes[idx*2 + 1] = ny;
				dRCost[idx] = min_cost;
                        }
                }
        }

        // view propagation
        if(VIEW_PROPAGATION)
        {
                new_x = (int)lroundf(dRDispPtr[idx]) + x;

                // check if in range
                if(new_x>=0 && new_x<cols)
                {
                        tmp_idx = idx + new_x - x;
                        tmp_disp = dLDispPtr[tmp_idx];

                        tmp_idx *= 2;

			nx = dRPlanes[tmp_idx];	ny = dRPlanes[tmp_idx+1]; nz = sqrtf(1.0f-nx*nx-ny*ny);

                        cost = evaluateCost(lR_to, lG_to, lB_to, lGrad_to, rR_to, rG_to, rB_to, rGrad_to, lGray_to, rGray_to,
						u, v, x, y, tmp_disp, cols, rows, 0, maxDisp, winRadius, nx, ny, nz, 1);

                        if(cost < min_cost)
                        {
                                min_cost = cost;
                               	dRCost[idx] = min_cost;
                                dRDispPtr[idx] = tmp_disp;
                                dRPlanes[2*idx] = nx;
                                dRPlanes[2*idx+1] = ny;
                        }
                }
        }


        // right plane refinement
        if(PLANE_REFINE)
        {
                s = 1.0f;

                for(delta_disp=maxDisp*0.5f; delta_disp>=0.1f; delta_disp *= 0.5f)
                {
                        cur_disp = dRDispPtr[idx];

                        cur_disp += (curand_uniform(&states[idx])*2.0f-1.0f)*delta_disp;

                        if(cur_disp<0.0f || cur_disp>(float)maxDisp)
                        {
                                s *= 0.5;
                                continue;
                        }

			nx = dRPlanes[idx*2];
			ny = dRPlanes[idx*2+1];
			nz = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + sqrtf(1.0f-nx*nx-ny*ny);
                        nx = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + nx;
                        ny = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + ny;
                       
                        
			//normalize
                        norm = sqrtf(nx*nx+ny*ny+nz*nz);

			nx /= norm;
			ny /= norm;
			nz /= norm;
			nz = fabsf(nz);

			if(isinf(nx)!=0 || isinf(ny)!=0 || isinf(nz)!=0)
			{
				s *= 0.5f;
				continue;
			}

			cost = evaluateCost(lR_to, lG_to, lB_to, lGrad_to, rR_to, rG_to, rB_to, rGrad_to, lGray_to, rGray_to,
						u, v, x, y, cur_disp, cols, rows, 0, maxDisp, winRadius,
						nx, ny, nz, 1);


			if(cost < min_cost)
			{
				min_cost = cost;
				dRCost[idx] = min_cost;
				dRDispPtr[idx] = cur_disp;
				dRPlanes[idx*2] = nx;
				dRPlanes[idx*2 + 1] = ny;
			}

                        s *= 0.5;
                }
        }

	if(even_iteration) return;


        //------------------------------------------------------------
LEFT:
        min_cost = 1e10f;

        // spatial  propagation
        for(i=-1; i<=1; i++)
        {
                for(j=-1; j<=1; j++)
                {
                        if(x+j<0 || x+j>=cols || y+i<0 || y+i>=rows) continue;

                        tmp_idx = idx + i*cols + j;

                        tmp_disp = dLDispPtr[tmp_idx];

                        tmp_idx *= 2;

			nx = dRPlanes[tmp_idx];	ny = dRPlanes[tmp_idx+1]; nz = sqrtf(1.0f-nx*nx-ny*ny);

                        cost =  evaluateCost(lR_to, lG_to, lB_to, lGrad_to, rR_to, rG_to, rB_to, rGrad_to, lGray_to, rGray_to,
						u, v, x, y, tmp_disp, cols, rows, 0, maxDisp, winRadius, nx, ny, nz, 0);

                        if(cost < min_cost)
                        {
                                min_cost = cost;
				dLDispPtr[idx] = tmp_disp;
				dLPlanes[idx*2] = nx;
				dLPlanes[idx*2 + 1] = ny;
				dLCost[idx] = min_cost;
                        }
                }
        }

        // view propagation
        if(VIEW_PROPAGATION)
        {
                new_x = x - (int)lroundf(dLDispPtr[idx]);

                // check if in range
                if(new_x>=0 && new_x<cols)
                {
                        tmp_idx = idx + new_x - x;
                        tmp_disp = dRDispPtr[tmp_idx];

                        tmp_idx *= 2;

			nx = dRPlanes[tmp_idx]; ny = dRPlanes[tmp_idx+1]; nz = sqrtf(1.0f-nx*nx-ny*ny);

                        cost = evaluateCost(lR_to, lG_to, lB_to, lGrad_to, rR_to, rG_to, rB_to, rGrad_to, lGray_to, rGray_to,
						u, v, x, y, tmp_disp, cols, rows, 0, maxDisp, winRadius,
                                        	nx, ny, nz, 0);

                        if(cost < min_cost)
                        {
                                min_cost = cost;
				dLCost[idx] = min_cost;
                                dLDispPtr[idx] = tmp_disp;
                                dLPlanes[2*idx] = nx;
                                dLPlanes[2*idx+1] = dRPlanes[tmp_idx+1];
                        }
                }
        }

        // left plane refinement
        // exponentially reduce disparity search range
        if(PLANE_REFINE)
        {
                s = 1.0f;

                for(delta_disp=maxDisp*0.5f; delta_disp>=0.1f; delta_disp *= 0.5f)
                {
                        cur_disp = dLDispPtr[idx];

                        cur_disp += (curand_uniform(&states[idx])*2.0f-1.0f)*delta_disp;

                        if(cur_disp<0.0f || cur_disp>(float)maxDisp)
                        {
                                s *= 0.5;
                                continue;
                        }

			nx = dLPlanes[idx*2];
			ny = dLPlanes[idx*2+1];
                        nz = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + sqrtf(1.0f-nx*nx-ny*ny);	
                        nx = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + nx;
                        ny = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + ny;


                        //normalize
                        norm = sqrtf(nx*nx+ny*ny+nz*nz);

			nx /= norm;
			ny /= norm;
			nz /= norm;
			nz = fabsf(nz);

			if( isinf(nx)!=0 || isinf(ny)!=0 || isinf(nz)!=0 )
			{
				s *= 0.5f;
				continue;
			}

			cost = evaluateCost(lR_to, lG_to, lB_to, lGrad_to, rR_to, rG_to, rB_to, rGrad_to, lGray_to, rGray_to,
						u, v, x, y, cur_disp, cols, rows, 0, maxDisp, winRadius,
						nx, ny, nz, 0);


			if(cost < min_cost)
			{
				min_cost = cost;
				dLCost[idx] = min_cost;
				dLDispPtr[idx] = cur_disp;
				dLPlanes[idx*2] = nx;
				dLPlanes[idx*2 + 1] = ny;
			}

                        s *= 0.5;
                }
        }

	if(even_iteration) goto RIGHT;
}



__global__ void imgGradient_huber( int cols, int rows, cudaTextureObject_t lGray_to, cudaTextureObject_t rGray_to, 
				   float* lGradX, float* rGradX, float* lGradY, float* rGradY,
				   float* lGradXY, float* rGradXY, float* lGradYX, float* rGradYX)
{
        const int x = blockIdx.x*blockDim.x + threadIdx.x;
        const int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x >= cols || y>= rows) return;

        const int idx = y*cols+x;
	const float du = 1.0f/((float)cols);
	const float dv = 1.0f/((float)rows);
        const float u = ((float)x+0.5f)*du;
        const float v = ((float)y+0.5f)*dv;

	const float s = 1.0f/sqrtf(8.0f);

	// horizontal sobel	
	lGradX[idx] = 2.0f*(tex2D<float>(lGray_to, u+du, v)-tex2D<float>(lGray_to, u-du, v))
		      + (tex2D<float>(lGray_to, u+du, v+dv)-tex2D<float>(lGray_to, u-du, v+dv))
		      + (tex2D<float>(lGray_to, u+du, v-dv)-tex2D<float>(lGray_to, u-du, v-dv));

	rGradX[idx] = 2.0f*(tex2D<float>(rGray_to, u+du, v)-tex2D<float>(rGray_to, u-du, v))
		      + (tex2D<float>(rGray_to, u+du, v+dv)-tex2D<float>(rGray_to, u-du, v+dv))
		      + (tex2D<float>(rGray_to, u+du, v-dv)-tex2D<float>(rGray_to, u-du, v-dv));

	// vertical sobel
	lGradY[idx] = 2.0f*(tex2D<float>(lGray_to, u, v+dv)-tex2D<float>(lGray_to, u, v-dv))
		      + (tex2D<float>(lGray_to, u+du, v+dv)-tex2D<float>(lGray_to, u+du, v-dv))
		      + (tex2D<float>(lGray_to, u-du, v+dv)-tex2D<float>(lGray_to, u-du, v-dv));

	rGradY[idx] = 2.0f*(tex2D<float>(rGray_to, u, v+dv)-tex2D<float>(rGray_to, u, v-dv))
		      + (tex2D<float>(rGray_to, u+du, v+dv)-tex2D<float>(rGray_to, u+du, v-dv))
		      + (tex2D<float>(rGray_to, u-du, v+dv)-tex2D<float>(rGray_to, u-du, v-dv));

	// central difference 45 deg
	lGradXY[idx] = (tex2D<float>(lGray_to, u+du, v+dv)-tex2D<float>(lGray_to, u-du, v-dv))*s;
	rGradXY[idx] = (tex2D<float>(rGray_to, u+du, v+dv)-tex2D<float>(rGray_to, u-du, v-dv))*s;

	// central difference 135 deg
	lGradYX[idx] = (tex2D<float>(lGray_to, u-du, v+dv)-tex2D<float>(lGray_to, u+du, v-dv))*s;
	rGradYX[idx] = (tex2D<float>(rGray_to, u-du, v+dv)-tex2D<float>(rGray_to, u+du, v-dv))*s;
	
}

__global__ void gradient(float* lGradPtr, float*rGradPtr, cudaTextureObject_t lGray_to, cudaTextureObject_t rGray_to, int cols, int rows)
{
        const int x = blockIdx.x*blockDim.x + threadIdx.x;
        const int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x >= cols || y>= rows) return;

        const int idx = y*cols+x;
        const float u = ((float)x+0.5f)/(float)(cols);
        const float v = ((float)y+0.5f)/(float)(rows);

        lGradPtr[idx] = 0.5f*(tex2D<float>(lGray_to, u + 1.0f/(float)(cols), v ) - tex2D<float>(lGray_to, u - 1.0f/(float)(cols), v) );
        rGradPtr[idx] = 0.5f*(tex2D<float>(rGray_to, u + 1.0f/(float)(cols), v ) - tex2D<float>(rGray_to, u - 1.0f/(float)(cols), v) );
}

// initialize random states
__global__ void init(unsigned int seed, curandState_t* states, int cols, int rows)
{
        const int x = blockIdx.x*blockDim.x + threadIdx.x;
        const int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x >= cols || y>= rows) return;

        const int idx = y*cols+x;
        curand_init(seed, idx, 0, &states[idx]);
}

// initialize random uniformally distributed plane normals
__global__ void init_plane_normals_disp(float* dRPlanes, float* d_r_disp, float max_disp, curandState_t* states, int cols, int rows)
{
        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x >= cols || y>= rows) return;

        int idx = y*cols+x;

	d_r_disp[idx] *= max_disp;

        float x1, x2;
	
	while(true)
	{
	        x1 = curand_uniform(&states[idx])*2.0f - 1.0f;
	        x2 = curand_uniform(&states[idx])*2.0f - 1.0f;

	        if( x1*x1 + x2*x2 < 1.0f ) break;
	}

	int i = idx*2;
	dRPlanes[i] = 2.0f*x1*sqrtf(1.0f - x1*x1 - x2*x2);
	dRPlanes[i+1] = 2.0f*x2*sqrtf(1.0f - x1*x1 - x2*x2);

	//dRPlanes[i] = dRPlanes[i+1] = 0.0f;
}

__global__ void leftRightCheck(float* dRDispPtr, float* dLDispPtr, int* dLOccludeMask, int* dROccludeMask, int cols, int rows, float minDisp, float maxDisp, int iter)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x >= cols || y>= rows) return;	

	int idx = y*cols+x;

	const float thresh = minDisp;

	float tmp_disp = dLDispPtr[idx];

	int tmp_idx = x - (int)lroundf(tmp_disp);

	const float d_thresh = 1.0f;

	if( (tmp_disp<=0.0f) || (tmp_disp>maxDisp) ||(tmp_idx < 0) || tmp_idx>=cols || fabsf(tmp_disp - dRDispPtr[idx + tmp_idx - x]) > d_thresh
		|| tmp_disp < thresh )
	{
		if(iter == 1) dLDispPtr[idx] = 0.0f;
		dLOccludeMask[idx] = 1;
	}
	else
		dLOccludeMask[idx] = 0;

	tmp_disp = dRDispPtr[idx];
	tmp_idx = x + (int)lroundf(tmp_disp);

	if( (tmp_disp<=0.0f) || (tmp_disp>maxDisp) ||(tmp_idx < 0) || tmp_idx>=cols || fabsf(tmp_disp - dLDispPtr[idx + tmp_idx - x]) > d_thresh
		|| tmp_disp < thresh )
	{
		if(iter == 1) dRDispPtr[idx] = 0.0f;
		dROccludeMask[idx] = 1;
	}
	else
		dROccludeMask[idx] = 0;
	
}

__global__ void fillInOccluded(float* dLDisp, float* dRDisp, float* dLPlanes, float* dRPlanes,
				float* dLCost, float* dRCost, int* dLOccludeMask, int* dROccludeMask, int cols, int rows, 
				int winRadius, float maxDisp, int iteration,
 				cudaTextureObject_t lR_to, cudaTextureObject_t lG_to, cudaTextureObject_t lB_to,
				cudaTextureObject_t lGray_to, cudaTextureObject_t lGrad_to, cudaTextureObject_t rR_to,
				cudaTextureObject_t rG_to, cudaTextureObject_t rB_to, cudaTextureObject_t rGray_to,
				cudaTextureObject_t rGrad_to)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x>cols-1 || y>rows-1 ) return;	

	int idx = y*cols+x;	

	float xf = (float)x;
	float yf = (float)y;
	
	float nx, ny, nz, af, bf, cf, disp, tmp_disp;
	float best_nx, best_ny;
	int i = 1;
	disp = 2*maxDisp;
	float u = (xf+0.5f)/(float)(cols);
	float v = (yf+0.5f)/(float)(rows);
	int tmp_idx;
	float min_cost = 1e10f;
	float tmp_cost;

	float best_disp = 0.0f;

	if(iteration == 0)
	{
		// left disparity search horizontally
		best_disp = 0.0f;
		tmp_disp = 0.0f;
		for(int i=-cols/4; i<=cols/4; i++)
		{
			if(i+x<0 || i+x>=cols)
				continue;
			
			tmp_idx = idx + i;

			if(dLOccludeMask[tmp_idx] == 1)
				continue;

			tmp_disp = dLDisp[tmp_idx];

			if(tmp_disp == best_disp)
				continue;

			tmp_idx *= 2;

			nx = dLPlanes[tmp_idx];
			ny = dLPlanes[tmp_idx+1];
			nz = sqrtf(1.0f - nx*nx - ny*ny);

			tmp_cost =  evaluateCost(lR_to, lG_to, lB_to, lGrad_to, rR_to, rG_to, rB_to, rGrad_to, lGray_to, rGray_to,
					u, v, x, y, tmp_disp, cols, rows, 0, maxDisp, winRadius,
					nx, ny, nz, 0);

	
			if(tmp_cost  < min_cost)
			{
				min_cost = tmp_cost;
				best_disp = tmp_disp;
			}
		}

		dLDisp[idx] = best_disp;
		dLCost[idx] = min_cost;

		// right disparity search horizontally
		min_cost = 1e10f;
		best_disp = 0.0f;
		for(int i=-cols/4; i<=cols/4; i++)
		{
			if(i+x<0 || i+x>=cols)
				continue;
			
			tmp_idx = idx + i;

			if(dROccludeMask[tmp_idx] == 1)
				continue;

			tmp_disp = dRDisp[tmp_idx];

			if(tmp_disp == best_disp)
				continue;

			tmp_idx *= 2;

			nx = dRPlanes[tmp_idx];
			ny = dRPlanes[tmp_idx+1];
			nz = sqrtf(1.0f - nx*nx - ny*ny);

			
			tmp_cost =  evaluateCost(lR_to, lG_to, lB_to, lGrad_to, rR_to, rG_to, rB_to, rGrad_to, lGray_to, rGray_to,
						u, v, x, y, tmp_disp, cols, rows, 0, maxDisp, winRadius,
						nx, ny, nz, 1);
	
	
			if(tmp_cost  < min_cost)
			{
				min_cost = tmp_cost;
				best_disp = tmp_disp;
			}
		}

		dRDisp[idx] = best_disp;

	}

	if(iteration == 1 )
	{
		if(dLOccludeMask[idx] != 0)
		{
			best_disp =1e10f;
			// search right
			i = 1;
			while( x+i < cols )
			{
				if( dLOccludeMask[idx+i] == 0 )
				{
					nx = dLPlanes[(idx+i)*2];
					ny = dLPlanes[(idx+i)*2+1];
					nz = sqrtf(1.0f - nx*nx - ny*ny);

					// af = -nx/nz, bf = -ny/nz, cf = (nx*x+ny*y+nz*disp)/nz			
					af = nx/nz*(-1.0f);
					bf = ny/nz*(-1.0f);
					cf = (nx*(xf+(float)i) + ny*yf + nz*dLDisp[idx+i])/nz;
	      
					if(isinf(af)!=0 || isinf(bf)!=0 || isinf(cf)!=0)
					{
						i++;
						continue;
					}

					tmp_disp = af*xf + bf*yf + cf;
				
					if(tmp_disp>0.0f && tmp_disp <= maxDisp)
					{
						best_disp = tmp_disp;
						best_nx = nx;
						best_ny = ny;
						break;
					}
				}

				i++;
			}

			//search left for the nearest valid(none zero) disparity
			i = 1;
			while( x-i>=0 )
			{
				// valid disparity
				if( dLOccludeMask[idx-i] == 0)
				{
					nx = dLPlanes[(idx-i)*2];
					ny = dLPlanes[(idx-i)*2+1];
					nz = sqrtf(1.0f - nx*nx -ny*ny);
		
					af = nx/nz*(-1.0f);
					bf = ny/nz*(-1.0f);
					cf = (nx*(xf-(float)i) + ny*yf + nz*dLDisp[idx-i])/nz;
			
					if( isinf(af)!=0 || isinf(bf)!=0 || isinf(cf)!=0 )
					{
						i++;
						continue;
					}
		
					tmp_disp = af*xf + bf*yf + cf;
				
					if(tmp_disp>0.0f && tmp_disp<=maxDisp && tmp_disp < best_disp)
					{
						best_disp = tmp_disp;
						best_nx = nx;
						best_ny = ny;
						break;
					}
				}
			
				i++;
			}


			if(best_disp != 1e10f)
			{
				dLDisp[idx] = best_disp;
				dLPlanes[2*idx] = best_nx;
				dLPlanes[2*idx+1] = best_ny;
			}
		}

		if(dROccludeMask[idx] != 0)
		{
			best_disp = 1e10f;
			// search right
			i = 1;
			while( x+i < cols )
			{
				if( dROccludeMask[idx+i] == 0 )
				{
					nx = dRPlanes[(idx+i)*2];
					ny = dRPlanes[(idx+i)*2+1];
					nz = sqrtf(1.0f - nx*nx - ny*ny);

					// af = -nx/nz, bf = -ny/nz, cf = (nx*x+ny*y+nz*disp)/nz			
					af = nx/nz*(-1.0f);
					bf = ny/nz*(-1.0f);
					cf = (nx*(xf+(float)i) + ny*yf + nz*dRDisp[idx+i])/nz;
	      
					if(isinf(af)!=0 || isinf(bf)!=0 || isinf(cf)!=0)
					{
						i++;
						continue;
					}

					tmp_disp = af*xf + bf*yf + cf;
				
					if(tmp_disp>0.0f && tmp_disp <= maxDisp)
					{
						best_disp = tmp_disp;
						best_nx = nx;
						best_ny = ny;
						break;
					}
				}

				i++;
			}

			//search left for the nearest valid(none zero) disparity
			i = 1;
			while( x-i>=0 )
			{
				// valid disparity
				if( dROccludeMask[idx-i] == 0)
				{
					nx = dRPlanes[(idx-i)*2];
					ny = dRPlanes[(idx-i)*2+1];
					nz = sqrtf(1.0f - nx*nx -ny*ny);
		
					af = nx/nz*(-1.0f);
					bf = ny/nz*(-1.0f);
					cf = (nx*(xf-(float)i) + ny*yf + nz*dRDisp[idx-i])/nz;
			
					if( isinf(af)!=0 || isinf(bf)!=0 || isinf(cf)!=0 )
					{
						i++;
						continue;
					}
		
					tmp_disp = af*xf + bf*yf + cf;
				
					if(tmp_disp>0.0f && tmp_disp<=maxDisp && tmp_disp < best_disp)
					{
						best_disp = tmp_disp;
						best_nx = nx;
						best_ny = ny;
						break;
					}
				}
			
				i++;
			}


			if(best_disp != 1e10f)
			{
				dRDisp[idx] = best_disp;
				dRPlanes[2*idx] = best_nx;
				dRPlanes[2*idx+1] = best_ny;
			}
		}
	}
}


__global__ void weightedMedianFilter(float* dLDisp, int* dLOccludeMask, float* dRDisp, int* dROccludeMask, 
					int cols, int rows, float maxDisp, bool normalized_intensity,
	 				cudaTextureObject_t lR_to, cudaTextureObject_t lG_to, cudaTextureObject_t lB_to,
					cudaTextureObject_t lGray_to, cudaTextureObject_t rR_to,
					cudaTextureObject_t rG_to, cudaTextureObject_t rB_to, cudaTextureObject_t rGray_to)

{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	int idx = y*cols + x;

	if( x>=cols || y >=rows) return;

	float u = ((float)x+0.5f)/(float)(cols);
	float v = ((float)y+0.5f)/(float)(rows);

	const int winRa = 10;
	const int winSize = winRa*2+1;

	float weightedMask[winSize*winSize];
	float dispMask[winSize*winSize];


	float color_diff, weight, tmp, gamma;
	float weight_sum = 0.0f;

	gamma = normalized_intensity ? 25.5f : 0.1f;
	
	int tmp_idx, invalid_count, i, j;

	if(dLOccludeMask[idx] == 1)
	{
		// apply weights
		tmp_idx = 0;
		invalid_count = 0;
		for(i=-winRa; i<=winRa; i++)
		{
			for(j=-winRa; j<=winRa; j++)
			{
	
				color_diff = sqrt(fabsf(tex2D<float>(lR_to, u+(float)j/(float)(cols), v+(float)i/(float)(rows)) - tex2D<float>(lR_to, u, v))
					          +fabsf(tex2D<float>(lG_to, u+(float)j/(float)(cols), v+(float)i/(float)(rows)) - tex2D<float>(lG_to, u, v))
				    		  +fabsf(tex2D<float>(lB_to, u+(float)j/(float)(cols), v+(float)i/(float)(rows)) - tex2D<float>(lB_to, u, v)));

				weight = expf(-(color_diff)*gamma);

				if(x+j>=0 && x+j<cols && y+i>=0 && y+i<rows)
				{
					weightedMask[tmp_idx] = weight;
					dispMask[tmp_idx] = dLDisp[idx+i*cols+j];
					weight_sum += weight;
				}
				else
				{
					weightedMask[tmp_idx] = 0.0f;
					dispMask[tmp_idx] = 0.0f;
					invalid_count++;
				}
				tmp_idx++; 
			}
		}


		// insertion sort
		for(i=1; i<winSize*winSize; i++)
		{
			for(j=i; j>0; j--)
			{
				if(dispMask[j] < dispMask [j-1])
				{
					tmp = weightedMask[j];
					weightedMask[j] = weightedMask[j-1];
					weightedMask[j-1] = tmp;

					tmp = dispMask[j];
					dispMask[j] = dispMask[j-1];
					dispMask[j-1]= tmp;
				}
			}
		}

		// 1/2 weight
		weight = 0.0f;
		for(i=0; i<winSize*winSize; i++)
		{
			weight += weightedMask[i]/weight_sum;
	
			if(weight >= 0.5f)
			{		
				dLDisp[idx] = dispMask[i];	
				break;	
			}

		}
	}

	if(dROccludeMask[idx] == 1)
	{
		// apply weights
		tmp_idx = 0;
		invalid_count = 0;
		for(i=-winRa; i<=winRa; i++)
		{
			for(j=-winRa; j<=winRa; j++)
			{
	
				color_diff = sqrt(fabsf(tex2D<float>(rR_to, u+(float)j/(float)(cols), v+(float)i/(float)(rows)) - tex2D<float>(rR_to, u, v))
					     	 +fabsf(tex2D<float>(rG_to, u+(float)j/(float)(cols), v+(float)i/(float)(rows)) - tex2D<float>(rG_to, u, v))
				    		 +fabsf(tex2D<float>(rB_to, u+(float)j/(float)(cols), v+(float)i/(float)(rows)) - tex2D<float>(rB_to, u, v)));

				weight = expf(-(color_diff)*gamma);

				if(x+j>=0 && x+j<cols && y+i>=0 && y+i<rows)
				{
					weightedMask[tmp_idx] = weight;
					dispMask[tmp_idx] = dRDisp[idx+i*cols+j];
					weight_sum += weight;
				}
				else
				{
					weightedMask[tmp_idx] = 0.0f;
					dispMask[tmp_idx] = 0.0f;
					invalid_count++;
				}
				tmp_idx++; 
			}
		}


		// insertion sort
		for(i=1; i<winSize*winSize; i++)
		{
			for(j=i; j>0; j--)
			{
				if(dispMask[j] < dispMask [j-1])
				{
					tmp = weightedMask[j];
					weightedMask[j] = weightedMask[j-1];
					weightedMask[j-1] = tmp;

					tmp = dispMask[j];
					dispMask[j] = dispMask[j-1];
					dispMask[j-1]= tmp;
				}
			}
		}

		// 1/2 weight
		weight = 0.0f;
		for(i=0; i<winSize*winSize; i++)
		{
			weight += weightedMask[i]/weight_sum;
	
			if(weight >= 0.5f)
			{		
				dRDisp[idx] = dispMask[i];	
				break;	
			}

		}
	}
	
}


void PatchMatchStereoGPU(const cv::Mat& leftImg, const cv::Mat& rightImg, int winRadius, int Dmin, int Dmax, int iteration, float scale, bool showLeftDisp, cv::Mat& leftDisp, cv::Mat& rightDisp)
{
	const int cols = leftImg.cols;
	const int rows = leftImg.rows;

	// split channels
	std::vector<cv::Mat> cvLeftBGR_v;
	std::vector<cv::Mat> cvRightBGR_v;

	cv::split(leftImg, cvLeftBGR_v);
	cv::split(rightImg, cvRightBGR_v);

	// BGR 2 grayscale
	cv::Mat cvLeftGray;
	cv::Mat cvRightGray;

	cv::cvtColor(leftImg, cvLeftGray, CV_BGR2GRAY);
	cv::cvtColor(rightImg, cvRightGray, CV_BGR2GRAY);	

	// convert to float
	cv::Mat cvLeftB_f;
	cv::Mat cvLeftG_f;
	cv::Mat cvLeftR_f;
	cv::Mat cvRightB_f;
	cv::Mat cvRightG_f;
	cv::Mat cvRightR_f;
	cv::Mat cvLeftGray_f;
	cv::Mat cvRightGray_f;	

	cvLeftBGR_v[0].convertTo(cvLeftB_f, CV_32F);
	cvLeftBGR_v[1].convertTo(cvLeftG_f, CV_32F);
	cvLeftBGR_v[2].convertTo(cvLeftR_f, CV_32F);	
	cvRightBGR_v[0].convertTo(cvRightB_f, CV_32F);	
	cvRightBGR_v[1].convertTo(cvRightG_f, CV_32F);	
	cvRightBGR_v[2].convertTo(cvRightR_f, CV_32F);	
	cvLeftGray.convertTo(cvLeftGray_f, CV_32F);
	cvRightGray.convertTo(cvRightGray_f, CV_32F);
		
	float* leftRImg_f = cvLeftR_f.ptr<float>(0);
	float* leftGImg_f = cvLeftG_f.ptr<float>(0);
	float* leftBImg_f = cvLeftB_f.ptr<float>(0);
	float* leftGrayImg_f = cvLeftGray_f.ptr<float>(0);
	float* rightRImg_f = cvRightR_f.ptr<float>(0);
	float* rightGImg_f = cvRightG_f.ptr<float>(0);
	float* rightBImg_f = cvRightB_f.ptr<float>(0);
	float* rightGrayImg_f = cvRightGray_f.ptr<float>(0);

	unsigned int imgSize = (unsigned int)cols*rows;

	// allocate floating disparity map, plane normals and gradient image (global memory)
	float* dRDisp = NULL;
	float* dLDisp = NULL;
	float* dRPlanes = NULL;
	float* dLPlanes = NULL;
	float* dLGrad = NULL;
	float* dRGrad = NULL;
	float* dLCost = NULL;
	float* dRCost = NULL;

	cudaMalloc(&dRDisp, imgSize*sizeof(float));
	cudaMalloc(&dLDisp, imgSize*sizeof(float));
	cudaMalloc(&dRPlanes, 2*imgSize*sizeof(float));
	cudaMalloc(&dLPlanes, 2*imgSize*sizeof(float));

	cudaMalloc(&dRCost, imgSize*sizeof(float));
	cudaMalloc(&dLCost, imgSize*sizeof(float));

	cudaMalloc(&dRGrad, imgSize*sizeof(float));
	cudaMalloc(&dLGrad, imgSize*sizeof(float));

	cudaArray* lR_ca;
	cudaArray* lG_ca;
	cudaArray* lB_ca;
	cudaArray* lGray_ca;
	cudaArray* rR_ca;
	cudaArray* rG_ca;
	cudaArray* rB_ca;
	cudaArray* rGray_ca;
	cudaArray* lGrad_ca;
	cudaArray* rGrad_ca;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaMallocArray(&lR_ca, &desc, cols, rows);
	cudaMallocArray(&lG_ca, &desc, cols, rows);
	cudaMallocArray(&lB_ca, &desc, cols, rows);
	cudaMallocArray(&lGray_ca, &desc, cols, rows);
	cudaMallocArray(&lGrad_ca, &desc, cols, rows);
	cudaMallocArray(&rR_ca, &desc, cols, rows);
	cudaMallocArray(&rG_ca, &desc, cols, rows);
	cudaMallocArray(&rB_ca, &desc, cols, rows);
	cudaMallocArray(&rGray_ca, &desc, cols, rows);
	cudaMallocArray(&rGrad_ca, &desc, cols, rows);

	cudaMemcpyToArray(lR_ca, 0, 0, leftRImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(lG_ca, 0, 0, leftGImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(lB_ca, 0, 0, leftBImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(lGray_ca, 0, 0, leftGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rR_ca, 0, 0, rightRImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rG_ca, 0, 0, rightGImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rB_ca, 0, 0, rightBImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rGray_ca, 0, 0, rightGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);

	
	// texture object test
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = lR_ca;


	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.normalizedCoords = 1;

	cudaTextureObject_t lR_to = 0;
	cudaCreateTextureObject(&lR_to, &resDesc, &texDesc, NULL);


	cudaTextureObject_t lG_to = 0;
	resDesc.res.array.array = lG_ca;
	cudaCreateTextureObject(&lG_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t lB_to = 0;
	resDesc.res.array.array = lB_ca;
	cudaCreateTextureObject(&lB_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t lGray_to = 0;
	resDesc.res.array.array = lGray_ca;
	cudaCreateTextureObject(&lGray_to, &resDesc, &texDesc, NULL);


	cudaTextureObject_t rR_to = 0;
	resDesc.res.array.array = rR_ca;
	cudaCreateTextureObject(&rR_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rG_to = 0;
	resDesc.res.array.array = rG_ca;
	cudaCreateTextureObject(&rG_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rB_to = 0;
	resDesc.res.array.array = rB_ca;
	cudaCreateTextureObject(&rB_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rGray_to = 0;
	resDesc.res.array.array = rGray_ca;
	cudaCreateTextureObject(&rGray_to, &resDesc, &texDesc, NULL);

	// launch kernels
	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1)/blockSize.x, (rows + blockSize.y - 1)/blockSize.y); 

	// calculate gradient
        gradient<<<gridSize, blockSize>>>(dLGrad, dRGrad, lGray_to, rGray_to, cols, rows);

        // copy gradient back
	cudaMemcpyToArray(lGrad_ca, 0, 0, dLGrad, sizeof(float)*imgSize, cudaMemcpyDeviceToDevice);
	cudaMemcpyToArray(rGrad_ca, 0, 0, dRGrad, sizeof(float)*imgSize, cudaMemcpyDeviceToDevice);

	cudaTextureObject_t lGrad_to = 0;
	resDesc.res.array.array = lGrad_ca;
	cudaCreateTextureObject(&lGrad_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rGrad_to = 0;
	resDesc.res.array.array = rGrad_ca;
	cudaCreateTextureObject(&rGrad_to, &resDesc, &texDesc, NULL);

	StartTimer();

	// allocate memory for states
        curandState_t* states;
        cudaMalloc(&states, imgSize*sizeof(curandState_t));
        // initialize random states
        init<<<gridSize, blockSize>>>(1234, states, cols, rows);
        cudaDeviceSynchronize();
                              
        curandGenerator_t gen;
        // host CURAND
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        // set seed
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        // random initial right and left disparity
        curandGenerateUniform(gen, dLDisp, imgSize);

        // set seed
        curandSetPseudoRandomGeneratorSeed(gen, 4321ULL);
        // random initial right and left disparity
        curandGenerateUniform(gen, dRDisp, imgSize);


        // random initial right and left plane
        init_plane_normals_disp<<<gridSize, blockSize>>>(dRPlanes, dRDisp, (float)Dmax, states, cols, rows);

        init_plane_normals_disp<<<gridSize, blockSize>>>(dLPlanes, dLDisp, (float)Dmax, states, cols, rows);

	std::cout<<"Random Init:"<<GetTimer()<<std::endl;

	// result
	cv::Mat cvLeftDisp_f, cvRightDisp_f;
	cvLeftDisp_f.create(rows, cols, CV_32F);
	cvRightDisp_f.create(rows, cols, CV_32F);
	
	StartTimer();

	for(int i=0; i<iteration; i++)
	{
		stereoMatching<<<gridSize, blockSize>>>(dRDisp, dRPlanes, dLDisp, dLPlanes,
							dLCost, dRCost, cols, rows, winRadius,
							states, i, (float)Dmax, lR_to, lG_to, lB_to,
							lGray_to, lGrad_to, rR_to, rG_to, rB_to,
							rGray_to, rGrad_to);

		cudaMemcpy(cvRightDisp_f.ptr<float>(0), dRDisp, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(cvLeftDisp_f.ptr<float>(0), dLDisp, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
		cvRightDisp_f /= (float)Dmax;
		cvLeftDisp_f /= (float)Dmax;
		std::cout<<"iteration: "<<i+1<<"\n";
		cv::imshow("Left Disp", cvLeftDisp_f);
		cv::imshow("Right Disp", cvRightDisp_f);
		cv::waitKey(500);
	}
	cudaDeviceSynchronize();
	std::cout<<"Main loop:"<<GetTimer()<<std::endl;


	StartTimer();

	int* dLOccludeMask;
	int* dROccludeMask;

	cudaMalloc(&dLOccludeMask, imgSize*sizeof(int));
	cudaMalloc(&dROccludeMask, imgSize*sizeof(int));

#if 0//POST_PROCESSING

	leftRightCheck<<<gridSize, blockSize>>>(dRDisp, dLDisp, dLOccludeMask, dROccludeMask, cols, rows, (float)Dmin, (float)Dmax, 1);	//last augument: 0: don't set disp to 0, 1: set disp to 0

//	fillInOccluded<<<gridSize, blockSize>>>(dLDisp, dRDisp, dLPlanes, dRPlanes, dLCost, dRCost, dLOccludeMask, dROccludeMask, cols, rows, winRadius, (float)Dmax, 1,
//								lR_to, lG_to, lB_to, lGray_to, lGrad_to, rR_to, rG_to, rB_to, rGray_to, rGrad_to);
/*
	leftRightCheck<<<gridSize, blockSize>>>(dRDisp, dLDisp, dLOccludeMask, dROccludeMask, cols, rows, (float)Dmin, (float)Dmax, 1);


	fillInOccluded<<<gridSize, blockSize>>>(dLDisp, dRDisp, dLPlanes, dRPlanes, dLCost, dRCost, dLOccludeMask, 
								dROccludeMask, cols, rows, winRadius, (float)Dmax, 1,
								lR_to, lG_to, lB_to, lGray_to, lGrad_to, rR_to, rG_to, rB_to, rGray_to, rGrad_to);
*/
    //   	weightedMedianFilter<<<gridSize, blockSize>>>(dLDisp, dLOccludeMask, dRDisp, dROccludeMask, cols, rows, (float)Dmax, false, lR_to, lG_to, lB_to, lGray_to, rR_to, rG_to, rB_to, rGray_to);

	cudaDeviceSynchronize();

	std::cout<<"Post Process:"<<GetTimer()<<std::endl;	
#endif


        // copy disparity map from global memory on device to host
        cudaMemcpy(cvRightDisp_f.ptr<float>(0), dRDisp, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(cvLeftDisp_f.ptr<float>(0), dLDisp, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);

	leftDisp = cvLeftDisp_f.clone();
	rightDisp = cvRightDisp_f.clone();


	if(showLeftDisp)
	{
		cv::Mat tmpDisp, tmpDisp1;
		cvRightDisp_f.convertTo(tmpDisp1, CV_8U, scale);
		cv::imshow("Right Disp", tmpDisp1);
		cvLeftDisp_f.convertTo(tmpDisp, CV_8U, scale);
		cv::imshow("Left Disp", tmpDisp);
		cv::waitKey(0);
	}

#if USE_PCL

	float *normal_xy = new float[2*imgSize];

	cudaMemcpy(normal_xy, dRPlanes, imgSize*2*sizeof(float), cudaMemcpyDeviceToHost);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer ("3D Viewer"));

	struct callback_args cb_args;
	cb_args.cloud = cloud;
	cb_args.normals = normals;
        cb_args.viewerPtr = viewer;
	viewer->registerPointPickingCallback(pp_callback, (void*)&cb_args);

	float scale_along_z =10.0f;
	float nx, ny, nz, norm;

	for(int y=0; y<rows; y++)
	{
		for(int x=0; x<cols; x++)
		{
			int idx = y*cols+x;
			pcl::PointXYZRGB p;
			p.x = x; p.y = y;
			p.z = (Dmax - cvRightDisp_f.ptr<float>(0)[idx])*scale_along_z;
			//p.z = cvRightDisp_f.ptr<float>(0)[idx]*scale_along_z;
			cv::Vec3b bgr = rightImg.at<cv::Vec3b>(y, x);
			p.b = bgr.val[0];
			p.g = bgr.val[1];
			p.r = bgr.val[2];
			cloud->push_back(p);
			pcl::Normal n;
			n.normal_x = normal_xy[2*idx];
			n.normal_y = normal_xy[2*idx+1];
			n.normal_z = -sqrtf(1.0f-n.normal_x*n.normal_x-n.normal_y*n.normal_y)/scale_along_z;
				
			norm = sqrtf(n.normal_x*n.normal_x+n.normal_y*n.normal_y+n.normal_z*n.normal_z);
			n.normal_x /= norm;
			n.normal_y /= norm;
			n.normal_z /= norm;

			normals->push_back(n);
		}
	}


/*	viewer->addPointCloud(cloud,"cloud", 0);
	viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals, 10, 10.0f, "normals");
	viewer->spin();
	viewer->removeAllPointClouds(0);
	viewer->removeAllShapes(0);

	cloud->points.clear();
	normals->points.clear();
*/
	cudaMemcpy(normal_xy, dLPlanes, imgSize*2*sizeof(float), cudaMemcpyDeviceToHost);


	for(int y=0; y<rows; y++)
	{
		for(int x=0; x<cols; x++)
		{
			int idx = y*cols+x;
			pcl::PointXYZRGB p;
			p.x = x-1.1f*cols; 
			p.y = y;
			p.z = (Dmax - cvLeftDisp_f.ptr<float>(0)[idx])*scale_along_z;
			//p.z = cvLeftDisp_f.ptr<float>(0)[idx]*scale_along_z;
			cv::Vec3b bgr = leftImg.at<cv::Vec3b>(y, x);
			p.b = bgr.val[0];
			p.g = bgr.val[1];
			p.r = bgr.val[2];
			cloud->push_back(p);
			pcl::Normal n;
			n.normal_x = normal_xy[2*idx];
			n.normal_y = normal_xy[2*idx+1];
			n.normal_z = -sqrtf(1.0f-n.normal_x*n.normal_x-n.normal_y*n.normal_y)/scale_along_z;

			norm = sqrtf(n.normal_x*n.normal_x+n.normal_y*n.normal_y+n.normal_z*n.normal_z);
			n.normal_x /= norm;
			n.normal_y /= norm;
			n.normal_z /= norm;
			normals->push_back(n);
		}
	}

	viewer->addPointCloud(cloud,"cloud", 0);
	viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals, 5, 10.0f, "normals");
	viewer->spin();

	delete[] normal_xy;
	
#endif	



        // Free device memory
        cudaFree(dRDisp);
        cudaFree(dRPlanes);
        cudaFree(dLDisp);
        cudaFree(dLPlanes);
        cudaFree(states);
        cudaFree(dRGrad);
        cudaFree(dLGrad);
	cudaFree(dLCost);
	cudaFree(dRCost);
	cudaFree(lR_ca);
	cudaFree(lG_ca);
	cudaFree(lB_ca);
	cudaFree(lGray_ca);
	cudaFree(lGrad_ca);
	cudaFree(rR_ca);
	cudaFree(rG_ca);
	cudaFree(rB_ca);
	cudaFree(rGray_ca);
	cudaFree(rGrad_ca);
	cudaFree(dLOccludeMask);
	cudaFree(dROccludeMask);

        curandDestroyGenerator(gen);
	cudaDestroyTextureObject(lR_to);
	cudaDestroyTextureObject(lG_to);
	cudaDestroyTextureObject(lB_to);
	cudaDestroyTextureObject(lGray_to);
	cudaDestroyTextureObject(lGrad_to);
	cudaDestroyTextureObject(rR_to);
	cudaDestroyTextureObject(rG_to);
	cudaDestroyTextureObject(rB_to);
	cudaDestroyTextureObject(rGray_to);
	cudaDestroyTextureObject(rGrad_to);

        cudaDeviceReset();

	// save disparity image
	 
//	savePNG(lImgPtr_8u, "l.png", cols, rows);

//	savePNG(rImgPtr_8u, "r.png", cols, rows);
	

}

__device__ float deviceSmoothStep(float a, float b, float x)
{
    float t = fminf((x - a)/(b - a), 1.0f);
    t = fmaxf(t, 0.0f);
    return t*t*(3.0f - (2.0f*t));
}

// evaluate window-based disimilarity unary cost
__device__ float evaluateCost_huber(cudaTextureObject_t lR_to, cudaTextureObject_t lG_to, cudaTextureObject_t lB_to,
				cudaTextureObject_t lGradX_to, cudaTextureObject_t lGradY_to, 
				cudaTextureObject_t lGradXY_to, cudaTextureObject_t lGradYX_to, 
				cudaTextureObject_t rR_to, cudaTextureObject_t rG_to, cudaTextureObject_t rB_to, 
				cudaTextureObject_t rGradX_to, cudaTextureObject_t rGradY_to, 
				cudaTextureObject_t rGradXY_to, cudaTextureObject_t rGradYX_to,
				cudaTextureObject_t lGray_to, cudaTextureObject_t rGray_to,
				float* d_left_cost_vol, float* d_right_cost_vol,
				const float &u, const float &v, const int &x, const int &y, const float &disp, const int &cols, const int &rows, 
				const float &min_disp, const float &max_disp, const int &winRadius, /*const*/ float &nx, /*const*/ float &ny, /*const*/ float &nz, const int &base) // base 0 left, 1 right
{
        float cost = 0.0f;
        float weight, af, bf, cf;
	float weight_c_pmsh = 255.0f/5.0f;

        const float xf = (float)x;
        const float yf = (float)y;
	const float du = 1.0f/(float)cols;
	const float dv = 1.0f/(float)rows;
		
	//const float alpha_c = 0.05f;
	//const float alpha_g = 1.0f - alpha_c;
	//const float gammaMin = 5.0f;
	//const float gammaMax = 28.0f;
	//const float gammaRadius = 39.0f;
	//const float color_truncation = 0.04f;
	//const float gradient_truncation = 0.01f;
	//const float bad_cost = alpha_c*color_truncation+alpha_g*gradient_truncation;
	const int census_r = 2; 
	float bc, mc, new_disp;	

	//mc-cnn cost volume
	float* base_cost_vol;
	const int img_size = cols*rows;

//	nx = 0.f; ny = 0.f; nz = 1.f;	//uncomment to force frontal parallel window

        // af = -nx/nz, bf = -ny/nz, cf = (nx*x+ny*y+nz*disp)/nz
        af = nx/nz*(-1.0f);
        bf = ny/nz*(-1.0f);
        cf = (nx*xf + ny*yf + nz*disp)/nz;

//        if( isinf(af)!=0 || isinf(bf)!=0 || isinf(cf)!=0) return bad_cost;
        if( isinf(af)!=0 || isinf(bf)!=0 || isinf(cf)!=0) return 1.0f;

        float tmp_disp, r, g, b, color_L1, sign, wn, hn, match_center_disp;
        float weight_sum = 0.0f;
	int h, w, y_h, x_w;

	cudaTextureObject_t* baseR;
	cudaTextureObject_t* baseG;
	cudaTextureObject_t* baseB;
	cudaTextureObject_t* matchR;
	cudaTextureObject_t* matchG;
	cudaTextureObject_t* matchB;
	cudaTextureObject_t* baseGradX;
	cudaTextureObject_t* baseGradY;
	cudaTextureObject_t* baseGradXY;
	cudaTextureObject_t* baseGradYX;
	cudaTextureObject_t* matchGradX;
	cudaTextureObject_t* matchGradY;
	cudaTextureObject_t* matchGradXY;
	cudaTextureObject_t* matchGradYX;
	cudaTextureObject_t* baseGray;
	cudaTextureObject_t* matchGray;

	if(base == 0) // left base
	{
		sign = -1.0f;
		baseR = &lR_to; baseG = &lG_to;	baseB = &lB_to;
		matchR = &rR_to; matchG = &rG_to; matchB = &rB_to;
		baseGradX = &lGradX_to;	baseGradY = &lGradY_to;
		baseGradXY = &lGradXY_to; baseGradYX = &lGradYX_to;
		matchGradX = &rGradX_to; matchGradY = &rGradY_to;
		matchGradXY = &rGradXY_to; matchGradYX = &rGradYX_to;
		baseGray = &lGray_to; matchGray = &rGray_to;
		base_cost_vol = d_left_cost_vol;
	}
	else	// right base
	{
		sign = 1.0f;
		baseR = &rR_to; baseG = &rG_to; baseB = &rB_to;
		matchR = &lR_to; matchG = &lG_to; matchB = &lB_to;
		baseGradX = &rGradX_to;	baseGradY = &rGradY_to;
		baseGradXY = &rGradXY_to; baseGradYX = &rGradYX_to;
		matchGradX = &lGradX_to; matchGradY = &lGradY_to;
		matchGradXY = &lGradXY_to; matchGradYX = &lGradYX_to;
		baseGray = &rGray_to; matchGray = &lGray_to;
		base_cost_vol = d_right_cost_vol;
	}

	//match_center_disp = (af*xf + bf*yf + cf);

	for(h=-winRadius; h<=winRadius; h+=5)
        {
                for(w=-winRadius; w<=winRadius; w+=5)
                {
                        tmp_disp = (af*(xf+(float)w) + bf*(yf+(float)h) + cf);

			y_h = y+h;
			x_w = x+w;

			if(  tmp_disp>=min_disp && tmp_disp<=max_disp 
				&& ((-(int)tmp_disp+x>=0 && base==0) || ((int)tmp_disp+x<cols && base==1))
				//   && match_center_disp>=min_disp && match_center_disp<=max_disp && match_center_disp+x>=0 && match_center_disp+x<cols
				&& (x_w>=0) && (y_h>=0) && (x_w<cols) && (y_h<rows)
			)
			{
		                wn = (float)w*du;
		                hn = (float)h*dv;

		                r = fabsf(tex2D<float>(*baseR, u, v)-tex2D<float>(*baseR, u+wn, v+hn));
		                g = fabsf(tex2D<float>(*baseG, u, v)-tex2D<float>(*baseG, u+wn, v+hn));
		                b = fabsf(tex2D<float>(*baseB, u, v)-tex2D<float>(*baseB, u+wn, v+hn));

				//weight_c_pmsh = gammaMin+gammaRadius*deviceSmoothStep(0.0f, gammaMax, sqrtf((float)(w*w+h*h)));

		                weight = expf(-(r+b+g)*weight_c_pmsh);

		                weight_sum += weight;

				color_L1 = 0.0f; 

				// scale integer disp to texture scale
		                //tmp_disp = tmp_disp*du*sign;
				//match_center_disp = sign*match_center_disp*du;

				// census
#if 0
				tmp_disp = tmp_disp*du*sign;
				for(int hc=-census_r; hc<=census_r; hc++)
				{
					for(int wc=-census_r; wc<=census_r; wc++)
					{
						new_disp = (af*(xf+(float)w + (float)wc) + bf*(yf+(float)h+(float)hc) + cf)*sign*du;
						
						bc = (tex2D<float>(*baseGray, u+wn, v+hn)-tex2D<float>(*baseGray, u+wn+(float)wc*du, v+hn+(float)hc*dv));
						mc = (tex2D<float>(*matchGray, u+wn+tmp_disp, v+hn)-tex2D<float>(*matchGray, u+wn+new_disp+(float)wc*du, v+hn+(float)hc*dv));
						color_L1 += bc*mc < 0.0f ? 0.04f : 0.0f;


						/*bc = (tex2D<float>(*baseR, u+wn, v+hn)-tex2D<float>(*baseR, u+wn+(float)wc*du, v+hn+(float)hc*dv));
						mc = (tex2D<float>(*matchR, u+wn+tmp_disp, v+hn)-tex2D<float>(*matchR, u+wn+new_disp+(float)wc*du, v+hn+(float)hc*dv));
						color_L1 += bc*mc < 0.0f ? 0.01388888888f : 0.0f;

						bc = (tex2D<float>(*baseG, u+wn, v+hn)-tex2D<float>(*baseG, u+wn+(float)wc*du, v+hn+(float)hc*dv));
						mc = (tex2D<float>(*matchG, u+wn+tmp_disp, v+hn)-tex2D<float>(*matchG, u+wn+new_disp+(float)wc*du, v+hn+(float)hc*dv));
						color_L1 += bc*mc < 0.0f ? 0.01388888888f : 0.0f;

						bc = (tex2D<float>(*baseB, u+wn, v+hn)-tex2D<float>(*baseB, u+wn+(float)wc*du, v+hn+(float)hc*dv));
						mc = (tex2D<float>(*matchB, u+wn+tmp_disp, v+hn)-tex2D<float>(*matchB, u+wn+new_disp+(float)wc*du, v+hn+(float)hc*dv));
						color_L1 += bc*mc < 0.0f ? 0.01388888888f : 0.0f;*/
					}
				}
#endif


#if 1				// mc-cnn window-based
				float disp_ceil = ceilf(tmp_disp);
				float weight_floor = disp_ceil - tmp_disp;
				if(disp_ceil == max_disp) disp_ceil -= 1.0f;				
				int disp_int = (int)disp_ceil;


				if(disp_int == 0)
					color_L1 = min( base_cost_vol[img_size*disp_int+y_h*cols+x_w], 0.5f);
				else
					color_L1 = min( weight_floor*base_cost_vol[img_size*(disp_int-1)+y_h*cols+x_w] + (1.0f-weight_floor)*base_cost_vol[img_size*disp_int+y_h*cols+x_w], 0.5f);
#endif

#if 0	//pixe-wise wrong
				float disp_o = roundf(tmp_disp);
				if(disp_o <= 0) disp_o = 0.0f;
				else if(disp_o > max_disp) disp_o = max_disp-1.0f;
				else disp_o = disp_o-1.0f;
				//mc-cnn cost volume disparity range [1, max_disp]
				
				//color_L1 = min( (base_cost_vol[img_size*((int)disp_o)+y*cols+x] + 1.0f)*0.5f, 0.5f); //fast mc-cnn
				//color_L1 = min(base_cost_vol[img_size*((int)disp_o)+y*cols+x], 0.5f);  //slow mc-cnn 
				color_L1 = base_cost_vol[img_size*((int)disp_o)+y*cols+x];

				if(isnan(color_L1)) color_L1 = 1.0f;
#endif


//color_L1 = (tex2D<float>(*baseGray, u, v)-tex2D<float>(*baseGray, u+wn, v+hn))*(tex2D<float>(*matchGray, u + match_center_disp, v)-tex2D<float>(*matchGray, u+wn+tmp_disp, v+hn))< 0.0f ? 0.01f : 0.0f;

/*	color_L1 += (tex2D<float>(*baseR, u, v)-tex2D<float>(*baseR, u+wn, v+hn))*(tex2D<float>(*matchR, u + match_center_disp, v)-tex2D<float>(*matchR, u+wn+tmp_disp, v+hn))< 0.0f ? 0.01f : 0.0f;
	color_L1 += (tex2D<float>(*baseG, u, v)-tex2D<float>(*baseG, u+wn, v+hn))*(tex2D<float>(*matchG, u + match_center_disp, v)-tex2D<float>(*matchG, u+wn+tmp_disp, v+hn))< 0.0f ? 0.01f : 0.0f;
	color_L1 += (tex2D<float>(*baseB, u, v)-tex2D<float>(*baseB, u+wn, v+hn))*(tex2D<float>(*matchB, u + match_center_disp, v)-tex2D<float>(*matchB, u+wn+tmp_disp, v+hn))< 0.0f ? 0.01f : 0.0f;
				color_L1 *= 0.333333333333333f;
*/
		                cost += weight*(color_L1*1.f
		              /*                  + 0.0f*min( (fabsf( tex2D<float>(*baseGradX, u + wn, v + hn) - tex2D<float>(*matchGradX, u + tmp_disp + wn, v + hn))
							      +fabsf( tex2D<float>(*baseGradY, u + wn, v + hn) - tex2D<float>(*matchGradY, u + tmp_disp + wn, v + hn))	
							      +fabsf( tex2D<float>(*baseGradXY, u + wn, v + hn) - tex2D<float>(*matchGradXY, u + tmp_disp + wn, v + hn))	
							      +fabsf( tex2D<float>(*baseGradYX, u + wn, v + hn) - tex2D<float>(*matchGradYX, u + tmp_disp + wn, v + hn)))*0.25f, gradient_truncation)
				*/		);	



/*		                r = fabsf( tex2D<float>(*baseR, u + wn, v + hn) - tex2D<float>(*matchR, u + tmp_disp + wn, v + hn));
		                g = fabsf( tex2D<float>(*baseG, u + wn, v + hn) - tex2D<float>(*matchG, u + tmp_disp + wn, v + hn));
		                b = fabsf( tex2D<float>(*baseB, u + wn, v + hn) - tex2D<float>(*matchB, u + tmp_disp + wn, v + hn));

		                color_L1 = (r+g+b)*0.33333333333f;

				
		                cost += weight * (alpha_c*min(color_L1, color_truncation)
		                                + alpha_g*min( (fabsf( tex2D<float>(*baseGradX, u + wn, v + hn) - tex2D<float>(*matchGradX, u + tmp_disp + wn, v + hn))
							      +fabsf( tex2D<float>(*baseGradY, u + wn, v + hn) - tex2D<float>(*matchGradY, u + tmp_disp + wn, v + hn))	
							      +fabsf( tex2D<float>(*baseGradXY, u + wn, v + hn) - tex2D<float>(*matchGradXY, u + tmp_disp + wn, v + hn))	
							      +fabsf( tex2D<float>(*baseGradYX, u + wn, v + hn) - tex2D<float>(*matchGradYX, u + tmp_disp + wn, v + hn)))*0.25f, gradient_truncation));
*/
			}
			else
			{
				/*wn = (float)w*du;
		                hn = (float)h*dv;

		                r = fabsf(tex2D<float>(*baseR, u, v)-tex2D<float>(*baseR, u+wn, v+hn));
		                g = fabsf(tex2D<float>(*baseG, u, v)-tex2D<float>(*baseG, u+wn, v+hn));
		                b = fabsf(tex2D<float>(*baseB, u, v)-tex2D<float>(*baseB, u+wn, v+hn));

				//weight_c_pmsh = gammaMin+gammaRadius*deviceSmoothStep(0.0f, gammaMax, sqrtf((float)(w*w+h*h)));

		                weight = expf(-(r+b+g)*weight_c_pmsh);

		                weight_sum += weight;*/
				//cost += bad_cost;
				//cost += 0.01f;	//AL_TGV
				cost += 1.0f;//*weight;
				weight_sum += 0.7f;
			}
                }
        }
	
	return cost/weight_sum;
}

__global__ void stereoMatching_huber( float* d_left_cost_vol, float* d_right_cost_vol, float* dRDispV, float* dLDispV, float* dRPlanesV, float* dLPlanesV,
					float* dRDisp, float* dRPlanes, float* dLDisp, float* dLPlanes,
		                        float* dLCost, float* dRCost, int cols, int rows, int winRadius,
		                        curandState* states, float maxDisp, 
					cudaTextureObject_t lR_to, cudaTextureObject_t lG_to, cudaTextureObject_t lB_to,
					cudaTextureObject_t lGray_to, cudaTextureObject_t lGradX_to, cudaTextureObject_t lGradY_to, 
					cudaTextureObject_t lGradXY_to, cudaTextureObject_t lGradYX_to, 
					cudaTextureObject_t rR_to, cudaTextureObject_t rG_to, cudaTextureObject_t rB_to, 
					cudaTextureObject_t rGray_to, cudaTextureObject_t rGradX_to, cudaTextureObject_t rGradY_to, 
					cudaTextureObject_t rGradXY_to, cudaTextureObject_t rGradYX_to, 
					float theta_sigma_d, float theta_sigma_n, int iteration, bool VIEW_PROPAGATION = true, bool PLANE_REFINE = true)
{
        const int x = blockIdx.x*blockDim.x + threadIdx.x;
        const int y = blockIdx.y*blockDim.y + threadIdx.y;

        // does not need to process borders
        if(x>=cols || y>=rows) return;

        const float u = ((float)x+0.5f)/(float)(cols);
        const float v = ((float)y+0.5f)/(float)(rows);

        const int idx = y*cols + x;

        // evaluate disparity of current pixel (based on right)
        float min_cost, cost, nx, ny, nz, norm, tmp_disp, delta_disp, cur_disp, s;
        int tmp_idx, i, j, new_x;
	const float lambda = 50.0f;
	bool even_iteration = iteration%2==0 ? true : false;

	if(even_iteration) goto RIGHT;

  	//------------------------------------------------------------ base left 0
LEFT:
        min_cost = 1e10f;

        // spatial  propagation
        for(i=-1; i<=1; i++)
        {
                for(j=-1; j<=1; j++)
                {
                        if(x+j<0 || x+j>=cols || y+i<0 || y+i>=rows) continue;

                        tmp_idx = idx + i*cols + j;

                        tmp_disp = dLDisp[tmp_idx]*maxDisp;
		
			nx = dLPlanes[tmp_idx*2];
			ny = dLPlanes[tmp_idx*2+1];				
			nz = sqrtf(1.0f-nx*nx-ny*ny);

                        cost =  lambda*evaluateCost_huber(lR_to, lG_to, lB_to, lGradX_to, lGradY_to, lGradXY_to, lGradYX_to,
							 rR_to, rG_to, rB_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to, lGray_to, rGray_to, d_left_cost_vol, d_right_cost_vol,
								u, v, x, y, tmp_disp, cols, rows, 0.0f, maxDisp, winRadius,
                                        			nx, ny, nz, 0);

			cost += 0.5f*( theta_sigma_d*powf(tmp_disp/maxDisp-dLDispV[idx], 2.0f) 
					+ theta_sigma_n*(powf(nx-dLPlanesV[idx*2], 2.0f)
					               + powf(ny-dLPlanesV[idx*2+1], 2.0f)) );

                        if(cost < min_cost)
                        {
                                min_cost = cost;
				dLCost[idx] = min_cost;
				dLDisp[idx] = tmp_disp/maxDisp;
				dLPlanes[idx*2] = nx;
				dLPlanes[idx*2 + 1] = ny;
                        }
                }
        }


        // view propagation
        if(VIEW_PROPAGATION)
        {
                new_x = x - (int)lroundf(dLDisp[idx]);

                // check if in range
                if(new_x>=0 && new_x<cols)
                {
                        tmp_idx = idx + new_x - x;
                        tmp_disp = dRDisp[tmp_idx]*maxDisp;
		
			nx = dRPlanes[tmp_idx*2];
			ny = dRPlanes[tmp_idx*2+1];
			nz = sqrtf(1.0f-nx*nx-ny*ny);

                        cost = lambda*evaluateCost_huber(lR_to, lG_to, lB_to, lGradX_to, lGradY_to, lGradXY_to, lGradYX_to,
							 rR_to, rG_to, rB_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to, lGray_to, rGray_to, d_left_cost_vol, d_right_cost_vol,
							u, v, x, y, tmp_disp, cols, rows, 0.0f, maxDisp, winRadius,
                                        		nx, ny, nz, 0);

			cost += 0.5f*( theta_sigma_d*powf(tmp_disp/maxDisp-dLDispV[idx], 2.0f) 
					+ theta_sigma_n*(powf(nx-dLPlanesV[idx*2], 2.0f)
						       + powf(ny-dLPlanesV[idx*2+1], 2.0f)) );
											
                        if(cost < min_cost)
                        {
                                min_cost = cost;
                                dLCost[idx] = min_cost;
                                dLDisp[idx] = tmp_disp/maxDisp;
                                dLPlanes[2*idx] = nx;
                                dLPlanes[2*idx+1] = ny;
                        }
                }
        }


        // left plane refinement
        // exponentially reduce disparity search range
        if(PLANE_REFINE)
        {
                s = 1.0f;

                for(delta_disp=maxDisp*0.5f; delta_disp>=0.1f; delta_disp *= 0.5f)
                {
                        cur_disp = dLDisp[idx]*maxDisp;

                        cur_disp += (curand_uniform(&states[idx])*2.0f-1.0f)*delta_disp;

                        if(cur_disp<0.0f || cur_disp>maxDisp)
                        {
                                s *= 0.5f;
                                continue;
                        }

			nx = dLPlanes[idx*2];
			ny = dLPlanes[idx*2+1];
			nz = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + sqrtf(1.0f-nx*nx-ny*ny);				
                        nx = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + nx;
                        ny = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + ny;


                        //normalize
                        norm = sqrtf(nx*nx+ny*ny+nz*nz);

			nx /= norm;
			ny /= norm;
			nz /= norm;
			nz = fabs(nz);

			if( isinf(nx)!=0 || isinf(ny)!=0 || isinf(nz)!=0 )
			{
				s *= 0.5f;
				continue;
			}

			
			cost = lambda*evaluateCost_huber(lR_to, lG_to, lB_to, lGradX_to, lGradY_to, lGradXY_to, lGradYX_to,
							 rR_to, rG_to, rB_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to, lGray_to, rGray_to, d_left_cost_vol, d_right_cost_vol,
							 u, v, x, y, cur_disp, cols, rows, 0.0f, maxDisp, winRadius, nx, ny, nz, 0);

			cost += 0.5f*( theta_sigma_d*powf(cur_disp/maxDisp-dLDispV[idx],2.0f) 
				    + theta_sigma_n*(powf(nx-dLPlanesV[idx*2], 2.0f)
						   + powf(ny-dLPlanesV[idx*2+1], 2.0f)) );


			if(cost < min_cost)
			{
				min_cost = cost;
				dLCost[idx] = min_cost;
				dLDisp[idx] = cur_disp/maxDisp;
				dLPlanes[idx*2] = nx;
				dLPlanes[idx*2 + 1] = ny;
			}

			s *= 0.5f;
                }
        }

        if(even_iteration) return;


	//--------------------------------------------  base right 1
RIGHT:
        min_cost = 1e10f;
        // spatial  propagation
        for(i=-1; i<=1; i++)
        {
                for(j=-1; j<=1; j++)
                {
                        if(x+j<0 || x+j>=cols || y+i<0 || y+i>=rows) continue;

                        tmp_idx = idx + i*cols + j;

                        tmp_disp = dRDisp[tmp_idx]*maxDisp;
							
			nx = dRPlanes[tmp_idx*2];
			ny = dRPlanes[tmp_idx*2+1];
			nz = sqrtf(1.0f-nx*nx-ny*ny);

                        cost =  lambda*evaluateCost_huber(lR_to, lG_to, lB_to, lGradX_to, lGradY_to, lGradXY_to, lGradYX_to, 
							rR_to, rG_to, rB_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to, lGray_to, rGray_to, d_left_cost_vol, d_right_cost_vol,
							u, v, x, y, tmp_disp, cols, rows, 0.0f, maxDisp, winRadius,
							nx, ny, nz, 1);
												
			cost += 0.5f*( theta_sigma_d*powf(tmp_disp/maxDisp-dRDispV[idx], 2.0f) 
				     + theta_sigma_n*( powf(nx-dRPlanesV[idx*2], 2.0f)
				                     + powf(ny-dRPlanesV[idx*2+1], 2.0f) ) );

                        // base 0 left, 1 right
                        if(cost < min_cost)
                        {
                                min_cost = cost;
				dRCost[idx] = min_cost;
				dRDisp[idx] = tmp_disp/maxDisp;
				dRPlanes[idx*2] = nx;
				dRPlanes[idx*2 + 1] = ny;
                        }
                }
        }

        // view propagation
        if(VIEW_PROPAGATION)
        {
                new_x = (int)lroundf(dRDisp[idx]) + x;

                // check if in range
                if(new_x>=0 && new_x<cols)
                {
                        tmp_idx = idx + new_x - x;
                        tmp_disp = dLDisp[tmp_idx]*maxDisp;
							
			nx = dLPlanes[tmp_idx*2];
			ny = dLPlanes[tmp_idx*2+1];
			nz = sqrtf(1.0f-nx*nx-ny*ny);

                        cost = lambda*evaluateCost_huber(lR_to, lG_to, lB_to, lGradX_to, lGradY_to, lGradXY_to, lGradYX_to,
							 rR_to, rG_to, rB_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to, lGray_to, rGray_to, d_left_cost_vol, d_right_cost_vol,
							u, v, x, y, tmp_disp, cols, rows, 0.0f, maxDisp, winRadius,
                                        		nx, ny, nz, 1);
							
			cost += 0.5*( theta_sigma_d*powf(tmp_disp/maxDisp-dRDispV[idx],2.0f) 
				    + theta_sigma_n*(powf(nx-dRPlanesV[idx*2], 2.0f)
					           + powf(ny-dRPlanesV[idx*2+1], 2.0f)) );

                        if(cost < min_cost)
                        {
                                min_cost = cost;
                                dRCost[idx] = min_cost;
                                dRDisp[idx] = tmp_disp/maxDisp;
                                dRPlanes[2*idx] = nx;
                                dRPlanes[2*idx+1] = ny;
                        }
                }
        }


        // right plane refinement
        if(PLANE_REFINE)
        {
                s = 1.0f;

                for(delta_disp=maxDisp*0.5f; delta_disp>=0.1f; delta_disp *= 0.5f)
                {
                        cur_disp = dRDisp[idx]*maxDisp;

                        cur_disp += (curand_uniform(&states[idx])*2.0f-1.0f)*delta_disp;

                        if(cur_disp<0.0f || cur_disp>(float)maxDisp)
                        {
                                s *= 0.5f;
                                continue;
                        }

			nx = dRPlanes[idx*2];
			ny = dRPlanes[idx*2+1];
			nz = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + sqrtf(1.0f-nx*nx-ny*ny);
                        nx = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + nx;
                        ny = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + ny;
			
                        
			//normalize
                        norm = sqrtf(nx*nx+ny*ny+nz*nz);

			nx /= norm;
			ny /= norm;
			nz /= norm;
			nz = fabs(nz);

			if(isinf(nx)!=0 || isinf(ny)!=0 || isinf(nz)!=0)
			{
				s *= 0.5f;
				continue;
			}

			cost = lambda*evaluateCost_huber(lR_to, lG_to, lB_to, lGradX_to, lGradY_to, lGradXY_to, lGradYX_to,
							 rR_to, rG_to, rB_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to, lGray_to, rGray_to, d_left_cost_vol, d_right_cost_vol,
							u, v, x, y, cur_disp, cols, rows, 0.0f, maxDisp, winRadius, nx, ny, nz, 1);

			cost += 0.5f*( theta_sigma_d*powf(cur_disp/maxDisp-dRDispV[idx], 2.0f) 
						+ theta_sigma_n*(powf(nx-dRPlanesV[idx*2], 2.0f)
						               + powf(ny-dRPlanesV[idx*2+1], 2.0f)) );

			if(cost < min_cost)
			{
				min_cost = cost;
				dRCost[idx] = min_cost;
				dRDisp[idx] = cur_disp/maxDisp;
				dRPlanes[idx*2] = nx;
				dRPlanes[idx*2 + 1] = ny;
			}
			
			s *= 0.5f;
                }
        }

	if(even_iteration) goto LEFT;
}

// DispV = u; Disp = a
__global__ void stereoMatching_AL_TGV(  float* d_left_ambiguity, float* d_right_ambiguity,
					float* d_left_cost_vol, float* d_right_cost_vol, float* dRDispV, float* dLDispV, float* dRPlanesV, float* dLPlanesV,
					float* dRDisp, float* dRPlanes, float* dLDisp, float* dLPlanes,
		                        float* dLCost, float* dRCost, float* d_right_L, float* d_left_L, const float theta, const float lambda_d,
					const int cols, const int rows, int winRadius,
		                        curandState* states, float maxDisp, 
					cudaTextureObject_t lR_to, cudaTextureObject_t lG_to, cudaTextureObject_t lB_to,
					cudaTextureObject_t lGray_to, cudaTextureObject_t lGradX_to, cudaTextureObject_t lGradY_to, 
					cudaTextureObject_t lGradXY_to, cudaTextureObject_t lGradYX_to, 
					cudaTextureObject_t rR_to, cudaTextureObject_t rG_to, cudaTextureObject_t rB_to, 
					cudaTextureObject_t rGray_to, cudaTextureObject_t rGradX_to, cudaTextureObject_t rGradY_to, 
					cudaTextureObject_t rGradXY_to, cudaTextureObject_t rGradYX_to, 
					float theta_sigma_d, float theta_sigma_n, int iteration, bool VIEW_PROPAGATION = false, bool PLANE_REFINE = true)
{
        const int x = blockIdx.x*blockDim.x + threadIdx.x;
        const int y = blockIdx.y*blockDim.y + threadIdx.y;

        // does not need to process borders
        if(x>=cols || y>=rows) return;

        const float u = ((float)x+0.5f)/(float)(cols);
        const float v = ((float)y+0.5f)/(float)(rows);

        const int idx = y*cols + x;

        // evaluate disparity of current pixel (based on right)
        float min_cost, cost, nx, ny, nz, norm, tmp_disp, delta_disp, cur_disp, s, u_a;
        int tmp_idx, i, j, new_x;
	bool even_iteration = iteration%2==0 ? true : false;

	const float alpha_c = 0.05f;
	const float alpha_g = 1.0f - alpha_c;
	const float color_truncation = 0.04f;
	const float gradient_truncation = 0.01f;
	const float max_cost = 1.0f;
	//const float max_cost = alpha_c*color_truncation+alpha_g*gradient_truncation;
	const float left_ambiguity = d_left_ambiguity[idx] == 0.0f ? 0.5f : 1.0f;
	const float right_ambiguity = d_right_ambiguity[idx] == 0.0f ? 0.5f : 1.0f;

	if(even_iteration) goto RIGHT;

  	//------------------------------------------------------------ base left 0
LEFT:
        min_cost = 1e10f;

        // spatial  propagation
        for(i=-1; i<=1; i++)
        {
                for(j=-1; j<=1; j++)
                {
                        if(x+j<0 || x+j>=cols || y+i<0 || y+i>=rows) continue;

                        tmp_idx = idx + i*cols + j;

                        tmp_disp = dLDisp[tmp_idx]*maxDisp;
		
			nx = dLPlanes[tmp_idx*2];
			ny = dLPlanes[tmp_idx*2+1];				
			nz = sqrtf(1.0f-nx*nx-ny*ny);

                        cost =  left_ambiguity*lambda_d*evaluateCost_huber(lR_to, lG_to, lB_to, lGradX_to, lGradY_to, lGradXY_to, lGradYX_to,
							 rR_to, rG_to, rB_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to, lGray_to, rGray_to, d_left_cost_vol, d_right_cost_vol,
								u, v, x, y, tmp_disp, cols, rows, 0.0f, maxDisp, winRadius,
                                        			nx, ny, nz, 0);

			cost /= max_cost;
			u_a = dLDispV[idx]-tmp_disp/maxDisp;
			cost += d_left_L[idx]*u_a + u_a*u_a/(2.0f*theta); 

                        if(cost < min_cost)
                        {
                                min_cost = cost;
				dLCost[idx] = min_cost;
				dLDisp[idx] = tmp_disp/maxDisp;
				dLPlanes[idx*2] = nx;
				dLPlanes[idx*2 + 1] = ny;
                        }
                }
        }


        // view propagation
        if(VIEW_PROPAGATION)
        {
                new_x = x - (int)lroundf(dLDisp[idx]);

                // check if in range
                if(new_x>=0 && new_x<cols)
                {
                        tmp_idx = idx + new_x - x;
                        tmp_disp = dRDisp[tmp_idx]*maxDisp;
		
			nx = dRPlanes[tmp_idx*2];
			ny = dRPlanes[tmp_idx*2+1];
			nz = sqrtf(1.0f-nx*nx-ny*ny);

                        cost = left_ambiguity*lambda_d*evaluateCost_huber(lR_to, lG_to, lB_to, lGradX_to, lGradY_to, lGradXY_to, lGradYX_to,
							 rR_to, rG_to, rB_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to, lGray_to, rGray_to, d_left_cost_vol, d_right_cost_vol,
							u, v, x, y, tmp_disp, cols, rows, 0.0f, maxDisp, winRadius,
                                        		nx, ny, nz, 0);

			cost /= max_cost;
			u_a = dLDispV[idx]-tmp_disp/maxDisp;
			cost += d_left_L[idx]*u_a + u_a*u_a/(2.0f*theta); 
											
                        if(cost < min_cost)
                        {
                                min_cost = cost;
                                dLCost[idx] = min_cost;
                                dLDisp[idx] = tmp_disp/maxDisp;
                                dLPlanes[2*idx] = nx;
                                dLPlanes[2*idx+1] = ny;
                        }
                }
        }


        // left plane refinement
        // exponentially reduce disparity search range
        if(PLANE_REFINE)
        {
                s = 1.0f;

                for(delta_disp=maxDisp*0.5f; delta_disp>=0.1f; delta_disp *= 0.5f)
                {
                        cur_disp = dLDisp[idx]*maxDisp;

                        cur_disp += (curand_uniform(&states[idx])*2.0f-1.0f)*delta_disp;

                        if(cur_disp<0.0f || cur_disp>maxDisp)
                        {
                                s *= 0.5f;
                                continue;
                        }

			nx = dLPlanes[idx*2];
			ny = dLPlanes[idx*2+1];
			nz = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + sqrtf(1.0f-nx*nx-ny*ny);				
                        nx = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + nx;
                        ny = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + ny;


                        //normalize
                        norm = sqrtf(nx*nx+ny*ny+nz*nz);

			nx /= norm;
			ny /= norm;
			nz /= norm;
			nz = fabs(nz);

			if( isinf(nx)!=0 || isinf(ny)!=0 || isinf(nz)!=0 )
			{
				s *= 0.5f;
				continue;
			}

			
			cost = left_ambiguity*lambda_d*evaluateCost_huber(lR_to, lG_to, lB_to, lGradX_to, lGradY_to, lGradXY_to, lGradYX_to,
							 rR_to, rG_to, rB_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to, lGray_to, rGray_to, d_left_cost_vol, d_right_cost_vol,
							 u, v, x, y, cur_disp, cols, rows, 0.0f, maxDisp, winRadius, nx, ny, nz, 0);

			cost /= max_cost;
			u_a = dLDispV[idx]-cur_disp/maxDisp;
			cost += d_left_L[idx]*u_a + u_a*u_a/(2.0f*theta); 


			if(cost < min_cost)
			{
				min_cost = cost;
				dLCost[idx] = min_cost;
				dLDisp[idx] = cur_disp/maxDisp;
				dLPlanes[idx*2] = nx;
				dLPlanes[idx*2 + 1] = ny;
			}

			s *= 0.5f;
                }
        }

        if(even_iteration) return;


	//--------------------------------------------  base right 1
RIGHT:
        min_cost = 1e10f;
        // spatial  propagation
        for(i=-1; i<=1; i++)
        {
                for(j=-1; j<=1; j++)
                {
                        if(x+j<0 || x+j>=cols || y+i<0 || y+i>=rows) continue;

                        tmp_idx = idx + i*cols + j;

                        tmp_disp = dRDisp[tmp_idx]*maxDisp;
							
			nx = dRPlanes[tmp_idx*2];
			ny = dRPlanes[tmp_idx*2+1];
			nz = sqrtf(1.0f-nx*nx-ny*ny);

                        cost =  right_ambiguity*lambda_d*evaluateCost_huber(lR_to, lG_to, lB_to, lGradX_to, lGradY_to, lGradXY_to, lGradYX_to, 
							rR_to, rG_to, rB_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to, lGray_to, rGray_to, d_left_cost_vol, d_right_cost_vol,
							u, v, x, y, tmp_disp, cols, rows, 0.0f, maxDisp, winRadius,
							nx, ny, nz, 1);
												
			cost /= max_cost;
			u_a = dRDispV[idx]-tmp_disp/maxDisp;
			cost += d_right_L[idx]*u_a + u_a*u_a/(2.0f*theta); 

                        // base 0 left, 1 right
                        if(cost < min_cost)
                        {
                                min_cost = cost;
				dRCost[idx] = min_cost;
				dRDisp[idx] = tmp_disp/maxDisp;
				dRPlanes[idx*2] = nx;
				dRPlanes[idx*2 + 1] = ny;
                        }
                }
        }

        // view propagation
        if(VIEW_PROPAGATION)
        {
                new_x = (int)lroundf(dRDisp[idx]) + x;

                // check if in range
                if(new_x>=0 && new_x<cols)
                {
                        tmp_idx = idx + new_x - x;
                        tmp_disp = dLDisp[tmp_idx]*maxDisp;
							
			nx = dLPlanes[tmp_idx*2];
			ny = dLPlanes[tmp_idx*2+1];
			nz = sqrtf(1.0f-nx*nx-ny*ny);

                        cost = right_ambiguity*lambda_d*evaluateCost_huber(lR_to, lG_to, lB_to, lGradX_to, lGradY_to, lGradXY_to, lGradYX_to,
							 rR_to, rG_to, rB_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to, lGray_to, rGray_to, d_left_cost_vol, d_right_cost_vol,
							u, v, x, y, tmp_disp, cols, rows, 0.0f, maxDisp, winRadius,
                                        		nx, ny, nz, 1);
							
			cost /= max_cost;
			u_a = dRDispV[idx]-tmp_disp/maxDisp;
			cost += d_right_L[idx]*u_a + u_a*u_a/(2.0f*theta); 

                        if(cost < min_cost)
                        {
                                min_cost = cost;
                                dRCost[idx] = min_cost;
                                dRDisp[idx] = tmp_disp/maxDisp;
                                dRPlanes[2*idx] = nx;
                                dRPlanes[2*idx+1] = ny;
                        }
                }
        }


        // right plane refinement
        if(PLANE_REFINE)
        {
                s = 1.0f;

                for(delta_disp=maxDisp*0.5f; delta_disp>=0.1f; delta_disp *= 0.5f)
                {
                        cur_disp = dRDisp[idx]*maxDisp;

                        cur_disp += (curand_uniform(&states[idx])*2.0f-1.0f)*delta_disp;

                        if(cur_disp<0.0f || cur_disp>(float)maxDisp)
                        {
                                s *= 0.5f;
                                continue;
                        }

			nx = dRPlanes[idx*2];
			ny = dRPlanes[idx*2+1];
			nz = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + sqrtf(1.0f-nx*nx-ny*ny);
                        nx = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + nx;
                        ny = (curand_uniform(&states[idx])*2.0f - 1.0f)*s + ny;
			
                        
			//normalize
                        norm = sqrtf(nx*nx+ny*ny+nz*nz);

			nx /= norm;
			ny /= norm;
			nz /= norm;
			nz = fabs(nz);

			if(isinf(nx)!=0 || isinf(ny)!=0 || isinf(nz)!=0)
			{
				s *= 0.5f;
				continue;
			}

			cost = right_ambiguity*lambda_d*evaluateCost_huber(lR_to, lG_to, lB_to, lGradX_to, lGradY_to, lGradXY_to, lGradYX_to,
							 rR_to, rG_to, rB_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to, lGray_to, rGray_to, d_left_cost_vol, d_right_cost_vol,
							u, v, x, y, cur_disp, cols, rows, 0.0f, maxDisp, winRadius, nx, ny, nz, 1);

			cost /= max_cost;
			u_a = dRDispV[idx]-cur_disp/maxDisp;
			cost += d_right_L[idx]*u_a + u_a*u_a/(2.0f*theta); 

			if(cost < min_cost)
			{
				min_cost = cost;
				dRCost[idx] = min_cost;
				dRDisp[idx] = cur_disp/maxDisp;
				dRPlanes[idx*2] = nx;
				dRPlanes[idx*2 + 1] = ny;
			}
			
			s *= 0.5f;
                }
        }

	if(even_iteration) goto LEFT;
}


__global__ void UpdateDualVariablesKernel(int cols, int rows, float* dDispV, float* dPlanesV, float* dWeight, float* dDispPd,
					  float* dPlanesPn, float theta_sigma_d, float theta_sigma_n)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
        const int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x >= cols || y>= rows) return;
		
	const int idx = y*cols+x;

	// forward difference derivative
	float dDisp_dx = x==cols-1 ? 0.0f : dDispV[idx+1] - dDispV[idx];
	float dDips_dy = y==rows-1 ? 0.0f : dDispV[idx+cols] - dDispV[idx];

	float nx = dPlanesV[2*idx];
	float ny = dPlanesV[2*idx+1];

	float dnx_dx = x==cols-1 ? 0.0f : dPlanesV[2*(idx+1)] - nx;
	float dny_dx = x==cols-1 ? 0.0f : dPlanesV[2*(idx+1)+1] - ny; 

	float dnx_dy = y==rows-1 ? 0.0f : dPlanesV[2*(idx+cols)] - nx;
	float dny_dy = y==rows-1 ? 0.0f : dPlanesV[2*(idx+cols)+1] - ny;

	// weight
	const float gp = dWeight[idx];
	const float gp_inv = 1.0f/gp;
	
	// from Pock's paper ALG3 Huber-ROF model
	// L = sqrt(8)
	// gamma = lambda, delta = alpha huber param, mu = 2*sqrt(gamma*delta)/L   
	// tau = mu/(2*gamma) primal, sigma = mu/(2*delta) dual
	const float eps = 0.001f;	//huber param		
	//const float beta_d = 2.0f*sqrtf(theta_sigma_d*eps)/sqrtf(8.0f)/(2.0f*eps);	// dual sigma	
	//const float beta_n = 2.0f*sqrtf(theta_sigma_n*eps)/sqrtf(8.0f)/(2.0f*eps);

	const float beta_d = 1.0f/sqrtf(8.0f);	// dual sigma	
	const float beta_n = 1.0f/sqrtf(8.0f);

	// update dual disparity x direction
	float tmp[2];
	tmp[0] = (dDispPd[2*idx]+beta_d*gp*dDisp_dx)/(1.0f+beta_d*eps*gp_inv);

	// dual disparity y
	tmp[1] = (dDispPd[2*idx+1]+beta_d*gp*dDips_dy)/(1.0f+beta_d*eps*gp_inv);

	float norm = sqrtf(tmp[0]*tmp[0]+tmp[1]*tmp[1]);
		
	// project back to unit ball x
	dDispPd[2*idx] = tmp[0]/fmaxf(1.0f, norm);
	
	// project back y
	dDispPd[2*idx+1] = tmp[1]/fmaxf(1.0f, norm);

	// update dual unit normal
	// gradient of normal
	// x direction of normal x element
	tmp[0]	= (dPlanesPn[4*idx]+beta_n*gp*dnx_dx)/(1.0f+beta_n*eps*gp_inv);
	// y direction of normal x element
	tmp[1] = (dPlanesPn[4*idx+1]+beta_n*gp*dnx_dy)/(1.0f+beta_n*eps*gp_inv);
	
	// norm
	norm = sqrtf(tmp[0]*tmp[0]+tmp[1]*tmp[1]);
	
	// 0, 1 index: x dir of normal x element, y dir of normal x element
	dPlanesPn[4*idx] = tmp[0]/fmaxf(1.0f, norm);
	dPlanesPn[4*idx+1] = tmp[1]/fmaxf(1.0f, norm);
	
	// x direction of normal y element
	tmp[0] = (dPlanesPn[4*idx+2]+beta_n*gp*dny_dx)/(1.0f+beta_n*eps*gp_inv);
	// y direction of normal y element
	tmp[1] = (dPlanesPn[4*idx+3]+beta_n*gp*dny_dy)/(1.0f+beta_n*eps*gp_inv);
	
	// norm
	norm = sqrtf(tmp[0]*tmp[0]+tmp[1]*tmp[1]);

	// project back
	// 2, 3 index: x dir of normal y element, y dir of normal y element
	dPlanesPn[4*idx+2] = tmp[0]/fmaxf(1.0f, norm);
	dPlanesPn[4*idx+3] = tmp[1]/fmaxf(1.0f, norm);
}


__global__ void UpdatePrimalVariablesKernel(int cols, int rows, float* dDispPd, float* dPlanesPn, float* dWeight, float* dDispV, float* dDisp,
					    float* dPlanesV, float* dPlanes, float theta_sigma_d, float theta_sigma_n)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
        const int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x >= cols || y>= rows) return;
		
	const int idx = y*cols+x;

	float divPd, divPnx, divPny;

	if(x==0)
	{
		divPd = dDispPd[2*idx];
		divPnx = dPlanesPn[4*idx];
		divPny = dPlanesPn[4*idx+2];
	}
	else if(x==cols-1)
	{
		divPd = -dDispPd[2*(idx-1)];
		divPnx = -dPlanesPn[4*(idx-1)];
		divPny = -dPlanesPn[4*(idx-1)+2];
	}
	else
	{
		divPd = dDispPd[2*idx] - dDispPd[2*(idx-1)];
		divPnx = dPlanesPn[4*idx] - dPlanesPn[4*(idx-1)];
		divPny = dPlanesPn[4*idx+2] - dPlanesPn[4*(idx-1)+2];
	}

	if(y==0)
	{
		divPd += dDispPd[2*idx+1];
		divPnx += dPlanesPn[4*idx+1];
		divPny += dPlanesPn[4*idx+3];
	}
	else if(y==rows-1)
	{
		divPd += -dDispPd[2*(idx-cols)+1];
		divPnx += -dPlanesPn[4*(idx-cols)+1];
		divPny += -dPlanesPn[4*(idx-cols)+3];
	}
	else
	{
		divPd += dDispPd[2*idx+1] - dDispPd[2*(idx-cols)+1];
		divPnx += dPlanesPn[4*idx+1] - dPlanesPn[4*(idx-cols)+1];
		divPny += dPlanesPn[4*idx+3] - dPlanesPn[4*(idx-cols)+3];
	}

	// weight
	const float gp = dWeight[idx];

	// from Pock's paper ALG3 Huber-ROF model
	// L = sqrt(8)
	// gamma = lambda, delta = alpha huber param, mu = 2*sqrt(gamma*delta)/L   
	// tau = mu/(2*gamma) primal, sigma = mu/(2*delta) dual
//	const float eps = 0.001f;	//huber param		
//	const float nu_d = 2.0f*sqrtf(theta_sigma_d*eps)/sqrtf(8.0f)/(2.0f*theta_sigma_d);	// primal tau
//	const float nu_n = 2.0f*sqrtf(theta_sigma_n*eps)/sqrtf(8.0f)/(2.0f*theta_sigma_n);

	const float nu_d = 1.0f/sqrtf(8.0f);	// primal tau
	const float nu_n = 1.0f/sqrtf(8.0f);

/*	dDispV[idx] = (dDispV[idx]+nu_d*(theta_sigma_d*dDisp[idx]+gp*divPd))/(1.0f+nu_d*theta_sigma_d);
	dPlanesV[2*idx] = (dPlanesV[2*idx]+nu_n*(theta_sigma_n*dPlanes[2*idx]+gp*divPnx))/(1.0f+nu_n*theta_sigma_n);
	dPlanesV[2*idx+1] = (dPlanesV[2*idx+1]+nu_n*(theta_sigma_n*dPlanes[2*idx+1]+gp*divPny))/(1.0f+nu_n*theta_sigma_n);
*/
	// extrapolation
	dDispV[idx] = 2.0f*(dDispV[idx]+nu_d*(theta_sigma_d*dDisp[idx]+gp*divPd))/(1.0f+nu_d*theta_sigma_d) - dDispV[idx];
	dPlanesV[2*idx] = 2.0f*(dPlanesV[2*idx]+nu_n*(theta_sigma_n*dPlanes[2*idx]+gp*divPnx))/(1.0f+nu_n*theta_sigma_n) - dPlanesV[2*idx];
	dPlanesV[2*idx+1] = 2.0f*(dPlanesV[2*idx+1]+nu_n*(theta_sigma_n*dPlanes[2*idx+1]+gp*divPny))/(1.0f+nu_n*theta_sigma_n) - dPlanesV[2*idx+1];		
}

void huberROFSmooth(float* dDispV, float* dPlanesV, float* dDisp, float* dPlanes,
		    float* dDispPd, float* dPlanesPn, float* dWeight, int cols, int rows, float theta_sigma_d, float theta_sigma_n)
{
	// kernels size
	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1)/blockSize.x, (rows + blockSize.x - 1)/blockSize.x); 

	// update dual variables
	UpdateDualVariablesKernel<<<gridSize, blockSize>>>(cols, rows, dDispV, dPlanesV, dWeight, dDispPd,
					 		   dPlanesPn, theta_sigma_d, theta_sigma_n);

	// update primal variables
	UpdatePrimalVariablesKernel<<<gridSize, blockSize>>>(cols, rows, dDispPd, dPlanesPn, dWeight, dDispV, dDisp,
					    		     dPlanesV, dPlanes, theta_sigma_d, theta_sigma_n);
}



// initialize random uniformally distributed plane normals
__global__ void init_plane_normals_weights_huber(float* dPlanes, float* d_ls_mask, float* dWeight, cudaTextureObject_t GradX_to, cudaTextureObject_t GradY_to, 
					cudaTextureObject_t GradXY_to, cudaTextureObject_t GradYX_to, curandState_t* states, const int cols, const int rows)
{
        const int x = blockIdx.x*blockDim.x + threadIdx.x;
        const int y = blockIdx.y*blockDim.y + threadIdx.y;

        if(x >= cols || y>= rows) return;

        int idx = y*cols+x;
	const float u = ((float)x+0.5f)/(float)(cols);
        const float v = ((float)y+0.5f)/(float)(rows);

        float x1, x2, nx, ny;

	const float ksi = 3.0f;
	const float eta = 0.8f;

	// per-pixel weight
	dWeight[idx] = (1.0f - d_ls_mask[idx])*expf( -ksi*powf( tex2D<float>(GradX_to, u, v)*tex2D<float>(GradX_to, u, v)
							 	+tex2D<float>(GradY_to, u, v)*tex2D<float>(GradY_to, u, v)				
					      		 	+tex2D<float>(GradXY_to, u, v)*tex2D<float>(GradXY_to, u, v)
					      		 	+tex2D<float>(GradYX_to, u, v)*tex2D<float>(GradYX_to, u, v), eta*0.5f) );

	while(true)
	{
		while(true)
		{
		        x1 = curand_uniform(&states[idx])*2.0f - 1.0f;
		        x2 = curand_uniform(&states[idx])*2.0f - 1.0f;

		        if( x1*x1 + x2*x2 < 1.0f ) break;
		}

		nx = 2.0f*x1*sqrtf(1.0f - x1*x1 - x2*x2);
		ny = 2.0f*x2*sqrtf(1.0f - x1*x1 - x2*x2);

		//draw normal samples more restrictively
		if(nx*nx+ny*ny<=0.25f) break;
	}

        idx = idx << 1;
        dPlanes[idx] = nx;
        dPlanes[idx+1] = ny;
}

// initialize disp, plane, dual variable ...
__global__ void init_variables(float* dLDisp, float* dRDisp, float* dLPlanes, float* dRPlanes,
			       float* dLDispV, float* dRDispV, float* dLPlanesV, float* dRPlanesV,
			       float* dLDispPd, float* dRDispPd, float* dLPlanesPn, float* dRPlanesPn,
			       float* dLWeight, float* dRWeight,
			       cudaTextureObject_t lGradX_to, cudaTextureObject_t rGradX_to, 
			       cudaTextureObject_t lGradY_to, cudaTextureObject_t rGradY_to, 
     			       cudaTextureObject_t lGradXY_to, cudaTextureObject_t rGradXY_to, 
			       cudaTextureObject_t lGradYX_to, cudaTextureObject_t rGradYX_to, 
			       int cols, int rows)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x >= cols || y>= rows) return;	
		
	const int idx = y*cols+x;
	const float u = ((float)x+0.5f)/(float)(cols);
        const float v = ((float)y+0.5f)/(float)(rows);
	const float ksi = 3.0f;
	const float eta = 0.8f;
	
	dLDispV[idx] = dLDisp[idx];
	dRDispV[idx] = dRDisp[idx];
	dLPlanesV[2*idx] = dLPlanes[2*idx];
	dLPlanesV[2*idx+1] = dLPlanes[2*idx+1];
	dRPlanesV[2*idx] = dRPlanes[2*idx];
	dRPlanesV[2*idx+1] = dRPlanes[2*idx+1];
	
	// per-pixel weight
/*	dLWeight[idx] = expf( -ksi* powf( tex2D<float>(lGradX_to, u, v)*tex2D<float>(lGradX_to, u, v)
			                  +tex2D<float>(lGradY_to, u, v)*tex2D<float>(lGradY_to, u, v)				
			      		  +tex2D<float>(lGradXY_to, u, v)*tex2D<float>(lGradXY_to, u, v)
			      		  +tex2D<float>(lGradYX_to, u, v)*tex2D<float>(lGradYX_to, u, v), eta*0.5f) );


	dRWeight[idx] = expf( -ksi* powf( tex2D<float>(rGradX_to, u, v)*tex2D<float>(rGradX_to, u, v)
			      		  +tex2D<float>(rGradY_to, u, v)*tex2D<float>(rGradY_to, u, v) 
			      		  +tex2D<float>(rGradXY_to, u, v)*tex2D<float>(rGradXY_to, u, v)
			      		  +tex2D<float>(rGradYX_to, u, v)*tex2D<float>(rGradYX_to, u, v), eta*0.5f) );
	*/
	dLDispPd[2*idx] = 0.0f;
	dLDispPd[2*idx+1] = 0.0f;
	dRDispPd[2*idx] = 0.0f;
	dRDispPd[2*idx+1] = 0.0f;
	dLPlanesPn[4*idx] = 0.0f;
	dLPlanesPn[4*idx+1] = 0.0f;
	dLPlanesPn[4*idx+2] = 0.0f;
	dLPlanesPn[4*idx+3] = 0.0f;
	dRPlanesPn[4*idx] = 0.0f;
	dRPlanesPn[4*idx+1] = 0.0f;
	dRPlanesPn[4*idx+2] = 0.0f;
	dRPlanesPn[4*idx+3] = 0.0f;
}

float smoothstep(float a, float b, float x)
{
    float t = min((x - a)/(b - a), 1.);
	t = max(t, 0.);
    return t*t*(3.0 - (2.0*t));
}



__global__ void leftRightCheckHuber(float* dRDispPtr, float* dLDispPtr, float* dRPlanes, float* dLPlanes, int* dROMask, int* dLOMask,				    
				    int cols, int rows, float minDisp, float maxDisp,  
				    const float d_thresh = 0.5f, const float angle_thresh = 5.0f, bool set_zero=false)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x >= cols || y>= rows) return;	

	const int idx = y*cols+x;

	float tmp_disp = dLDispPtr[idx]*(maxDisp-1.0f);

	float nx0 = dLPlanes[2*idx];
	float ny0 = dLPlanes[2*idx+1];
	float nz0 = sqrtf(1.0f-nx0*nx0-ny0*ny0);

	int tmp_idx = x - (int)lroundf(tmp_disp);

	float nx1 = dRPlanes[2*(idx+tmp_idx-x)];
	float ny1 = dRPlanes[2*(idx+tmp_idx-x)+1];
	float nz1 = sqrtf(1.0f-nx1*nx1-ny1*ny1);
	
	// unit normal, dot = cos(angle)
	float dot = nx0*nx1+ny0*ny1+nz0*nz1;

	const float theta_thresh = cospif(angle_thresh/180.0f);

	if( (tmp_disp<=0.f) || (tmp_disp>maxDisp-1.0f) ||(tmp_idx < 0) || tmp_idx>=cols || 
	    fabsf(tmp_disp - dRDispPtr[idx + tmp_idx - x]*(maxDisp-1.0f)) > d_thresh 
	 //   || dot < theta_thresh	// angle
	   )
	{	
		dLOMask[idx] = 1;
		if(set_zero) dLDispPtr[idx] = 0.0f;
	}
	else	dLOMask[idx] = 0;


	tmp_disp = dRDispPtr[idx]*(maxDisp-1.0f);
	nx0 = dRPlanes[2*idx];
	ny0 = dRPlanes[2*idx+1];
	nz0 = sqrtf(1.0f-nx0*nx0-ny0*ny0);

	tmp_idx = x + (int)lroundf(tmp_disp);

	nx1 = dLPlanes[2*(idx+tmp_idx-x)];
	ny1 = dLPlanes[2*(idx+tmp_idx-x)+1];
	nz1 = sqrtf(1.0f-nx1*nx1-ny1*ny1);

	dot = nx0*nx1+ny0*ny1+nz0*nz1;

	if( (tmp_disp<=0.f) || (tmp_disp>maxDisp-1.0f) ||(tmp_idx < 0) || tmp_idx>=cols || 
	    fabsf(tmp_disp - dLDispPtr[idx + tmp_idx - x]*(maxDisp-1.0f)) > d_thresh 
	  //  || dot < theta_thresh
	  )
	{
		dROMask[idx] = 1;
		if(set_zero) dRDispPtr[idx] = 0.0f;
	}
	else dROMask[idx] = 0;
}

__global__ void setVZero(float* dV, int* dOMask, int cols, int rows)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x>=cols || y>=rows ) return;	

	const int idx = y*cols+x;

	if( dOMask[idx] == 0 ) return;

	dV[2*idx] = dV[2*idx+1] = 0.0f;
}

__global__ void fillInOccludedHuber(float* dDisp, float* dPlanes, int* dOMask, int cols, int rows, float maxDisp)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x>=cols || y>=rows ) return;	

	const int idx = y*cols+x;

	if( dOMask[idx] == 0 ) return;

	const float xf = (float)x;
	const float yf = (float)y;
	
	float nx, ny, nz, af, bf, cf, tmp_disp;

	// memory of min disp range (0, maxDisp)
	float final_disp = 1e10f;
	float final_nx, final_ny;

	// search right
	int i = 1;
	while( x+i<cols )
	{
		if( dOMask[idx+i] == 0)	// not occlusion
		{
			nx = dPlanes[(idx+i)*2];
			ny = dPlanes[(idx+i)*2+1];
			nz = sqrtf(1.0f - nx*nx - ny*ny);

			nx = 0.0f; ny = 0.0f; nz = 1.0f;	// no extrapolation

			// disp in memory ranges 0 to 1, scale back 
			float dispOrigionalScale = dDisp[idx+i]*(maxDisp-1.0f);

			// af = -nx/nz, bf = -ny/nz, cf = (nx*x+ny*y+nz*disp)/nz			
	/*		af = nx/nz*(-1.0f);
			bf = ny/nz*(-1.0f);
			cf = (nx*(xf+(float)i) + ny*yf + nz*dispOrigionalScale)/nz;

			if( isinf(af)!=0 || isinf(bf)!=0 || isinf(cf)!=0 )
			{
				i++;
				continue;
			}

			// extrapolate
			tmp_disp = af*(xf-(float)i) + bf*yf + cf;
*/
			tmp_disp = dispOrigionalScale;
			
			if(tmp_disp>0.0f && tmp_disp <= maxDisp-1.0f)
			{
				final_disp = tmp_disp;
				final_nx = nx;
				final_ny = ny;
				dOMask[idx] = 0;
				break;		
			}
		}

		i++;
	}

	//search left for the nearest valid(none zero) disparity
	i = -1;
	while( x+i>=0 )
	{
		// valid disparity
		if( dOMask[idx+i] == 0)
		{
			nx = dPlanes[(idx+i)*2];
			ny = dPlanes[(idx+i)*2+1];
			nz = sqrtf(1.0f - nx*nx - ny*ny);

			nx = 0.0f; ny = 0.0f; nz = 1.0f;	// no extrapolation

			// disp in memory ranges 0 to 1, scale back 
			float dispOrigionalScale = dDisp[idx+i]*(maxDisp-1.0f);
	
	/*		af = nx/nz*(-1.0f);
			bf = ny/nz*(-1.0f);
			cf = (nx*(xf+(float)i) + ny*yf + nz*dispOrigionalScale)/nz;
	
			if( isinf(af)!=0 || isinf(bf)!=0 || isinf(cf)!=0 )
			{
				i--;
				continue;
			}
	
			// extrapolate 
			tmp_disp = af*(xf+(float)i) + bf*yf + cf;
			*/

			tmp_disp = dispOrigionalScale;

			if( (tmp_disp>0.0f && tmp_disp<=maxDisp-1.0f && tmp_disp < final_disp) || dOMask[idx] == 1)
			{
				final_disp = tmp_disp;
				final_nx = nx;
				final_ny = ny;	
				break;		
			}
		}
		
		i--;
	}

	if(final_disp != 1e10f)
	{
		dDisp[idx] = final_disp/(maxDisp-1.0f);
		dPlanes[idx*2] = final_nx;
		dPlanes[idx*2+1] = final_ny;

		// auxiliary 
	/*	dDispV[idx] = final_disp/maxDisp;
		dPlanesV[idx*2] = final_nx;
		dPlanesV[idx*2+1] = final_ny;

		// primal
		dDispPd[2*idx] = dDispPd[2*(idx+i)];
		dDispPd[2*idx+1] = dDispPd[2*(idx+i)+1];
		dPlanesPn[4*idx] = dPlanesPn[4*(idx+i)];
		dPlanesPn[4*idx+1] = dPlanesPn[4*(idx+i)+1];
		dPlanesPn[4*idx+2] = dPlanesPn[4*(idx+i)+2];
		dPlanesPn[4*idx+3] = dPlanesPn[4*(idx+i)+3];*/
	}
	else	
	{
	//	dDisp[idx] = 0.0f;
	//	dPlanes[idx*2] = 0.0f;
	//	dPlanes[idx*2+1] = 0.0f;

/*		// auxiliary 
		dDispV[idx] = 0.0f;
		dPlanesV[idx*2] = 0.0f;
		dPlanesV[idx*2+1] = 0.0f;

		// primal
		dDispPd[2*idx] = 0.0f;
		dDispPd[2*idx+1] = 0.0f;
		dPlanesPn[4*idx] = 0.0f;
		dPlanesPn[4*idx+1] = 0.0f;
		dPlanesPn[4*idx+2] = 0.0f;
		dPlanesPn[4*idx+3] = 0.0f;
*/	}
}

__global__ void anisotropicDiffusionTensorG(const int cols, const int rows, float* d_gray_gauss, 
						float* d_ls_mask, float* d_G, const float a, const float b)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x>=cols || y>=rows ) return;	

	const int idx = y*cols+x;

	//d_G[idx*4] = 1.0f; d_G[idx*4+1] = 0.0f; d_G[idx*4+2] = 0.0f; d_G[idx*4+3] = 1.0f; return;


	const float zero = 1e-4f;

	// central difference
	float nx, ny;
	if(x==0) nx = d_gray_gauss[idx+1] - d_gray_gauss[idx];
	else if(x==cols-1) nx = d_gray_gauss[idx] - d_gray_gauss[idx-1];
	else nx = (d_gray_gauss[idx+1] - d_gray_gauss[idx-1])*0.5f;

	if(y==0) ny = d_gray_gauss[idx+cols] - d_gray_gauss[idx];
	else if(y==rows-1) ny = d_gray_gauss[idx] - d_gray_gauss[idx-cols];
 	else ny = (d_gray_gauss[idx+cols] - d_gray_gauss[idx-cols])*0.5f;

	float s = expf(-a*powf(nx*nx+ny*ny, b*0.5f));

	// normalize
	float norm = sqrtf(nx*nx + ny*ny);

	if(norm > zero)	
	{
		nx /= norm; 
		ny /= norm;
		d_G[idx*4] = s*nx*nx + ny*ny;
		d_G[idx*4+1] = d_G[idx*4+2] = (s-1.0f)*nx*ny;
		d_G[idx*4+3] = s*ny*ny + nx*nx;
	}
	else 
	{
		d_G[idx*4] = 1.0f;
		d_G[idx*4+1] = 0.0f;
		d_G[idx*4+2] = 0.0f;
		d_G[idx*4+3] = 1.0f;
	}

	// update Diffusion tensor where the line segment is
	if(d_ls_mask[idx] < zero) return;

	if(x==0) nx = d_ls_mask[idx+1] - d_ls_mask[idx];
	else if(x==cols-1) nx = d_ls_mask[idx] - d_ls_mask[idx-1];
	else nx = (d_ls_mask[idx+1] - d_ls_mask[idx-1])*0.5f;

	if(y==0) ny = d_ls_mask[idx+cols] - d_ls_mask[idx];
	else if(y==rows-1) ny = d_ls_mask[idx] - d_ls_mask[idx-cols];
 	else ny = (d_ls_mask[idx+cols] - d_ls_mask[idx-cols])*0.5f;	
	
	s = expf(-a*powf(nx*nx+ny*ny, b*0.5f));

	norm = sqrtf(nx*nx + ny*ny);

	if(norm > zero)	
	{
		nx /= norm; 
		ny /= norm;
		d_G[idx*4] = s*nx*nx + ny*ny;
		d_G[idx*4+1] = d_G[idx*4+2] = (s-1.0f)*nx*ny;
		d_G[idx*4+3] = s*ny*ny + nx*nx;
	}
	else 
	{
		d_G[idx*4] = 1.0f;
		d_G[idx*4+1] = 0.0f;
		d_G[idx*4+2] = 0.0f;
		d_G[idx*4+3] = 1.0f;
	}
}


__global__ void AL_TGV_dualUpdate(const int cols, const int rows, float* d_G, float* d_p, float* d_u, float* d_v, 
					float* d_q, const float tau_p, const float tau_q,
					const float lambda_s, const float lambda_a, const bool TGV=true)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x>=cols || y>=rows ) return;	

	const int idx = y*cols+x;

	// forward differences with with Neumann boundary conditions
	float du_dx = x==cols-1 ? 0.0f : d_u[idx+1]-d_u[idx];
	float du_dy = y==rows-1 ? 0.0f : d_u[idx+cols]-d_u[idx];

	// use du_dx & du_dy as a temporary storage for (du-v)
	du_dx -= d_v[2*idx];
	du_dy -= d_v[2*idx+1];

	float px = d_p[idx*2] + tau_p*(d_G[4*idx]*du_dx + d_G[4*idx+1]*du_dy);
	float py = d_p[idx*2+1] + tau_p*(d_G[4*idx+2]*du_dx + d_G[4*idx+3]*du_dy);


	//huber
//	px /= 1.0f+tau_p*0.001f; py /= 1.0f+tau_p*0.001f;

//	float px = d_p[idx*2] + tau_p*du_dx;
//	float py = d_p[idx*2+1] + tau_p*du_dy;

	float norm = sqrtf(px*px+py*py);

	// project back 
	d_p[idx*2] = px/fmaxf(1.0f, norm/lambda_s);
	d_p[idx*2+1] = py/fmaxf(1.0f, norm/lambda_s);

	if(TGV)
	{
		float dvx_dx = x==cols-1 ? 0.0f : d_v[2*(idx+1)] - d_v[2*idx];
		float dvy_dx = x==cols-1 ? 0.0f : d_v[2*(idx+1)+1] - d_v[2*idx+1];

		float dvx_dy = y==rows-1 ? 0.0f : d_v[2*(idx+cols)] - d_v[2*idx];
		float dvy_dy = y==rows-1 ? 0.0f : d_v[2*(idx+cols)+1] - d_v[2*idx+1];


		float qxx = d_q[4*idx] + tau_q*dvx_dx; 
		float qxy = d_q[4*idx+1] + tau_q*dvx_dy;
		float qyx = d_q[4*idx+2] + tau_q*dvy_dx;
		float qyy = d_q[4*idx+3] + tau_q*dvy_dy;

		norm = sqrtf(qxx*qxx + qxy*qxy + qyx*qyx + qyy*qyy);

		d_q[4*idx]  = qxx/fmaxf(1.0f, norm/lambda_a);
		d_q[4*idx+1]  = qxy/fmaxf(1.0f, norm/lambda_a);
		d_q[4*idx+2]  = qyx/fmaxf(1.0f, norm/lambda_a);
		d_q[4*idx+3]  = qyy/fmaxf(1.0f, norm/lambda_a);
	}
}

__global__ void AL_TGV_computeGp(const int cols, const int rows, float* d_p, float* d_G, float* d_Gp)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x>=cols || y>=rows ) return;	

	const int idx = y*cols+x;
	d_Gp[2*idx] = d_G[4*idx]*d_p[2*idx] + d_G[4*idx+1]*d_p[2*idx+1];
	d_Gp[2*idx+1] = d_G[4*idx+2]*d_p[2*idx] + d_G[4*idx+3]*d_p[2*idx+1];
}

__global__ void AL_TGV_primalUpdate(const int cols, const int rows, float* d_u, float* d_G, float* d_p, 
					float* d_L, float* d_a, float* d_v, float* d_q, float* d_Gp,
					const float tau_u, const float tau_v, const float theta_inv, const bool TGV=true)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x>=cols || y>=rows ) return;	

	const int idx = y*cols+x;

	// divergence operator, backward difference with Dirichlet boundary conditions
	float div_x, div_y;	

	if(x==0) div_x = d_Gp[2*idx];
	else if(x==cols-1) div_x = -d_Gp[2*(idx-1)];
	else div_x = d_Gp[2*idx] - d_Gp[2*(idx-1)];

	if(y==0) div_y = d_Gp[2*idx+1];
	else if(y==rows-1) div_y = -d_Gp[2*(idx-cols)+1];
	else div_y = d_Gp[2*idx+1] - d_Gp[2*(idx-cols)+1];


/*	if(x==0) div_x = d_p[2*idx];
	else if(x==cols-1) div_x = -d_p[2*(idx-1)];
	else div_x = d_p[2*idx] - d_p[2*(idx-1)];

	if(y==0) div_y = d_p[2*idx+1];
	else if(y==rows-1) div_y = -d_p[2*(idx-cols)+1];
	else div_y = d_p[2*idx+1] - d_p[2*(idx-cols)+1];
*/
	// update u
	float tmp = (d_u[idx] + tau_u*(div_x+div_y) - tau_u*d_L[idx] + tau_u*theta_inv*d_a[idx])/(1.0f + tau_u*theta_inv);

	tmp = fminf(fmaxf(tmp, 0.0f), 1.0f);

	d_u[idx] = 2.0f*tmp - d_u[idx];

	
	if(TGV)
	{
		// qx divergence
		if(x==0) div_x = d_q[4*idx];
		else if(x==cols-1) div_x = -d_q[4*(idx-1)];
		else div_x = d_q[4*idx] - d_q[4*(idx-1)];

		if(y==0) div_y = d_q[4*idx+1];
		else if(y==rows-1) div_y = -d_q[4*(idx-cols)+1];
		else div_y = d_q[4*idx+1] - d_q[4*(idx-cols)+1];	

		tmp = d_v[2*idx] + tau_v*(d_p[2*idx] + div_x + div_y);

		d_v[2*idx] = 2.0f*tmp - d_v[2*idx];

		//qy divergence
		if(x==0) div_x = d_q[4*idx+2];
		else if(x==cols-1) div_x = -d_q[4*(idx-1)+2];
		else div_x = d_q[4*idx+2] - d_q[4*(idx-1)+2];

		if(y==0) div_y = d_q[4*idx+3];
		else if(y==rows-1) div_y = -d_q[4*(idx-cols)+3];
		else div_y = d_q[4*idx+3] - d_q[4*(idx-cols)+3];

		tmp = d_v[2*idx+1] + tau_v*(d_p[2*idx+1] + div_x + div_y);

		d_v[2*idx+1] = 2.0f*tmp - d_v[2*idx+1];
	}
}

__global__ void AL_TGV_augmentedLagranianUpdate(const int cols, const int rows, float* d_L, float* d_u, float* d_a, const float theta_inv)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x>=cols || y>=rows ) return;	

	const int idx = y*cols+x;

	d_L[idx] = d_L[idx] + (d_u[idx] - d_a[idx])*0.5f*theta_inv;
}

__global__ void Init2TGV(const int cols, const int rows, float* d_L, float* d_p, float* d_q, float* d_v)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x>=cols || y>=rows ) return;	

	int idx = y*cols+x;

	d_L[idx] = 0.0f;

	idx *= 2;
	d_p[idx] = 0.0f;
	d_p[idx+1] = 0.0f;
	d_v[idx] = 0.0f;
	d_v[idx+1] = 0.0f;

	idx *= 2;
	d_q[idx] = 0.0f;
	d_q[idx+1] = 0.0f;
	d_q[idx+2] = 0.0f;
	d_q[idx+3] = 0.0f;
}


struct MyData
{
	float* cost_vol;
	cv::Mat disp;
	int width;
	int height;
	int depth;
	std::vector<std::vector<int>> *mst_vertices_vec;
	std::vector<mst_graph_t> *mst_vec; 
	mst_graph_t* mst;
};

void ShowSlice(int event, int x, int y, int flags, void* userdata)
{

	if (event == cv::EVENT_LBUTTONDOWN)
	{
		const int width = ((MyData*)userdata)->width;
		const int depth = ((MyData*)userdata)->depth;
		const int height = ((MyData*)userdata)->height;
		const float* cost_vol = ((MyData*)userdata)->cost_vol;
		cv::Mat disp = ((MyData*)userdata)->disp;
		const int img_size = height*width;
		cv::Mat slice;
		const int n = 20;
		slice.create(depth+n, width, CV_32F);

		for(int i=0; i<n; i++)	std::memcpy(slice.ptr<float>(i), disp.ptr<float>(y), width*sizeof(float));

		for(int h=n; h<depth+n; h++)
		{
			for(int w=0; w<width; w++)
			{
				
				slice.at<float>(h, w) = cost_vol[(h-n)*img_size+y*width+w];// < 0.5f ? cost_vol[(h-n)*img_size+y*width+w] : 1.0f;
			}
		}

		cv::imshow("slice", slice);

		std::cout<<"x: "<<x <<" y: "<<y<<"\n";
	}
}



__global__ void RemoveNanFromCostVolume(float* d_mccnn_cv, float* d_ambiguity, const int disp_max, const int cols, const int rows, const bool fast_arch=false)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x>=cols || y>=rows ) return;	

	const int img_size = cols*rows;
	const int idx = y*cols+x;
	float cost;
	float cost_sum = 0.0f;

	for(int d=0; d<disp_max; d++)
	{
		cost = d_mccnn_cv[d*img_size+idx];
	
		if(!isfinite(cost)) 
		{
			cost = d_mccnn_cv[d*img_size+idx] = 1.0f;
			
		}
		else if(fast_arch)
		{
			cost = d_mccnn_cv[d*img_size+idx] = (cost+1.0f)*0.5f;
		}

		cost_sum += cost; 
	}

	cost_sum /= (float)disp_max;

	d_ambiguity[idx] = cost_sum < 0.8f ? 0.0f : 1.0f;
}


__global__ void MCCNN_ALTV_CostVolumeWTA(float* d_mccnn_cv, float* d_a, float* d_L, float* d_u, float theta,
					 const float lambda_d, const int disp_max, const int cols, const int rows)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x>=cols || y>=rows ) return;	

	float min_cost = 1e10f;
	const int img_size = cols*rows;
	const int idx = y*cols+x;
	float cur_cost;
	int best_disp = -1;
	float ua_diff;

	for(int d=0; d<disp_max; d++)
	{
		cur_cost = d_mccnn_cv[d*img_size+idx];

		ua_diff = d_u[idx]-(float)d/disp_max;

		cur_cost = cur_cost*lambda_d + d_L[idx]*ua_diff + ua_diff*ua_diff/(2.0f*theta);

		if(cur_cost<min_cost)
		{
			min_cost = cur_cost;
			best_disp = d;
		}
	}

//	if(best_disp != -1) d_a[idx] = (float)best_disp/disp_max;
//	else d_a[idx] = 0.0f;	return;

	if(best_disp == -1) return;

	const int idx_cv = best_disp*img_size+idx;

	ua_diff = d_u[idx]-fmaxf(0.0f, (float)(best_disp-1)/disp_max);

	float pre_cost = best_disp == 0 ? 0.0f : d_mccnn_cv[idx_cv-img_size]*lambda_d + d_L[idx]*ua_diff + ua_diff*ua_diff/(2.0f*theta);

	ua_diff = d_u[idx]-(float)best_disp/disp_max;

	cur_cost = d_mccnn_cv[idx_cv]*lambda_d + d_L[idx]*ua_diff + ua_diff*ua_diff/(2.0f*theta);

	ua_diff = d_u[idx]-fminf(1.0f, (float)(best_disp+1)/disp_max);

	float next_cost = best_disp == disp_max-1 ? 0.0f : d_mccnn_cv[idx_cv+img_size]*lambda_d + d_L[idx]*ua_diff + ua_diff*ua_diff/(2.0f*theta);

	float subpixel_update = (next_cost-pre_cost)*0.5f/(next_cost-2.0f*cur_cost+pre_cost);

	if(fabsf(subpixel_update) < 1.0f)
		d_a[idx] = ((float)best_disp - subpixel_update)/disp_max;	//minus if disparity, plus if depth
	else
		d_a[idx] = ((float)best_disp)/disp_max;	//1.0f mc_cnn cost volume disp from {1, 2, ..., dmax}
}

__global__ void InitNL2TGV( float* d_L, float* d_w/*local plane normal nx ny*/, float* d_p/*dual for u*/,
		            float* d_q/*dual for w*/, const int support_radius, const int cols, const int rows)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x>=cols || y>=rows ) return;	

	const int idx = y*cols+x;
	const int n_size = 2*support_radius*(support_radius+1);	//neighbor size
	const int idx_nl = idx*n_size;

	d_L[idx] = 0.0f;

	for(int i=0; i<n_size; i++) d_p[idx_nl+i] = 0.0f;

	for(int i=0; i<2*n_size; i++) d_q[2*idx_nl+i] = 0.0f;

	d_w[2*idx] = 0.0f;
	d_w[2*idx+1] = 0.0f; 
} 

//NL2TGV support weights j>i, j neighbor, i center
__global__ void InitAlpha1(float *d_Alpha1, cudaTextureObject_t R_to, cudaTextureObject_t G_to, cudaTextureObject_t B_to,
			   const int support_radius, const int cols, const int rows, const float wci/*color similarity*/, 
			   const float wpi/*proximity*/)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x>=cols || y>=rows ) return;	

	const int idx = y*cols+x;
	const int n_size = 2*support_radius*(support_radius+1);	//neighbor size

	const float du = 1.0f/(float)cols;
	const float dv = 1.0f/(float)rows;

        const float u = ((float)x+0.5f)*du;
        const float v = ((float)y+0.5f)*dv;

	float r, g, b, wn, hn;
	int count = -1;

	// simplified version without summing up weights
	for(int h=0; h<=support_radius; h++)
	{
		for(int w=-support_radius; w<=support_radius; w++)
		{
			count++;

			if(count <= support_radius) continue;
			
			if(x+w>=0 && x+w<cols && y+h<rows)
			{
				wn = (float)w*du;
				hn = (float)h*dv;
			 	r = tex2D<float>(R_to, u, v)-tex2D<float>(R_to, u+wn, v+hn);
				g = tex2D<float>(G_to, u, v)-tex2D<float>(G_to, u+wn, v+hn);
				b = tex2D<float>(B_to, u, v)-tex2D<float>(B_to, u+wn, v+hn);
				d_Alpha1[idx*n_size - support_radius + count - 1] = expf( - sqrtf(r*r+g*g+b*b)*wci - sqrtf((float)(h*h+w*w))*wpi );
			}
			else d_Alpha1[idx*n_size - support_radius + count - 1] = 0.0f;
		}
	}
}


__global__ void NL2TGV_primalUpdate(float* d_u/*disparity*/, float* d_w/*local plane normal nx ny*/, float* d_p/*dual for u*/,
				    float* d_q/*dual for w*/, float* d_a, float* d_L, float theta_inv, const int support_radius, 
				    const int cols, const int rows, const bool TGV=true)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x>=cols || y>=rows ) return;	

	const int idx = y*cols+x;
	const int n_size = 2*support_radius*(support_radius+1);	//neighbor size
	int count = -1;
	int j_nonlocal_idx, mirrored_j_idx;

	//Diagonal preconditioning for first order primal-dual algorithms
	const float tau_u = 1.0f/(2.0f*n_size);
	const float tau_w = 1.0f/(4.0f*n_size);


	//sum of nonlocal backward differences
	float sum_nl_bd_p = 0.0f;
	float sum_nl_bd_q[2] = {0.0f, 0.0f}; 

	float new_u;
	float new_w[2];

	for(int h=0; h<=support_radius; h++)
	{
		for(int w=-support_radius; w<=support_radius; w++)
		{
			count++;

			if(count <= support_radius) continue;

			j_nonlocal_idx = idx*n_size + count - support_radius - 1;
			mirrored_j_idx = j_nonlocal_idx + (-h*cols-w)*n_size;

			// backward differences
			if(x+w>=0 && x+w<cols && y+h<rows) sum_nl_bd_p += d_p[j_nonlocal_idx];	

			if(x-w>=0 && x-w<cols && y-h>=0) sum_nl_bd_p -= d_p[mirrored_j_idx];
		}
	}

	new_u = (d_u[idx] + tau_u*sum_nl_bd_p - tau_u*d_L[idx] + tau_u*theta_inv*d_a[idx])/(1.0f+tau_u*theta_inv);	//gradient descent
	new_u = fmaxf(0.0f, fminf(1.0f, new_u));	//clamp disparity to [0,1]
	d_u[idx] = 2.0f*new_u - d_u[idx];

	if(TGV)
	{
		count = -1;
		for(int h=0; h<=support_radius; h++)
		{
			for(int w=-support_radius; w<=support_radius; w++)
			{
				count++;

				if(count <= support_radius) continue;

				j_nonlocal_idx = idx*n_size + count - support_radius - 1;
				mirrored_j_idx = j_nonlocal_idx + (-h*cols-w)*n_size;

				// backward differences
				if(x+w>=0 && x+w<cols && y+h<rows)	//check nonlocal neighbor j inside image
				{
					sum_nl_bd_q[0] += d_q[j_nonlocal_idx*2] + w*d_p[j_nonlocal_idx];
					sum_nl_bd_q[1] += d_q[j_nonlocal_idx*2+1] + h*d_p[j_nonlocal_idx];
				}

				if(x-w>=0 && x-w<cols && y-h>=0)	//check the mirrored position of neighbor j around pixel i inside image
				{
					sum_nl_bd_q[0] -= d_q[mirrored_j_idx*2];
					sum_nl_bd_q[1] -= d_q[mirrored_j_idx*2+1];
				}
			}
		}

		new_w[0] = d_w[idx*2] + tau_w*sum_nl_bd_q[0];
		new_w[1] = d_w[idx*2+1] + tau_w*sum_nl_bd_q[1];

		d_w[2*idx] = 2.0f*new_w[0] - d_w[2*idx];
		d_w[2*idx+1] = 2.0f*new_w[1] - d_w[2*idx+1];
	}
}


__global__ void NL2TGV_dualUpdate(float* d_u/*disparity*/, float* d_w/*local plane normal nx ny*/, float* d_p/*dual for u*/,
				  float* d_q/*dual for w*/, float* d_alpha1, const int support_radius, const float lambda_s, 
				  const float lambda_a, const int cols, const int rows, const bool TGV=true)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x>=cols || y>=rows ) return;	

	int idx = y*cols+x;
	const int n_size = 2*support_radius*(support_radius+1);	//neighbor size
	int count = -1;
	int j_pixel_idx, j_nonlocal_idx, neibor_idx;

	//Diagonal preconditioning for first order primal-dual algorithms
	const float sigma_p = 1.0f/(2.0f);
	const float sigma_q = 1.0f/(2.0f);

//	float new_p;
//	float new_q[2];
	float alpha_ij_1, alpha_ij_0;
	float norm_p = 0.0f;
	float norm_q = 0.0f;

	//radius of 1
//	float alpha1s[4];
//	float new_ps[4];
//	float new_qs[8];

	//radius of 2
	float alpha1s[12];
	float new_ps[12];
	float new_qs[24];

	//radius of 3
//	float alpha1s[24];
//	float new_ps[24];
//	float new_qs[48];

	//radius of 4
//	float alpha1s[40];
//	float new_ps[40];
//	float new_qs[80];

	//radius of 5
//	float alpha1s[60];
//	float new_ps[60];
//	float new_qs[120];

	for(int h=0; h<=support_radius; h++)
	{
		for(int w=-support_radius; w<=support_radius; w++)
		{
			count++;

			if(count>support_radius && x+w>=0 && x+w<cols && y+h<rows)
			{
				j_pixel_idx = idx + h*cols + w;
				neibor_idx = count - support_radius - 1;
				j_nonlocal_idx = idx*n_size + neibor_idx;

				new_ps[neibor_idx] = d_p[j_nonlocal_idx] + sigma_p*(d_u[j_pixel_idx] - d_u[idx] - w*d_w[2*idx] - h*d_w[2*idx+1]); 

				norm_p += new_ps[neibor_idx]*new_ps[neibor_idx]; 

	/*			continue;
			
				new_p = d_p[j_nonlocal_idx] + sigma_p*(d_u[j_pixel_idx] - d_u[idx] - w*d_w[2*idx] - h*d_w[2*idx+1]); 
				norm_p += new_p*new_p; 
				d_p[j_nonlocal_idx] = new_p;

			//	new_q[0] = d_q[j_nonlocal_idx*2] + sigma_q*(-d_w[2*idx] + d_w[2*j_pixel_idx]);
			//	new_q[1] = d_q[j_nonlocal_idx*2+1] + sigma_q*(-d_w[2*idx+1] + d_w[2*j_pixel_idx+1]);
			//	norm_q += new_q[0]*new_q[0]+new_q[1]*new_q[1];
			//	d_q[j_nonlocal_idx*2] = new_q[0]; 
			//	d_q[j_nonlocal_idx*2+1] = new_q[1];				*/
			}
		}
	}

	if(TGV)
	{
		count = -1;
		for(int h=0; h<=support_radius; h++)
		{
			for(int w=-support_radius; w<=support_radius; w++)
			{
				count++;

				if(count>support_radius && x+w>=0 && x+w<cols && y+h<rows)
				{
					j_pixel_idx = idx + h*cols + w;
					neibor_idx = count - support_radius - 1;
					j_nonlocal_idx = idx*n_size + neibor_idx;

					new_qs[(neibor_idx)*2] = d_q[j_nonlocal_idx*2] + sigma_q*(-d_w[2*idx] + d_w[2*j_pixel_idx]);
					new_qs[(neibor_idx)*2+1] = d_q[j_nonlocal_idx*2+1] + sigma_q*(-d_w[2*idx+1] + d_w[2*j_pixel_idx+1]);

					norm_q += new_qs[(neibor_idx)*2]*new_qs[(neibor_idx)*2]+new_qs[(neibor_idx)*2+1]*new_qs[(neibor_idx)*2+1];

		/*			continue;
			
				//	new_p = d_p[j_nonlocal_idx] + sigma_p*(d_u[j_pixel_idx] - d_u[idx] - w*d_w[2*idx] - h*d_w[2*idx+1]); 
				//	norm_p += new_p*new_p; 
				//	d_p[j_nonlocal_idx] = new_p;

					new_q[0] = d_q[j_nonlocal_idx*2] + sigma_q*(-d_w[2*idx] + d_w[2*j_pixel_idx]);
					new_q[1] = d_q[j_nonlocal_idx*2+1] + sigma_q*(-d_w[2*idx+1] + d_w[2*j_pixel_idx+1]);
					norm_q += new_q[0]*new_q[0]+new_q[1]*new_q[1];
					d_q[j_nonlocal_idx*2] = new_q[0]; 
					d_q[j_nonlocal_idx*2+1] = new_q[1];				*/
				}
			}
		}
	}

	idx *= n_size;
	norm_p = sqrtf(norm_p);
	norm_q = sqrtf(norm_q);

	for(int n=0; n<n_size; n++) alpha1s[n] = d_alpha1[idx+n]*lambda_s;	

	for(int n=0; n<n_size; n++) d_p[idx+n] = new_ps[n]/fmaxf(1.0f, norm_p/alpha1s[n]);
	
	if(TGV)
	for(int n=0; n<n_size; n++)
	{
		alpha_ij_0 = lambda_a*alpha1s[n];
		d_q[(idx+n)*2] = new_qs[2*n]/fmaxf(1.0f, norm_q/alpha_ij_0);
		d_q[(idx+n)*2+1] = new_qs[2*n+1]/fmaxf(1.0f, norm_q/alpha_ij_0);	
	}

/*	return;

	for(int n=0; n<n_size; n++)
	{
		alpha_ij_1 = d_alpha1[idx+n]*lambda_s;	
		alpha_ij_0 = lambda_a*alpha_ij_1;

		d_p[idx+n] /= fmaxf(1.0f, norm_p/alpha_ij_1);
		d_q[(idx+n)*2] /= fmaxf(1.0f, norm_q/alpha_ij_0);
		d_q[(idx+n)*2+1] /= fmaxf(1.0f, norm_q/alpha_ij_0);
	}*/
}

struct MyAlpha1
{
	float* alpha1;
	int rad;
	int cols;
};

void ShowAlpha1(int event, int x, int y, int flags, void* userdata)
{

	if (event == cv::EVENT_LBUTTONDOWN)
	{
		const int rad = ((MyAlpha1*)userdata)->rad;
		const int cols = ((MyAlpha1*)userdata)->cols;
		const float* alpha1 = ((MyAlpha1*)userdata)->alpha1;
		const int width = (2*rad+1);
		const int height = (2*rad+1);
		const int n_size = 2*rad*(rad+1);

		cv::Mat weight;
		weight.create(height, width, CV_32F);

		int start_idx = (y*cols+x)*n_size;

		int count = -1;

		float sum = 0.0f;
		float sum_n = 0.0f;

		for(int h=0; h<=rad; h++)
		{
			for(int w=-rad; w<=rad; w++)
			{
				count++;

				if(count<=rad) continue; 
				
				weight.at<float>(h+rad, w+rad) = alpha1[start_idx + count-rad-1];

				sum += alpha1[start_idx + count-rad-1];
				
				if(x-w>=0 && y-h>=0)
				{
					//check the mirrored position of neighbor j around pixel i inside image 
					weight.at<float>(rad-h, rad-w) = alpha1[start_idx + count-rad-1 + (-h*cols-w)*n_size];
					sum_n += alpha1[start_idx + count-rad-1 + (-h*cols-w)*n_size];
				}
				else 
					weight.at<float>(rad-h, rad-w) = 0.0f;
	
			}
		}

		weight.at<float>(rad, rad) = 0.0f;

		cv::imshow("weight", weight);

		std::cout<<"x: "<<x <<" y: "<<y<<" sum+: "<<sum<< " sum-: "<<sum_n<<" 1/K: "<<1.0f/abs(sum-sum_n)<<"\n";
	}
}


// random color
rgb random_rgb()
{ 
  rgb c;  
  c.r = (uchar)random();
  c.g = (uchar)random();
  c.b = (uchar)random();

  return c;
}

// dissimilarity measure between pixels
double diff(cv::Mat& r, cv::Mat& g, cv::Mat& b, int x1, int y1, int x2, int y2) 
{
	const double rf = r.at<double>(y1, x1) - r.at<double>(y2, x2);
	const double gf = g.at<double>(y1, x1) - g.at<double>(y2, x2);
	const double bf = b.at<double>(y1, x1) - b.at<double>(y2, x2);

//	return sqrt(rf*rf+gf*gf+bf*bf);
//	return (abs(rf)+abs(gf)+abs(bf)) <= 1.0 ? 0.0 : (abs(rf)+abs(gf)+abs(bf));
	return (abs(rf)+abs(gf)+abs(bf));
//	return pow(pow(abs(rf), 1.5)+pow(abs(gf), 1.5)+pow(abs(bf), 1.5), 1/1.5);
//	return 255.f*max(max(abs(rf), abs(gf)), abs(bf));
}

// breath first search of mst
void aggregateCostFromParent(double* agg_cost, mst_graph_t& mst, std::vector<int>& vertices_vec, std::vector<int>& bfs_order, const float gamma, const int root=0)
{
	for(auto& node_id : bfs_order)
	{
		const int node_pixel_id = vertices_vec[node_id];

		for(auto& child_id : mst[node_id].children_indices)
		{
			const int child_pixel_id = vertices_vec[child_id];

			std::pair<mst_edge_descriptor, bool> edge_bool = boost::edge(child_id, node_id, mst);
	
			agg_cost[child_pixel_id] = mst[edge_bool.first].weight*agg_cost[node_pixel_id] + mst[edge_bool.first].weight2*agg_cost[child_pixel_id];
		}
	}

/*
	return;

	std::queue<int> vertices_queue;

	//root node
	vertices_queue.push(root);

	std::vector<uchar> color(boost::num_vertices(mst), 0); // white

	color[root] = 1;	//gray

	while( !vertices_queue.empty() )
	{
		// parent
		const int p = vertices_queue.front();	

		const int p_pixel_id = vertices_vec[p];	

		//const int p_pixel_id = p;	

		vertices_queue.pop();	

		boost::graph_traits<mst_graph_t>::adjacency_iterator ai, a_end;		

		boost::tie(ai, a_end) = boost::adjacent_vertices(p, mst); 


		for (; ai != a_end; ++ai) 	
		{
			if(color[*ai] == 0) //white
			{
				color[*ai] = 1;	//gray

				const int c_pixel_id = vertices_vec[*ai];

				// edge between parent and child
				std::pair<mst_edge_descriptor, bool> edge_bool = boost::edge(p, *ai, mst);

				agg_cost[c_pixel_id] = mst[edge_bool.first].weight*agg_cost[p_pixel_id] + mst[edge_bool.first].weight2*agg_cost[c_pixel_id];
	
				vertices_queue.push(*ai);	
			}
		}
	}

*/
}

struct abc
{
	float a;
	float b;
	float c;
};

float compute3DLabelCost(float* cost_vol, abc& abc, const int pixel_id, const int max_disp, const int width, const int img_size)
{
	const float disp = (pixel_id % width)*abc.a + (pixel_id / width)*abc.b + abc.c;

	const float disp_ceil = ceil(disp);

	const float disp_floor = floor(disp);

	const int disp_ceil_int = (int) disp_ceil;

	const int disp_floor_int = (int) disp_floor;

	if( (disp_ceil_int >= max_disp) || (disp_floor_int < 0) ) return 0.5f;	//mc-cnn cost volume disp range [0, ..., max_disp-1] ?

	return (disp_ceil-disp)*cost_vol[disp_floor_int*img_size+pixel_id] + (disp-disp_floor)*cost_vol[disp_ceil_int*img_size+pixel_id]; 
}

// recursive 
void aggregateCostFromChildren(mst_vertex_descriptor self, mst_vertex_descriptor parent, mst_graph_t& mst, double* cur_agg_cost,
				std::vector<int>& vertices_vec, std::vector<int>& bfs_order, float* cost_vol, abc* abc_map, abc& test_label, const int max_disp, const int width, 
				const int img_size, const float gamma)
{
#if 1
	for(int i=bfs_order.size()-1; i>0; i--)
	{
		const int node_id = bfs_order[i];
		const int parent_id = mst[node_id].parent_idx;
		const int pixel_id = vertices_vec[node_id];
		const int parent_pixel_id = vertices_vec[parent_id];	

		//self
		cur_agg_cost[pixel_id] += compute3DLabelCost(cost_vol, test_label, pixel_id, max_disp, width, img_size);

		std::pair<mst_edge_descriptor, bool> edge_bool = boost::edge(parent_id, node_id, mst);
	
		cur_agg_cost[parent_pixel_id] += mst[edge_bool.first].weight*cur_agg_cost[pixel_id];
	}

	cur_agg_cost[vertices_vec[bfs_order[0]]] += compute3DLabelCost(cost_vol, test_label, vertices_vec[bfs_order[0]], max_disp, width, img_size);

#else

	const int pixel_id = vertices_vec[self];

	double cost = compute3DLabelCost(cost_vol, test_label, pixel_id, max_disp, width, img_size);

	if( (boost::out_degree(self, mst) > 1) || (self == parent) ) 	// recursion if not leaf node
	{
		boost::graph_traits<mst_graph_t>::adjacency_iterator ai, a_end;		

		boost::tie(ai, a_end) = boost::adjacent_vertices(self, mst); 

		for (; ai != a_end; ++ai) 	
		{
			if( parent != *ai )
			{			
				std::pair<mst_edge_descriptor, bool> edge_bool = boost::edge(*ai, self, mst);

				aggregateCostFromChildren(*ai, self, mst, cur_agg_cost, vertices_vec, bfs_order, cost_vol, abc_map, test_label, max_disp, width, img_size, gamma);

				cost += mst[edge_bool.first].weight*cur_agg_cost[vertices_vec[*ai]];
			}
		}	
	}

	cur_agg_cost[pixel_id] = cost;
#endif
}

void aggregateCostFromChildrenNormFactor(mst_vertex_descriptor self, mst_vertex_descriptor parent, mst_graph_t& mst, 
					 double* cost_norm_factor, std::vector<int>& vertices_vec, std::vector<int>& bfs_order, const float gamma)
{
	for(int i=bfs_order.size()-1; i>0; i--)
	{
		const int node_id = bfs_order[i];
		const int parent_id = mst[node_id].parent_idx;
		const int pixel_id = vertices_vec[node_id];
		const int parent_pixel_id = vertices_vec[parent_id];	

		//self
		cost_norm_factor[pixel_id] += 1.0;

		std::pair<mst_edge_descriptor, bool> edge_bool = boost::edge(parent_id, node_id, mst);
	
		cost_norm_factor[parent_pixel_id] += mst[edge_bool.first].weight*cost_norm_factor[pixel_id];
	}

	cost_norm_factor[vertices_vec[0]] += 1.0;

/*	return;

	const int pixel_id = vertices_vec[self];

	double cost = 1.0;

	if( (boost::out_degree(self, mst) > 1) || (self == parent) ) 	// recursion if not leaf node
	{
		boost::graph_traits<mst_graph_t>::adjacency_iterator ai, a_end;		

		boost::tie(ai, a_end) = boost::adjacent_vertices(self, mst); 

		for (; ai != a_end; ++ai) 	
		{
			if( parent != *ai )
			{			
				std::pair<mst_edge_descriptor, bool> edge_bool = boost::edge(*ai, self, mst);

				aggregateCostFromChildrenNormFactor(*ai, self, mst, cost_norm_factor, vertices_vec, gamma);

				//cost += exp( -mst[edge_bool.first].weight*gamma )*cost_norm_factor[vertices_vec[*ai]];
				cost += mst[edge_bool.first].weight*cost_norm_factor[vertices_vec[*ai]];
				//cost += cost_norm_factor[vertices_vec[*ai]];
			}
		}	
	}

	cost_norm_factor[pixel_id] = cost;
*/
}

void MSTCostAggregationAndLabelUpdate(double* cost_norm_factor, double* min_cost, double* agg_cost, mst_graph_t& mst, abc* abc_map, abc& test_label, 
				      std::vector<int>& vertices_vec, std::vector<int>& bfs_order, float* cost_vol, float* disp_u, float* L, const float theta_inv, const float lambda_d, 
				      const int max_disp, const int width, const int height, const int img_size, const float gamma, const bool norm=false)
{
	// zero agg_cost in tree due to OpenMP
	for(auto& pixel_idx : vertices_vec) agg_cost[pixel_idx] = 0.0;

	// leaf to root cost aggregation
	aggregateCostFromChildren(0, 0, mst, agg_cost, vertices_vec, bfs_order, cost_vol, abc_map, test_label, max_disp, width, img_size, gamma);

	aggregateCostFromParent(agg_cost, mst, vertices_vec, bfs_order, gamma);	

	// update 3d labels
	for(auto& pixel_idx : vertices_vec)
	{
		double cost = agg_cost[pixel_idx];

		if(norm)
		{
			cost *= cost_norm_factor[pixel_idx];
		
			//const double disp_a = ( (pixel_idx % width)*test_label.a + (pixel_idx / width)*test_label.b + test_label.c ) / (max_disp-1.0);

			//const double u_a = disp_u[pixel_idx] - disp_a;

			//cost = lambda_d*cost + L[pixel_idx]*u_a + theta_inv*u_a*u_a*0.5f;
		}

		if( cost < min_cost[pixel_idx] )
		{
			min_cost[pixel_idx] = cost;
			abc_map[pixel_idx].a = test_label.a;
			abc_map[pixel_idx].b = test_label.b;
			abc_map[pixel_idx].c = test_label.c;
		}			
	}
}

void ComputeMSTCostNormFactor(double* cost_norm_factor, mst_graph_t& mst, std::vector<int>& vertices_vec, std::vector<int>& bfs_order, const float gamma)
{
	//const int last_idx = boost::num_vertices(mst)-1;

	aggregateCostFromChildrenNormFactor(0, 0, mst, cost_norm_factor, vertices_vec, bfs_order, gamma);

	aggregateCostFromParent(cost_norm_factor, mst, vertices_vec, bfs_order, gamma);
}

void LabelToDisp(abc* abc_map, cv::Mat& disp, const int height, const int width, const int max_disp)
{
//	StartTimer();
#pragma omp parallel for
	for(int i=0; i<height*width; i++)
	{
		abc* abc_ptr = abc_map + i;
			
	        disp.ptr<float>(0)[i] = max(0.0f, min(1.0f, ( (i % width)*abc_ptr->a + (i / width)*abc_ptr->b + abc_ptr->c )/(max_disp-1.0f) ) );	
		//disp.ptr<float>(0)[i] = ( (i % width)*abc_ptr->a + (i / width)*abc_ptr->b + abc_ptr->c );			
	}

//	std::cout<<"label to disp time "<<GetTimer()<<"\n";
}


void ShowWeights(int event, int x, int y, int flags, void* userdata)
{
	if (event == cv::EVENT_LBUTTONDOWN)
	{
		const int rows = ((MyData*)userdata)->height;
		const int cols = ((MyData*)userdata)->width;
		std::vector<std::vector<int>>* mst_vertices_vec = ((MyData*)userdata)->mst_vertices_vec;
		std::vector<mst_graph_t>* mst_vec = ((MyData*)userdata)->mst_vec;
		//mst_graph_t* mst = ((MyData*)userdata)->mst;

		int idx = y*cols+x;

		cv::Mat weights_d;
		weights_d.create(rows, cols, CV_64F);
		cv::Mat weights_f;

		for(int i=0; i<(*mst_vertices_vec).size(); i++)
		{
			std::vector<int>::iterator it = std::find( (*mst_vertices_vec)[i].begin(), (*mst_vertices_vec)[i].end(), idx );

			if ( it != (*mst_vertices_vec)[i].end() )
			{
				weights_d = 0.0;			
				weights_d.at<double>(y,x) = 1.0;

				std::queue<int> vertices_queue;

				const int root = std::distance((*mst_vertices_vec)[i].begin(),it);

				//root node
				vertices_queue.push(root);

				std::vector<uchar> color(boost::num_vertices((*mst_vec)[i]), 0); // white

				color[root] = 1;	//gray

				while( !vertices_queue.empty() )
				{
					// parent
					const int p = vertices_queue.front();	

					const int p_pixel_id = (*mst_vertices_vec)[i][p];	

					vertices_queue.pop();	

					boost::graph_traits<mst_graph_t>::adjacency_iterator ai, a_end;		

					boost::tie(ai, a_end) = boost::adjacent_vertices(p, (*mst_vec)[i]); 

					for (; ai != a_end; ++ai) 	
					{
						if(color[*ai] == 0) //white
						{
							color[*ai] = 1;	//gray

							const int c_pixel_id = (*mst_vertices_vec)[i][*ai];

							// edge between parent and child
							std::pair<mst_edge_descriptor, bool> edge_bool = boost::edge(p, *ai, (*mst_vec)[i]);

							weights_d.ptr<double>(0)[c_pixel_id] = (*mst_vec)[i][edge_bool.first].weight*weights_d.ptr<double>(0)[p_pixel_id];

							vertices_queue.push(*ai);	
						}
					}
				}

				weights_d.convertTo(weights_f, CV_32F);

				//cv::Mat small;
				//cv::resize(weights, small, cv::Size(), 0.5, 0.5);
				cv::imshow("weights", weights_f);
				std::cout<<"x: "<<x <<" y: "<<y<<"\n";
			}
		}
	}
}


void NonlocalTGVDenoise(cv::Mat& src, float* d_w, float* d_p, float* d_q, float* d_alpha1, 
			float* d_L, int rad, float lambda_s, float lambda_a, float lambda_d, const int cols, 
			const int rows, dim3& blockSize, dim3& gridSize, const int iterations = 200, const bool TGV=true)
{

	float* d_src = NULL;
	float* d_dst = NULL;
	cudaMalloc(&d_src, sizeof(float)*cols*rows);
	cudaMalloc(&d_dst, sizeof(float)*cols*rows);
	
	cudaMemcpy(d_src, src.ptr<float>(0), sizeof(float)*rows*cols, cudaMemcpyHostToDevice);
	cudaMemcpy(d_dst, d_src, sizeof(float)*rows*cols, cudaMemcpyDeviceToDevice);

	InitNL2TGV<<<gridSize, blockSize>>>(d_L, d_w, d_p, d_q, rad, cols, rows);

	//cv::Mat tmp;	tmp.create(rows, cols, CV_32F);

	for(int j=0; j<iterations; j++)
	{
		
		NL2TGV_dualUpdate<<<gridSize, blockSize>>>(d_dst, d_w, d_p, d_q, d_alpha1, rad, lambda_s, lambda_a, cols, rows, TGV);

		NL2TGV_primalUpdate<<<gridSize, blockSize>>>(d_dst, d_w, d_p, d_q, d_src, d_L, 1.0f, rad, cols, rows, TGV);


/*		std::cout<<j+1<<"\n";
		cudaMemcpy(tmp.ptr<float>(0), d_dst, sizeof(float)*rows*cols, cudaMemcpyDeviceToHost);

		cv::imshow("tmp", tmp);
		cv::waitKey(30);
*/
	}

	cudaMemcpy(src.ptr<float>(0), d_dst, sizeof(float)*rows*cols, cudaMemcpyDeviceToHost);

	cudaFree(d_src);
	cudaFree(d_dst);
}


/*
 * Segment an image
 *
 * Returns a color image representing the segmentation.
 *
 * im: image to segment.
 * sigma: to smooth the image.
 * c: constant for treshold function.
 * min_size: minimum component size (enforced by post-processing stage).
 * num_ccs: number of connected components in the segmentation.
 */
void segment_image_other_init(cv::Mat& r, cv::Mat& g, cv::Mat& b, std::vector<mst_graph_t>& mst_vec, std::vector<std::vector<int>>& mst_vertices_vec,
		              tree_graph_t& tree_g, abc* abc_map, cv::Mat& cost_norm_factor, std::vector<std::vector<int>>& bfs_order_vec, 
			      float c, int min_size, const int max_disp, float gamma, const int filter_size=3) 
{
	const int width = r.cols;
	const int height = r.rows;
	const int img_size = width*height;

	mst_vec.clear();
	mst_vertices_vec.clear();
	bfs_order_vec.clear();

	cv::Mat rs, gs, bs;

#if 1
	cv::medianBlur(r, rs, filter_size);
	cv::medianBlur(g, gs, filter_size);
	cv::medianBlur(b, bs, filter_size);
#endif

//	r.convertTo(rs, CV_64F, 255.);
//	g.convertTo(gs, CV_64F, 255.);
//	b.convertTo(bs, CV_64F, 255.);

	rs.convertTo(rs, CV_64F);
	gs.convertTo(gs, CV_64F);
	bs.convertTo(bs, CV_64F);

//	rs = r; gs = g; bs= b;

	// build graph
	edge *edges = new edge[img_size*4];
	int num = 0;
	for (int y = 0; y < height; y++) 
	{
		for (int x = 0; x < width; x++) 
		{
			if (x < width-1) 
			{
				edges[num].a = y * width + x;
				edges[num].b = y * width + (x+1);
				edges[num].w = diff(rs, gs, bs, x, y, x+1, y);
				num++;
			}

			if (y < height-1) 
			{
				edges[num].a = y * width + x;
				edges[num].b = (y+1) * width + x;
				edges[num].w = diff(rs, gs, bs, x, y, x, y+1);
				num++;
			}

#if 0
			if ((x < width-1) && (y < height-1)) 
			{
				edges[num].a = y * width + x;
				edges[num].b = (y+1) * width + (x+1);
				edges[num].w = sqrt(2.0f)*diff(rs, gs, bs, x, y, x+1, y+1);
				num++;
			}

			if ((x < width-1) && (y > 0)) 
			{
				edges[num].a = y * width + x;
				edges[num].b = (y-1) * width + (x+1);
				edges[num].w = sqrt(2.0f)*diff(rs, gs, bs, x, y, x+1, y-1);
				num++;
			}
#endif
		}
	}


	int* mst_edge_mask = new int[num];
	std::memset(mst_edge_mask, 0, sizeof(int)*num);

	universe *u = segment_graph(img_size, num, edges, mst_edge_mask, c);

	// post process small components, need to remove at least components of size 1 
	// other MST edges do not include them
#if 1
	min_size = min_size<2 ? 2 : min_size;
	for (int i = 0; i < num; i++) 
	{
		int a = u->find(edges[i].a);
		int b = u->find(edges[i].b);
		if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size)))
		{
//			int size_min = std::min(u->size(a), u->size(b));

			u->join(a, b);
			u->find(a);
			u->find(b);
			mst_edge_mask[i] = 1; //there are components of size 1

			//if(size_min > 50) 
			//	edges[i].w += 15;
		}
	}
#endif


#if 1
	cv::Mat out;
	out.create(height, width, CV_8UC3);
	// pick random colors for each component
	rgb *colors = new rgb[img_size];
	for (int i = 0; i < img_size; i++) colors[i] = random_rgb();
	for (int y = 0; y < height; y++) 
	{
		for (int x = 0; x < width; x++) 
		{
			const int idx = y*width+x;
			int comp = u->find(idx);

			out.at<cv::Vec3b>(y, x).val[0] = colors[comp].b;
			out.at<cv::Vec3b>(y, x).val[1] = colors[comp].g;
			out.at<cv::Vec3b>(y, x).val[2] = colors[comp].r;

			//int xp = comp%width;
			//int yp = comp/width;

			//circle(out, cv::Point(xp, yp), 5, cv::Scalar(255,255,255));
		}
	} 

	cv::imshow("segment", out);
	cv::waitKey(50);
	delete [] colors;  
#endif
	//StartTimer();
	// find representative ids
	std::vector<int> rep_id_vec;

	// connected component ids
	std::vector<int> cc_ids(img_size);

	// id in tree
	std::vector<int> id_in_tree(img_size);

	mst_vertices_vec.resize( u->num_sets() );

	for(int i=0; i<img_size; i++) 
	{
		std::vector<int>::iterator it = std::find( rep_id_vec.begin(), rep_id_vec.end(), u->find(i) );

		if ( it != rep_id_vec.end() )
		{
			cc_ids[i] = std::distance(rep_id_vec.begin(), it);
		}
		else
		{
			cc_ids[i] = rep_id_vec.size();
			rep_id_vec.push_back(u->find(i));
		}

		// partition pixel vertices
		id_in_tree[i] = mst_vertices_vec[ cc_ids[i] ].size();	

		mst_vertices_vec[ cc_ids[i] ].push_back(i);
		
		//std::cout<<cc_ids[i]<<" ";
	}

	//std::cout<<"vector time: "<<GetTimer()<<"\n";
	//std::cout<<"id vector size: "<<rep_id_vec.size()<<"\n";


	//build tree adjacency list
	//StartTimer();

	for(int i = 0; i < num; i++)
	{
		const int a = u->find(edges[i].a); 
		const int b = u->find(edges[i].b);

		//edge connects 2 components
		if (a != b) boost::add_edge(cc_ids[a], cc_ids[b], tree_g);
	}

	//std::cout<<"build tree graph time: "<<GetTimer()<<"\n";


	//StartTimer();
	// build each MST
	mst_vec.resize( u->num_sets() );

	for(int i = 0; i < num; i++)
	{
		if(mst_edge_mask[i] == 1)	//MST edge
		{
			const int tree_id = cc_ids[edges[i].b];

			mst_edge_descriptor e; bool inserted;

			boost::tie(e, inserted) = boost::add_edge(id_in_tree[edges[i].a], id_in_tree[edges[i].b], mst_vec[tree_id]);

			mst_vec[tree_id][e].weight = exp(-edges[i].w*gamma);	
			mst_vec[tree_id][e].weight2 = 1.0f - mst_vec[tree_id][e].weight*mst_vec[tree_id][e].weight; 
		}
	}


	// BFS
	bfs_order_vec.clear();
	bfs_order_vec.resize( mst_vec.size() );

	for(int t=0; t<mst_vec.size(); t++)
	{
		std::queue<int> vertices_queue;

		//root node
		vertices_queue.push(0);

		bfs_order_vec[t].push_back(0);

		mst_vec[t][0].parent_idx = 0;

		std::vector<uchar> color(boost::num_vertices(mst_vec[t]), 0); // white

		color[0] = 1;	//gray
	
		while( !vertices_queue.empty() )
		{
			// parent
			const int p = vertices_queue.front();	

			vertices_queue.pop();	

			boost::graph_traits<mst_graph_t>::adjacency_iterator ai, a_end;		

			boost::tie(ai, a_end) = boost::adjacent_vertices(p, mst_vec[t]); 

			for (; ai != a_end; ++ai) 	
			{
				if(color[*ai] == 0) //white
				{
					color[*ai] = 1;	//gray

					// edge between parent and child
					std::pair<mst_edge_descriptor, bool> edge_bool = boost::edge(p, *ai, mst_vec[t]);

					vertices_queue.push(*ai);	

					bfs_order_vec[t].push_back(*ai);

					mst_vec[t][*ai].parent_idx = p;
					mst_vec[t][p].children_indices.push_back(*ai);
				}
			}
		}		
	}


	//std::cout<<"build MSTs time: "<<GetTimer()<<"\n";

	//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	//std::default_random_engine generator(seed);
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
	auto dice = std::bind (distribution, generator);

	//StartTimer();
	//srand(0);
	// initialize abc map
	for(int y=0; y<height; y++)
	{
		for(int x=0; x<width; x++)
		{
			const int idx = y*width + x;
			const float d = dice()*max_disp;

			float x1, x2, nx, ny, nz;

			while(true)
			{
				while(true)
				{
					x1 = dice();
					x2 = dice();

					if( x1*x1 + x2*x2 < 1.0f ) break;
				}

				nx = 2.0f*x1*sqrt(1.0f - x1*x1 - x2*x2);
				ny = 2.0f*x2*sqrt(1.0f - x1*x1 - x2*x2);

				//draw normal samples more restrictively
				//if(nx*nx+ny*ny<=0.25f) 
					break;
			}

			nz = sqrt(1.0f - nx*nx - ny*ny);

			abc_map[idx].a = -nx/nz;
			abc_map[idx].b = -ny/nz;
			abc_map[idx].c = (nx*x+ny*y+nz*d)/nz;
		}
	}

	cost_norm_factor.create(height, width, CV_64F);
	cost_norm_factor = 0.0;

#if 0
	MyData my_data;
	my_data.width = width;
	my_data.height = height;
	my_data.mst_vec = &mst_vec;
	my_data.mst_vertices_vec = &mst_vertices_vec;

	cv::setMouseCallback("segment", ShowWeights, (void*)&my_data);
	cv::waitKey(0);
#endif

	//StartTimer();

	// compute normalization weight factor
	// go through each tree
	double* cost_norm_factor_ptr = cost_norm_factor.ptr<double>(0);
	
#pragma omp parallel for schedule(dynamic) num_threads(THREADS)
	for(int tree_id = 0; tree_id < mst_vec.size(); tree_id++)
	{
		ComputeMSTCostNormFactor(cost_norm_factor_ptr, mst_vec[tree_id], mst_vertices_vec[tree_id], bfs_order_vec[tree_id], gamma);


/*		double minval, maxval;

		cv::minMaxLoc(cost_norm_factor, &minval, &maxval);

		if(maxval != minval)
			cost_norm_factor = (cost_norm_factor-minval)/(maxval-minval);
		

		cv::imshow("cost norm factor", cost_norm_factor);
		cv::waitKey(0);
	
		cost_norm_factor = 0.0f;*/
	}
	//std::cout<<"cost norm factor time: "<<GetTimer()<<"\n";

	for(int i=0; i<img_size; i++) cost_norm_factor_ptr[i] = 1.0/cost_norm_factor_ptr[i];

#if 0
	cv::Mat norm_mask;
	norm_mask = cost_norm_factor < 0.0005;
	cv::imshow("norm_mask", norm_mask); cv::waitKey(0);

	uchar* mask_ptr = norm_mask.ptr<uchar>(0);

	StartTimer();

	for(int i=0; i<img_size; i++)
	{
		if( mask_ptr[i] == 255 )
		{
			int node_id = id_in_tree[i];
			int tree_id = cc_ids[i];

			boost::graph_traits<mst_graph_t>::adjacency_iterator ai, a_end;		

			boost::tie(ai, a_end) = boost::adjacent_vertices(node_id, mst_vec[tree_id]); 

			for (; ai != a_end; ++ai) 
			{
				std::pair<mst_edge_descriptor, bool> edge_bool = boost::edge(node_id, *ai, mst_vec[tree_id]);
				mst_vec[tree_id][edge_bool.first].weight = min(1.0, mst_vec[tree_id][edge_bool.first].weight*1.2);
				mst_vec[tree_id][edge_bool.first].weight2 = 1.0 - pow(mst_vec[tree_id][edge_bool.first].weight, 2.0);
			}

		}

	}

	std::cout<<"fix weight time: "<<GetTimer()<<"\n";
#endif


	delete[] mst_edge_mask;
	delete [] edges;
	delete u;
}


void MST_PMS(std::vector<mst_graph_t>& mst_vec, std::vector<std::vector<int>>& mst_vertices_vec, std::vector<std::vector<int>>& bfs_order_vec,
	     tree_graph_t& tree_g, abc* abc_map, double* min_cost, double* agg_cost, double* cost_norm_factor, float* cost_vol, float* disp_u, 
	     float* L, const float theta_inv, const float lambda_d, const int max_disp, const int width, const int height, 
	     const int img_size, const float gamma, std::default_random_engine& generator, 
	     std::uniform_real_distribution<float>& distribution, const bool norm = false)
{
	StartTimer();

	auto dice = std::bind (distribution, generator);

	//std::mutex mutex;

	// go through each tree
#pragma omp parallel for schedule(dynamic) num_threads(THREADS)
	for(int tree_id = 0; tree_id < mst_vec.size(); tree_id++)
	{
		// SPATIAL PROPAGATION
		boost::graph_traits<tree_graph_t>::adjacency_iterator ai, a_end;
		boost::tie(ai, a_end) = boost::adjacent_vertices(tree_id, tree_g);

		// random sample label from each neighbor tree
		for (; ai != a_end; ++ai) 
		{
			const int test_pixel_idx = mst_vertices_vec[*ai][(int)((dice()+1.0f)*0.5f*mst_vertices_vec[*ai].size())];

			abc test_label;
			//mutex.lock();
			test_label.a = abc_map[test_pixel_idx].a;
			test_label.b = abc_map[test_pixel_idx].b;	
			test_label.c = abc_map[test_pixel_idx].c;
			//mutex.unlock();

			// test label and update
			MSTCostAggregationAndLabelUpdate(cost_norm_factor, min_cost, agg_cost, mst_vec[tree_id], abc_map, test_label, 
							 mst_vertices_vec[tree_id], bfs_order_vec[tree_id], cost_vol, disp_u, L, theta_inv, lambda_d, max_disp, width, 
							 height, img_size, gamma, norm);
		}

		// RANDOM REFINEMENT
		//pick a random node in the tree
		const int test_pixel_idx = mst_vertices_vec[tree_id][std::rand() % mst_vertices_vec[tree_id].size()];

		const float px = (float)(test_pixel_idx % width);
		const float py = (float)(test_pixel_idx / width);

		abc* test_label_ptr = abc_map + test_pixel_idx;

		const float nz = 1.0f / sqrt(test_label_ptr->a*test_label_ptr->a + test_label_ptr->b*test_label_ptr->b + 1.0f);
		const float nx = -test_label_ptr->a * nz;
		const float ny = -test_label_ptr->b * nz;

		const float d = px*test_label_ptr->a + py*test_label_ptr->b + test_label_ptr->c;

		float max_n = 1.0f;
		float max_d = 0.5f*max_disp;

		for(; max_d > 0.1f; max_d *= 0.5f, max_n *= 0.5f)
		{
			float rand_d = d + dice()*max_d;

			if( rand_d < 0.0f || rand_d > (float)max_disp ) continue;

			float rand_nx = nx + dice()*max_n;
			float rand_ny = ny + dice()*max_n;
			float rand_nz = nz + dice()*max_n;

			const float norm_inv = 1.0f / sqrt(rand_nx*rand_nx + rand_ny*rand_ny + rand_nz*rand_nz);

			rand_nx *= norm_inv;
			rand_ny *= norm_inv;
			rand_nz = abs(rand_nz*norm_inv);

			abc test_label;

			test_label.a = -rand_nx/rand_nz;
			test_label.b = -rand_ny/rand_nz;	
			test_label.c = (rand_nx*px + rand_ny*py + rand_nz*rand_d)/rand_nz;

			MSTCostAggregationAndLabelUpdate(cost_norm_factor, min_cost, agg_cost, mst_vec[tree_id], abc_map, test_label, 
							 mst_vertices_vec[tree_id], bfs_order_vec[tree_id], cost_vol, disp_u, L, theta_inv, lambda_d, max_disp, 
							 width, height, img_size, gamma, norm);
		}			
	}

	//std::cout<<"time: "<<GetTimer()<<"\n";
}

void leftRightConsistencyCheck(float* left, float* right, const int width, const int height, const int max_disp, const bool fill=true)
{

	cv::Mat mask;
	mask.create(height, width, CV_8U);
	mask = 0;

#pragma omp parallel for
	for(int y=0; y<height; y++)
	{
		for(int x=0; x<width; x++)
		{
			const int idx = y*width + x;

			const float d_f = left[idx];

			int d = (int)round(d_f);

			if(x-d>=0 && d >=0 && d<max_disp)
			{
				if( abs(d_f-right[idx-d]) > 1.0f )
				{
					mask.ptr<uchar>(0)[idx] = 1;
					left[idx] = 0.0f;
				}
			}
			else
			{
				mask.ptr<uchar>(0)[idx] = 1;
				left[idx] = 0.0f;
			}
		}
	}

	if(!fill) return;

#pragma omp parallel for
	for(int y=0; y<height; y++)
	{
		for(int x=0; x<width; x++)
		{
			const int idx = y*width + x;

			if(mask.ptr<uchar>(0)[idx] == 0) continue;

			//search left
			int i = 1;
			while(true)
			{
				if(x-i<0) break;
			
				if(mask.ptr<uchar>(0)[idx-i] == 0)
				{
					left[idx] = left[idx-i];
					mask.ptr<uchar>(0)[idx] = 0;
					break;
				}

				i++;
			}

			// search right
			i = 1;
			while(true)
			{
				if(x+i>=width) break;
			
				if(mask.ptr<uchar>(0)[idx+i] == 0)
				{
					if( left[idx+i] < left[idx] || mask.ptr<uchar>(0)[idx] == 1)
						left[idx] = left[idx+i];
					break;
				}

				i++;
			}
		}
	}
}


void PatchMatchStereoNL2TGV(const cv::Mat& leftImg, const cv::Mat& rightImg, int winRadius, const int Dmin, const int Dmax, 
				int iteration, float scale, bool showLeftDisp, cv::Mat& leftDisp, cv::Mat& rightDisp, 
				std::string& data_cost, std::string& smoothness_prior, std::string& left_name, std::string& right_name)
{
	cudaDeviceReset();

	const int cols = leftImg.cols;
	const int rows = leftImg.rows;
	unsigned int imgSize = (unsigned int)cols*rows;

	// kernels size
	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1)/blockSize.x, (rows + blockSize.x - 1)/blockSize.x);

	leftDisp.create(rows, cols, CV_32F);
	rightDisp.create(rows, cols, CV_32F);

#if 0
	if(data_cost == "MCCNN_fst")
	{
		int status = chdir("/home/lietang/mc-cnn-master");

		std::string cmd = "./main.lua mb fast -a predict -net_fname net/net_mb_fast_-a_train_all.t7 -left ../MiddEval3/" + left_name 
				+ " -right ../MiddEval3/"+ right_name + " -disp_max " + std::to_string(Dmax)+ " -sm_terminate cnn";
	
		system(cmd.c_str());

		status = chdir("/home/lietang/MiddEval3");		
	}
	else if(data_cost == "MCCNN_acrt")
	{
		int status = chdir("/home/lietang/mc-cnn-master");

		std::string cmd = "./main.lua mb slow -a predict -net_fname net/net_mb_slow_-a_train_all.t7 -left ../MiddEval3/" + left_name 
				+ " -right ../MiddEval3/"+ right_name + " -disp_max " + std::to_string(Dmax)+ " -sm_terminate cnn";
	
		system(cmd.c_str());

		status = chdir("/home/lietang/MiddEval3");	
	}
#endif


#if 1	// phenotyping
	if(data_cost == "MCCNN_fst")
	{
		int status = chdir("/home/lietang/mc-cnn-master");

		std::string cmd = "./main.lua mb fast -a predict -net_fname net/net_mb_fast_-a_train_all.t7 -left l.png -right r.png -disp_max " 
				  + std::to_string(Dmax) + " -sm_terminate cnn";
	
		status = system(cmd.c_str());
	}
	else if(data_cost == "MCCNN_acrt")
	{
		int status = chdir("/home/lietang/mc-cnn-master");

		std::string cmd = "./main.lua mb slow -a predict -net_fname net/net_mb_slow_-a_train_all.t7 -left l.png -right r.png -disp_max " 
				  + std::to_string(Dmax) + " -sm_terminate cnn";
	
		status = system(cmd.c_str());
	}
#endif




	//load mc-cnn raw cost volume, origional range (-1,1)
	int fd;
	float *left_cost_vol_mccnn, *right_cost_vol_mccnn;

	if(data_cost == "MCCNN_fst" || data_cost == "MCCNN_acrt")
	{
		fd = open("/home/lietang/mc-cnn-master/left.bin", O_RDONLY);
		left_cost_vol_mccnn = (float*)mmap(NULL, 1 * Dmax * rows * cols * sizeof(float), PROT_READ, MAP_SHARED, fd, 0);
		close(fd);
		fd = open("/home/lietang/mc-cnn-master/right.bin", O_RDONLY);
		right_cost_vol_mccnn = (float*)mmap(NULL, 1 * Dmax * rows * cols * sizeof(float), PROT_READ, MAP_SHARED, fd, 0);
		close(fd);
	}



	float *left_cost_vol_mccnn_w = new float[imgSize*Dmax];
	float *right_cost_vol_mccnn_w = new float[imgSize*Dmax];

	std::memcpy(left_cost_vol_mccnn_w, left_cost_vol_mccnn, imgSize*Dmax*sizeof(float));
	std::memcpy(right_cost_vol_mccnn_w, right_cost_vol_mccnn, imgSize*Dmax*sizeof(float));

	// BGR 2 grayscale
/*	cv::Mat left_gray;
	cv::Mat right_gray;
	cv::Mat left_gray_f;
	cv::Mat right_gray_f;

	cv::cvtColor(leftImg, left_gray, CV_BGR2GRAY);
	cv::cvtColor(rightImg, right_gray, CV_BGR2GRAY);

	left_gray.convertTo(left_gray_f, CV_32F, 1./255.);
	right_gray.convertTo(right_gray_f, CV_32F, 1./255.);


	double eps = pow(0.01, 2.0);//*255*255;	// SUPER IMPORTANT, too large = box filter, too small = guide image
	int win_rad_bf = 5;
	int gfsize = 2*win_rad_bf+1;

	for(int i=0; i<imgSize*Dmax; i++) if(isnan(left_cost_vol_mccnn_w[i])) left_cost_vol_mccnn_w[i] = 1.0f;
	for(int i=0; i<imgSize*Dmax; i++) if(isnan(right_cost_vol_mccnn_w[i])) right_cost_vol_mccnn_w[i] = 1.0f;

	costVolumeGuidedFilterOMP(right_gray_f, right_cost_vol_mccnn_w, cols, rows, imgSize, Dmax, eps, gfsize);
	costVolumeGuidedFilterOMP(left_gray_f, left_cost_vol_mccnn_w, cols, rows, imgSize, Dmax, eps, gfsize);
*/

#if 0
	cv::Mat disp_test;
	disp_test.create(rows, cols, CV_32F);
	disp_test = 0.0f;
	float* cost_vol = left_cost_vol_mccnn_w;
	float minimum, maximum;
	minimum = 1000.f;
	maximum = -1000.f;

	cv::Mat ambiguity;
	ambiguity.create(rows, cols, CV_32F);

	for(int y=0; y<rows; y++)
	{
		for(int x=0; x<cols; x++)
		{
			float min_cost = 100.f;
			
			int best_d = -1;

			float cost_sum = 0.0f;

			for(int d=0; d<Dmax; d++)
			{

				float cur_cost = cost_vol[d*imgSize+y*cols+x];

				if(cur_cost < min_cost)
				{
					min_cost = cur_cost;
					best_d = d;
				}					


				if(cur_cost > maximum)	
					maximum = cur_cost;

				if( !isnan(cur_cost) )
					//cost_sum += cur_cost > 0.5f ? 1.0f : cur_cost;
					cost_sum += cur_cost;
			}

			if(best_d != -1)
			{
				disp_test.at<float>(y,x) = (float)best_d;
				
				if(min_cost < minimum)
					minimum = min_cost;
				
			}

			ambiguity.at<float>(y,x) = cost_sum/Dmax;
			//std::cout<<cost_sum<<" ";
		}
	}

	std::cout<<"minimum: "<<minimum<<" maximum: "<<maximum<<"\n";

	disp_test /= (float)Dmax;

	double min_val, max_val;
	cv::minMaxLoc(ambiguity, &min_val, &max_val); 

	
	std::cout<<"cost max: "<<max_val<<"\n";


	//ambiguity /= (float)max_val;

	cv::Mat mask;
	mask = ambiguity < 0.8f;

	cv::imshow("mask", mask);

	cv::imshow("ambiguity", ambiguity);



/*	for(int y=0; y<rows; y++)
	{
		for(int x=0; x<cols; x++)
		{
			for(int d=0; d<Dmax; d++)
			{
				if(mask.at<uchar>(y,x) == 255)
				{
					int idx = d*imgSize + y*cols + x;
					//left_cost_vol_mccnn_w[idx] = 0.0f;
					cost_vol[idx] = 0.0f;
				}
			}
		}
	}
*/

	MyData my_data;
	my_data.cost_vol = cost_vol;
	my_data.width = cols;
	my_data.height = rows;
	my_data.depth = Dmax;
	my_data.disp = disp_test;

	cv::imshow("disp test", disp_test);
	cv::setMouseCallback("disp test", ShowSlice, (void*)&my_data);
	cv::waitKey(0);

	//return;
#endif 
	

	// split channels
	std::vector<cv::Mat> cvLeftBGR_v;
	std::vector<cv::Mat> cvRightBGR_v;

	cv::split(leftImg, cvLeftBGR_v);
	cv::split(rightImg, cvRightBGR_v);

	// BGR 2 grayscale
	cv::Mat cvLeftGray;
	cv::Mat cvRightGray;

	cv::cvtColor(leftImg, cvLeftGray, CV_BGR2GRAY);
	cv::cvtColor(rightImg, cvRightGray, CV_BGR2GRAY);


	cv::Ptr<cv::LineSegmentDetector> lsd;
	cv::Mat left_ls_mask, right_ls_mask;
	cv::Mat left_ls_mask_f, right_ls_mask_f;
	double gauss_sigma = 1.0;
	//if(smoothness_prior == "2TGV" || smoothness_prior == "TV")
	{

		// line segment detector 
		lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_ADV);	
		// Detect the lines
		std::vector<cv::Vec4f> lines_std;
		
		left_ls_mask.create(rows, cols, CV_8UC1);
		right_ls_mask.create(rows, cols, CV_8UC1);

		lsd->detect(cvLeftGray, lines_std);
	 
		for(auto & l : lines_std) cv::line(left_ls_mask, cv::Point2f(l[0], l[1]), cv::Point2f(l[2], l[3]), cv::Scalar(255,255,255), 1);		

	    	lsd->detect(cvRightGray, lines_std);

		for(auto & l : lines_std) cv::line(right_ls_mask, cv::Point2f(l[0], l[1]), cv::Point2f(l[2], l[3]), cv::Scalar(255,255,255), 1);		

		left_ls_mask.convertTo(left_ls_mask_f, CV_32FC1, 1./255.);
		right_ls_mask.convertTo(right_ls_mask_f, CV_32FC1, 1./255.);

		cv::GaussianBlur(left_ls_mask_f, left_ls_mask_f, cv::Size(0,0), gauss_sigma);
		cv::GaussianBlur(right_ls_mask_f, right_ls_mask_f, cv::Size(0,0), gauss_sigma);

	 	//cv::imshow("line segment left", left_ls_mask_f); cv::imshow("line segment right", right_ls_mask_f);
		//cv::waitKey(0); return;

	}

	// convert to float
	cv::Mat cvLeftB_f;
	cv::Mat cvLeftG_f;
	cv::Mat cvLeftR_f;
	cv::Mat cvRightB_f;
	cv::Mat cvRightG_f;
	cv::Mat cvRightR_f;
	cv::Mat cvLeftGray_f;
	cv::Mat cvRightGray_f;	

	cvLeftBGR_v[0].convertTo(cvLeftB_f, CV_32F, 1./255.);
	cvLeftBGR_v[1].convertTo(cvLeftG_f, CV_32F, 1./255.);
	cvLeftBGR_v[2].convertTo(cvLeftR_f, CV_32F, 1./255.);	

	cvRightBGR_v[0].convertTo(cvRightB_f, CV_32F, 1./255.);	
	cvRightBGR_v[1].convertTo(cvRightG_f, CV_32F, 1./255.);	
	cvRightBGR_v[2].convertTo(cvRightR_f, CV_32F, 1./255.);	

	cvLeftGray.convertTo(cvLeftGray_f, CV_32F, 1./255.);
	cvRightGray.convertTo(cvRightGray_f, CV_32F, 1./255.);


#pragma omp parallel for num_threads(THREADS)
	for(int i=0; i<imgSize*Dmax; i++) 
	{
		if(isnan(left_cost_vol_mccnn_w[i])) 
			left_cost_vol_mccnn_w[i] = 0.5f;
		else
			left_cost_vol_mccnn_w[i] = min(0.5f, left_cost_vol_mccnn_w[i]);
//			left_cost_vol_mccnn_w[i] = min(0.5f, (left_cost_vol_mccnn_w[i]+1.0f)*0.5f);
	}

#pragma omp parallel for num_threads(THREADS)
	for(int i=0; i<imgSize*Dmax; i++) 
	{
		if(isnan(right_cost_vol_mccnn_w[i])) 
			right_cost_vol_mccnn_w[i] = 0.5f;
		else
			right_cost_vol_mccnn_w[i] = min(0.5f, right_cost_vol_mccnn_w[i]);
//			right_cost_vol_mccnn_w[i] = min(0.5f, (right_cost_vol_mccnn_w[i]+1.0f)*0.5f);
	}

	// allocate GPU memory for cost volume
	cudaMalloc(&d_left_cost_vol, imgSize*Dmax*sizeof(float));
	cudaMalloc(&d_right_cost_vol, imgSize*Dmax*sizeof(float));
	cudaMemcpy(d_left_cost_vol, left_cost_vol_mccnn, imgSize*Dmax*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_right_cost_vol, right_cost_vol_mccnn, imgSize*Dmax*sizeof(float), cudaMemcpyHostToDevice);

	
	float* d_left_ambiguity = NULL;
	float* d_right_ambiguity = NULL;

	cudaMalloc(&d_left_ambiguity, imgSize*sizeof(float));
	cudaMalloc(&d_right_ambiguity, imgSize*sizeof(float));


	
	cv::Mat left_gray_gauss_cv32f, right_gray_gauss_cv32f;
	cv::GaussianBlur(cvLeftGray_f, left_gray_gauss_cv32f, cv::Size(0,0), gauss_sigma);
	cv::GaussianBlur(cvRightGray_f, right_gray_gauss_cv32f, cv::Size(0,0), gauss_sigma);
//	cv::imshow("gaussian left", left_gray_gauss_cv32f); cv::imshow("gaussian right", right_gray_gauss_cv32f);
//	cv::waitKey(0);
	float* d_left_gray_gauss = NULL;
	float* d_right_gray_gauss = NULL;
	cudaMalloc(&d_left_gray_gauss, imgSize*sizeof(float));
	cudaMalloc(&d_right_gray_gauss, imgSize*sizeof(float));
	cudaMemcpy(d_left_gray_gauss, left_gray_gauss_cv32f.ptr<float>(0), imgSize*sizeof(float), cudaMemcpyHostToDevice); 
	cudaMemcpy(d_right_gray_gauss, right_gray_gauss_cv32f.ptr<float>(0), imgSize*sizeof(float), cudaMemcpyHostToDevice);
		
		
	float* leftRImg_f = cvLeftR_f.ptr<float>(0);
	float* leftGImg_f = cvLeftG_f.ptr<float>(0);
	float* leftBImg_f = cvLeftB_f.ptr<float>(0);
	float* leftGrayImg_f = cvLeftGray_f.ptr<float>(0);
	float* rightRImg_f = cvRightR_f.ptr<float>(0);
	float* rightGImg_f = cvRightG_f.ptr<float>(0);
	float* rightBImg_f = cvRightB_f.ptr<float>(0);
	float* rightGrayImg_f = cvRightGray_f.ptr<float>(0);


	// allocate floating disparity map, plane normals and gradient image (global memory)
	float* dRDisp = NULL;
	float* dLDisp = NULL;
	float* dLPlanes = NULL;
	float* dRPlanes = NULL;

	float* dLCost = NULL;
	float* dRCost = NULL;

	float* dLGradX = NULL;
	float* dRGradX = NULL;
	float* dLGradY= NULL;
	float* dRGradY= NULL;
	// 45 deg
	float* dLGradXY= NULL;
	float* dRGradXY= NULL;
	// 135 deg
	float* dLGradYX= NULL;
	float* dRGradYX= NULL;

	cudaMalloc(&dRDisp, imgSize*sizeof(float));
	cudaMalloc(&dLDisp, imgSize*sizeof(float));
	// changed 3 to 2, remove nz
	cudaMalloc(&dRPlanes, 2*imgSize*sizeof(float));
	cudaMalloc(&dLPlanes, 2*imgSize*sizeof(float));

	cudaMalloc(&dRCost, imgSize*sizeof(float));
	cudaMalloc(&dLCost, imgSize*sizeof(float));

	cudaMalloc(&dRGradX, imgSize*sizeof(float));
	cudaMalloc(&dLGradX, imgSize*sizeof(float));
	cudaMalloc(&dRGradY, imgSize*sizeof(float));
	cudaMalloc(&dLGradY, imgSize*sizeof(float));
	cudaMalloc(&dRGradXY, imgSize*sizeof(float));
	cudaMalloc(&dLGradXY, imgSize*sizeof(float));
	cudaMalloc(&dRGradYX, imgSize*sizeof(float));
	cudaMalloc(&dLGradYX, imgSize*sizeof(float));
	
	// huber smoothing
	float* dLPlanesV = NULL; 
	float* dRPlanesV = NULL;
	float* dLDispV = NULL;
	float* dRDispV = NULL;
	float* dLPlanesPn = NULL;
	float* dRPlanesPn = NULL;
	float* dLDispPd = NULL;
	float* dRDispPd = NULL;
	float* dLWeight = NULL;
	float* dRWeight = NULL;

	cudaMalloc(&dLPlanesV, 2*imgSize*sizeof(float));
	cudaMalloc(&dRPlanesV, 2*imgSize*sizeof(float));
	cudaMalloc(&dLPlanesPn, 4*imgSize*sizeof(float));
	cudaMalloc(&dRPlanesPn, 4*imgSize*sizeof(float));
	cudaMalloc(&dLDispV, imgSize*sizeof(float));
	cudaMalloc(&dRDispV, imgSize*sizeof(float));
	cudaMalloc(&dLDispPd, 2*imgSize*sizeof(float));
	cudaMalloc(&dRDispPd, 2*imgSize*sizeof(float));
	cudaMalloc(&dLWeight, imgSize*sizeof(float));
	cudaMalloc(&dRWeight, imgSize*sizeof(float));
	
	// set dual variables to zero
	cudaMemset(dLPlanesPn, 0, 4*imgSize*sizeof(float));
	cudaMemset(dRPlanesPn, 0, 4*imgSize*sizeof(float));
	cudaMemset(dLDispPd, 0,  2*imgSize*sizeof(float));
	cudaMemset(dRDispPd, 0,  2*imgSize*sizeof(float));

	// occlusion mask
	int* dLOMask;
	int* dROMask;
	cudaMalloc(&dLOMask, imgSize*sizeof(int));
	cudaMalloc(&dROMask, imgSize*sizeof(int));

	// line segment mask
	float* d_left_ls_mask = NULL;
	float* d_right_ls_mask = NULL;

//	if(smoothness_prior == "2TGV" || smoothness_prior == "TV")
	{
		cudaMalloc(&d_left_ls_mask, imgSize*sizeof(float));
		cudaMalloc(&d_right_ls_mask, imgSize*sizeof(float));
		cudaMemcpy(d_left_ls_mask, left_ls_mask_f.ptr<float>(0), imgSize*sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_right_ls_mask, right_ls_mask_f.ptr<float>(0), imgSize*sizeof(float), cudaMemcpyHostToDevice);
	}

	cudaArray* lR_ca;
	cudaArray* lG_ca;
	cudaArray* lB_ca;
	cudaArray* lGray_ca;
	cudaArray* rR_ca;
	cudaArray* rG_ca;
	cudaArray* rB_ca;
	cudaArray* rGray_ca;
	cudaArray* lGradX_ca;
	cudaArray* rGradX_ca;
	cudaArray* lGradY_ca;
	cudaArray* rGradY_ca;
	cudaArray* lGradXY_ca;
	cudaArray* rGradXY_ca;
	cudaArray* lGradYX_ca;
	cudaArray* rGradYX_ca;

	cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaMallocArray(&lR_ca, &desc, cols, rows);
	cudaMallocArray(&lG_ca, &desc, cols, rows);
	cudaMallocArray(&lB_ca, &desc, cols, rows);
	cudaMallocArray(&lGray_ca, &desc, cols, rows);
	cudaMallocArray(&lGradX_ca, &desc, cols, rows);
	cudaMallocArray(&lGradY_ca, &desc, cols, rows);
	cudaMallocArray(&lGradXY_ca, &desc, cols, rows);
	cudaMallocArray(&lGradYX_ca, &desc, cols, rows);
	cudaMallocArray(&rR_ca, &desc, cols, rows);
	cudaMallocArray(&rG_ca, &desc, cols, rows);
	cudaMallocArray(&rB_ca, &desc, cols, rows);
	cudaMallocArray(&rGray_ca, &desc, cols, rows);
	cudaMallocArray(&rGradX_ca, &desc, cols, rows);
	cudaMallocArray(&rGradY_ca, &desc, cols, rows);
	cudaMallocArray(&rGradXY_ca, &desc, cols, rows);
	cudaMallocArray(&rGradYX_ca, &desc, cols, rows);

	cudaMemcpyToArray(lR_ca, 0, 0, leftRImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(lG_ca, 0, 0, leftGImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(lB_ca, 0, 0, leftBImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(lGray_ca, 0, 0, leftGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rR_ca, 0, 0, rightRImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rG_ca, 0, 0, rightGImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rB_ca, 0, 0, rightBImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpyToArray(rGray_ca, 0, 0, rightGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	
	// texture object
	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = lR_ca;

	// border: 0, clamp: border pixvel value, 
	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.normalizedCoords = 1;

	cudaTextureObject_t lR_to = 0;
	cudaCreateTextureObject(&lR_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t lG_to = 0;
	resDesc.res.array.array = lG_ca;
	cudaCreateTextureObject(&lG_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t lB_to = 0;
	resDesc.res.array.array = lB_ca;
	cudaCreateTextureObject(&lB_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t lGray_to = 0;
	resDesc.res.array.array = lGray_ca;
	cudaCreateTextureObject(&lGray_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rR_to = 0;
	resDesc.res.array.array = rR_ca;
	cudaCreateTextureObject(&rR_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rG_to = 0;
	resDesc.res.array.array = rG_ca;
	cudaCreateTextureObject(&rG_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rB_to = 0;
	resDesc.res.array.array = rB_ca;
	cudaCreateTextureObject(&rB_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rGray_to = 0;
	resDesc.res.array.array = rGray_ca;
	cudaCreateTextureObject(&rGray_to, &resDesc, &texDesc, NULL);

	// image gradient
	imgGradient_huber<<<gridSize, blockSize>>>( cols, rows, lGray_to, rGray_to, 
						   dLGradX, dRGradX, dLGradY, dRGradY,
						   dLGradXY, dRGradXY, dLGradYX, dRGradYX);

	// copy gradient back sobel x
	//cudaMemcpy(rightGrayImg_f, dRGradX, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
	//cudaMemcpy(leftGrayImg_f, dLGradX, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);

	/*cv::imshow("Left grad X", cvLeftGray_f);
	cv::imshow("Right grad X", cvRightGray_f);
	cv::waitKey(0);*/

	cudaMemcpyToArray(lGradX_ca, 0, 0, dLGradX, sizeof(float)*imgSize, cudaMemcpyDeviceToDevice);
	cudaMemcpyToArray(rGradX_ca, 0, 0, dRGradX, sizeof(float)*imgSize, cudaMemcpyDeviceToDevice);

	cudaTextureObject_t lGradX_to = 0;
	resDesc.res.array.array = lGradX_ca;
	cudaCreateTextureObject(&lGradX_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rGradX_to = 0;
	resDesc.res.array.array = rGradX_ca;
	cudaCreateTextureObject(&rGradX_to, &resDesc, &texDesc, NULL);

	// sobel y
	//cudaMemcpy(rightGrayImg_f, dRGradY, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
	//cudaMemcpy(leftGrayImg_f, dLGradY, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);

	/*cv::imshow("Left grad Y", cvLeftGray_f);
	cv::imshow("Right grad Y", cvRightGray_f);
	cv::waitKey(0);*/

	//cudaMemcpyToArray(lGradY_ca, 0, 0, leftGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	//cudaMemcpyToArray(rGradY_ca, 0, 0, rightGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);

	cudaMemcpyToArray(lGradY_ca, 0, 0, dLGradY, sizeof(float)*imgSize, cudaMemcpyDeviceToDevice);
	cudaMemcpyToArray(rGradY_ca, 0, 0, dRGradY, sizeof(float)*imgSize, cudaMemcpyDeviceToDevice);

	cudaTextureObject_t lGradY_to = 0;
	resDesc.res.array.array = lGradY_ca;
	cudaCreateTextureObject(&lGradY_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rGradY_to = 0;
	resDesc.res.array.array = rGradY_ca;
	cudaCreateTextureObject(&rGradY_to, &resDesc, &texDesc, NULL);

	// central difference 45 deg
	//cudaMemcpy(rightGrayImg_f, dRGradXY, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
	//cudaMemcpy(leftGrayImg_f, dLGradXY, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);

	/*cv::imshow("Left grad 45", cvLeftGray_f);
	cv::imshow("Right grad 45", cvRightGray_f);
	cv::waitKey(0);*/

	//cudaMemcpyToArray(lGradXY_ca, 0, 0, leftGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	//cudaMemcpyToArray(rGradXY_ca, 0, 0, rightGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);

	cudaMemcpyToArray(lGradXY_ca, 0, 0, dLGradXY, sizeof(float)*imgSize, cudaMemcpyDeviceToDevice);
	cudaMemcpyToArray(rGradXY_ca, 0, 0, dRGradXY, sizeof(float)*imgSize, cudaMemcpyDeviceToDevice);

	cudaTextureObject_t lGradXY_to = 0;
	resDesc.res.array.array = lGradXY_ca;
	cudaCreateTextureObject(&lGradXY_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rGradXY_to = 0;
	resDesc.res.array.array = rGradXY_ca;
	cudaCreateTextureObject(&rGradXY_to, &resDesc, &texDesc, NULL);

	// central difference 135 deg
	//cudaMemcpy(rightGrayImg_f, dRGradYX, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
	//cudaMemcpy(leftGrayImg_f, dLGradYX, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);

	/*cv::imshow("Left grad 135", cvLeftGray_f);
	cv::imshow("Right grad 135", cvRightGray_f);
	cv::waitKey(0);*/

	//cudaMemcpyToArray(lGradYX_ca, 0, 0, leftGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	//cudaMemcpyToArray(rGradYX_ca, 0, 0, rightGrayImg_f, sizeof(float)*imgSize, cudaMemcpyHostToDevice);

	cudaMemcpyToArray(lGradYX_ca, 0, 0, dLGradYX, sizeof(float)*imgSize, cudaMemcpyDeviceToDevice);
	cudaMemcpyToArray(rGradYX_ca, 0, 0, dRGradYX, sizeof(float)*imgSize, cudaMemcpyDeviceToDevice);

	cudaTextureObject_t lGradYX_to = 0;
	resDesc.res.array.array = lGradYX_ca;
	cudaCreateTextureObject(&lGradYX_to, &resDesc, &texDesc, NULL);

	cudaTextureObject_t rGradYX_to = 0;
	resDesc.res.array.array = rGradYX_ca;
	cudaCreateTextureObject(&rGradYX_to, &resDesc, &texDesc, NULL);

	// result
	cv::Mat cvLeftDisp_f, cvRightDisp_f;
	cvLeftDisp_f.create(rows, cols, CV_32F);
	cvRightDisp_f.create(rows, cols, CV_32F);

	cv::Mat cvLeftDispV_f, cvRightDispV_f;
	cvLeftDispV_f.create(rows, cols, CV_32F);
	cvRightDispV_f.create(rows, cols, CV_32F);


	// PMS + 2nd-order TGV
	// anisotropic diffusion tensor	2x2 matrix G
	float* d_left_G = NULL;
	float* d_right_G = NULL;
	cudaMalloc(&d_left_G, imgSize*4*sizeof(float));
	cudaMalloc(&d_right_G, imgSize*4*sizeof(float));

	// second order primal variable v
	float* d_left_v = NULL;
	float* d_right_v = NULL;
	cudaMalloc(&d_left_v, imgSize*2*sizeof(float));
	cudaMalloc(&d_right_v, imgSize*2*sizeof(float));

	// dual variable for u
	float* d_left_p = NULL;
	float* d_right_p = NULL;
	cudaMalloc(&d_left_p, imgSize*2*sizeof(float));
	cudaMalloc(&d_right_p, imgSize*2*sizeof(float));

	// Gp
	float* d_left_Gp = NULL;
	float* d_right_Gp = NULL;
	cudaMalloc(&d_left_Gp, imgSize*2*sizeof(float));
	cudaMalloc(&d_right_Gp, imgSize*2*sizeof(float));

	// dual variable for v
	float* d_left_q = NULL;
	float* d_right_q = NULL;
	cudaMalloc(&d_left_q, imgSize*4*sizeof(float));
	cudaMalloc(&d_right_q, imgSize*4*sizeof(float));

	// augmented Lagrangian variable
	float* d_left_L = NULL;
	float* d_right_L = NULL;
	cudaMalloc(&d_left_L, imgSize*sizeof(float));
	cudaMalloc(&d_right_L, imgSize*sizeof(float));

	// reuse Ddisp as auxiliary variable a
	
	// parameters for AL_TGV
	float theta_inv = 0.0f;	// make (u-a)^2/(2*theta) zero
	float beta = 1e-2f;
	int nIterSmooth = 150;
	const float a = 3.0f;
	const float b = 0.8f;
	int PMS_init_iter = 3;
	float lambda_d = 1.0f;	//data	1.0f for middlebury; 0.4f for kitti
	float lambda_s = 0.01f;	//regularization 0.2f for middlebury; 1.0f for kitti
	float lambda_a = 8.0f*lambda_s;
	float tau_p, tau_q, tau_u, tau_v;
	tau_p = tau_q = 1.0f/2.0f;
	tau_u = 1.0f/4.0f;
	tau_v = 1.0f/8.0f;

//	tau_u = tau_p = 1.0f/sqrtf(12.0f);
//	tau_v = tau_q = 1.0f/sqrtf(8.0f);


	//NL2TGV
	const int support_radius_NL2TGV = 2;
	const int neighbor_size = 2*support_radius_NL2TGV*(support_radius_NL2TGV+1);
	bool TGV = (smoothness_prior == "NL2TGV") || (smoothness_prior == "2TGV") ? true : false;	//false NLTV or TV

	const float wci = 255.0f/10.0f;
	const float wpi = 1.0f/support_radius_NL2TGV;

	bool window_based = true;
	

	float* d_left_alpha1_NL2TGV = NULL;	// support weight or alpha1
	float* d_right_alpha1_NL2TGV = NULL;
	cudaMalloc(&d_left_alpha1_NL2TGV, neighbor_size*imgSize*sizeof(float));
	cudaMalloc(&d_right_alpha1_NL2TGV, neighbor_size*imgSize*sizeof(float));

	InitAlpha1<<<gridSize, blockSize>>>(d_left_alpha1_NL2TGV, lR_to, lG_to, lB_to, support_radius_NL2TGV, cols, rows, wci, wpi);
	InitAlpha1<<<gridSize, blockSize>>>(d_right_alpha1_NL2TGV, rR_to, rG_to, rB_to, support_radius_NL2TGV, cols, rows, wci, wpi);

	// test alpha1
/*	{
		float* alpha1 = new float[neighbor_size*imgSize*sizeof(float)];
		cudaMemcpy(alpha1, d_left_alpha1_NL2TGV, neighbor_size*imgSize*sizeof(float), cudaMemcpyDeviceToHost);

		MyAlpha1 my_alpha1;
		my_alpha1.alpha1 = alpha1;
		my_alpha1.rad = support_radius_NL2TGV;
		my_alpha1.cols = cols;
	
		cv::imshow("alpha1 test", leftImg);
		cv::setMouseCallback("alpha1 test", ShowAlpha1, (void*)&my_alpha1);
		cv::waitKey(0);
		return;
	}*/

	float* d_left_w_NL2TGV = d_left_v;		//local plane normal nx, ny	use d_v instead
	float* d_right_w_NL2TGV = d_right_v;

	float* d_left_p_NL2TGV = NULL;	//dual for u/disparity
	float* d_right_p_NL2TGV = NULL;
	cudaMalloc(&d_left_p_NL2TGV, neighbor_size*imgSize*sizeof(float));
	cudaMalloc(&d_right_p_NL2TGV, neighbor_size*imgSize*sizeof(float));

	float* d_left_q_NL2TGV = NULL; // dual for w
	float* d_right_q_NL2TGV = NULL; 
	cudaMalloc(&d_left_q_NL2TGV, 2*neighbor_size*imgSize*sizeof(float));
	cudaMalloc(&d_right_q_NL2TGV, 2*neighbor_size*imgSize*sizeof(float));

	// cudaMemset to zero do not set to 0.0f
	InitNL2TGV<<<gridSize, blockSize>>>(d_left_L, d_left_w_NL2TGV, d_left_p_NL2TGV, d_left_q_NL2TGV, support_radius_NL2TGV, cols, rows);
	InitNL2TGV<<<gridSize, blockSize>>>(d_left_L, d_right_w_NL2TGV, d_right_p_NL2TGV, d_right_q_NL2TGV, support_radius_NL2TGV, cols, rows);

//	StartTimer();
                              
	// allocate memory for states
        curandState_t* states;
	curandGenerator_t gen;

	if(window_based)
	{
		cudaMalloc(&states, imgSize*sizeof(curandState_t));
		// initialize random states
//		init<<<gridSize, blockSize>>>(1234, states, cols, rows);
		
		// host CURAND
		curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
		// set seed
		curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
		// random initial right and left disparity
		curandGenerateUniform(gen, dLDisp, imgSize);
		
		// set seed
		curandSetPseudoRandomGeneratorSeed(gen, 4321ULL);
		// random initial right and left disparity
		curandGenerateUniform(gen, dRDisp, imgSize);
	}


	//init dLDispV	
	cudaMemcpy(dLDispV, dLDisp, imgSize*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dRDispV, dRDisp, imgSize*sizeof(float), cudaMemcpyDeviceToDevice);

	Init2TGV<<<gridSize, blockSize>>>(cols, rows, d_left_L, d_left_p, d_left_q, d_left_v);
	Init2TGV<<<gridSize, blockSize>>>(cols, rows, d_right_L, d_right_p, d_right_q, d_right_v);

	if(smoothness_prior == "2TGV" || smoothness_prior == "TV")
	{
		anisotropicDiffusionTensorG<<<gridSize, blockSize>>>(cols, rows, d_left_gray_gauss, d_left_ls_mask, d_left_G, a, b);
		anisotropicDiffusionTensorG<<<gridSize, blockSize>>>(cols, rows, d_right_gray_gauss, d_right_ls_mask, d_right_G, a, b);
	}

#if 0	
        // random initial right and left plane
        init_plane_normals_weights_huber<<<gridSize, blockSize>>>(dRPlanes, d_right_ls_mask, dRWeight, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to, states, cols, rows);
        init_plane_normals_weights_huber<<<gridSize, blockSize>>>(dLPlanes, d_left_ls_mask, dLWeight, lGradX_to, lGradY_to, lGradXY_to, lGradYX_to, states, cols, rows);
#endif
	
//	cudaDeviceSynchronize();
//	std::cout<<"Init:"<<GetTimer()<<std::endl;
	
	//StartTimer();


//#define FILL_OCCLUSION
#define SHOW_INTERMEDIATE_RESULT

#define AL_TGV

	if(data_cost == "MCCNN_fst")
	{
		RemoveNanFromCostVolume<<<gridSize, blockSize>>>(d_left_cost_vol, d_left_ambiguity, Dmax, cols, rows, true);
		RemoveNanFromCostVolume<<<gridSize, blockSize>>>(d_right_cost_vol, d_right_ambiguity, Dmax, cols, rows, true);
	}
	else if(data_cost == "MCCNN_acrt")
	{
		RemoveNanFromCostVolume<<<gridSize, blockSize>>>(d_left_cost_vol, d_left_ambiguity, Dmax, cols, rows);
		RemoveNanFromCostVolume<<<gridSize, blockSize>>>(d_right_cost_vol, d_right_ambiguity, Dmax, cols, rows);
	}

	/*cudaDeviceSynchronize();

	cudaMemcpy(cvRightDisp_f.ptr<float>(0), d_right_ambiguity, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(cvLeftDisp_f.ptr<float>(0), d_left_ambiguity, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
	cv::imshow("Left ambiguity", cvLeftDisp_f);
	cv::imshow("Right ambiguity", cvRightDisp_f);
	cv::waitKey(1000);

*/

#define right

#if 0
	NonlocalTGVDenoise(cvLeftB_f, d_left_w_NL2TGV, d_left_p_NL2TGV, d_left_q_NL2TGV, d_left_alpha1_NL2TGV, d_left_L, 
				support_radius_NL2TGV, lambda_s, lambda_a, lambda_d, cols, rows, blockSize, gridSize);

	NonlocalTGVDenoise(cvLeftG_f, d_left_w_NL2TGV, d_left_p_NL2TGV, d_left_q_NL2TGV, d_left_alpha1_NL2TGV, d_left_L, 
				support_radius_NL2TGV, lambda_s, lambda_a, lambda_d, cols, rows, blockSize, gridSize);

	NonlocalTGVDenoise(cvLeftR_f, d_left_w_NL2TGV, d_left_p_NL2TGV, d_left_q_NL2TGV, d_left_alpha1_NL2TGV, d_left_L, 
				support_radius_NL2TGV, lambda_s, lambda_a, lambda_d, cols, rows, blockSize, gridSize);

	cv::imshow("denoiseLB", cvLeftB_f);
//	cv::imshow("denoiseLG", cvLeftG_f);
//	cv::imshow("denoiseLR", cvLeftR_f);

	NonlocalTGVDenoise(cvRightB_f, d_right_w_NL2TGV, d_right_p_NL2TGV, d_right_q_NL2TGV, d_right_alpha1_NL2TGV, d_right_L, 
				support_radius_NL2TGV, lambda_s, lambda_a, lambda_d, cols, rows, blockSize, gridSize);

	NonlocalTGVDenoise(cvRightG_f, d_right_w_NL2TGV, d_right_p_NL2TGV, d_right_q_NL2TGV, d_right_alpha1_NL2TGV, d_right_L, 
				support_radius_NL2TGV, lambda_s, lambda_a, lambda_d, cols, rows, blockSize, gridSize);

	NonlocalTGVDenoise(cvRightR_f, d_right_w_NL2TGV, d_right_p_NL2TGV, d_right_q_NL2TGV, d_right_alpha1_NL2TGV, d_right_L, 
				support_radius_NL2TGV, lambda_s, lambda_a, lambda_d, cols, rows, blockSize, gridSize);

	cv::imshow("denoiseRB", cvRightB_f);
//	cv::imshow("denoiseRG", cvRightG_f);
//	cv::imshow("denoiseRR", cvRightR_f);
	//cv::waitKey(0);

	cvLeftB_f *= 255.;
	cvLeftG_f *= 255.;
	cvLeftR_f *= 255.;
	cvRightB_f *= 255.;
	cvRightG_f *= 255.;
	cvRightR_f *= 255.;
#endif


	std::vector<mst_graph_t> left_mst_vec;
	std::vector<mst_graph_t> right_mst_vec;

	std::vector<std::vector<int>> left_mst_vertices_vec;
	std::vector<std::vector<int>> right_mst_vertices_vec;

	std::vector<std::vector<int>> left_bfs_order_vec;
	std::vector<std::vector<int>> right_bfs_order_vec;
	
	tree_graph_t left_tree_g;
	tree_graph_t right_tree_g;

	abc* left_abc_map = new abc[imgSize];
	abc* right_abc_map = new abc[imgSize];

	cv::Mat left_min_cost, right_min_cost;
	left_min_cost.create(rows, cols, CV_64F);
	right_min_cost.create(rows, cols, CV_64F);
	left_min_cost = std::numeric_limits<double>::max();
	right_min_cost = std::numeric_limits<double>::max();

	cv::Mat left_agg_cost, right_agg_cost;
	left_agg_cost.create(rows, cols, CV_64F);
	right_agg_cost.create(rows, cols, CV_64F);

	cv::Mat left_cost_norm_factor, right_cost_norm_factor;

#if 0
	std::vector<mst_graph_t> left_mst_vec5;
	std::vector<mst_graph_t> right_mst_vec5;

	std::vector<std::vector<int>> left_mst_vertices_vec5;
	std::vector<std::vector<int>> right_mst_vertices_vec5;

	std::vector<std::vector<int>> left_bfs_order_vec5;
	std::vector<std::vector<int>> right_bfs_order_vec5;
	
	tree_graph_t left_tree_g5;
	tree_graph_t right_tree_g5;

	abc* left_abc_map5 = new abc[imgSize];
	abc* right_abc_map5 = new abc[imgSize];

	cv::Mat left_min_cost5, right_min_cost5;
	left_min_cost5.create(rows, cols, CV_64F);
	right_min_cost5.create(rows, cols, CV_64F);
	left_min_cost5 = std::numeric_limits<double>::max();
	right_min_cost5 = std::numeric_limits<double>::max();

	cv::Mat left_cost_norm_factor5, right_cost_norm_factor5;

	cv::Mat leftDisp5; leftDisp5.create(rows, cols, CV_32F);
	cv::Mat rightDisp5; rightDisp5.create(rows, cols, CV_32F);
#endif

	cv::Mat left_L, right_L;
	left_L.create(rows, cols, CV_32F);
	right_L.create(rows, cols, CV_32F);
	left_L = 0.0f;
	right_L = 0.0f;

	cv::Mat left_disp_u, right_disp_u;
	left_disp_u.create(rows, cols, CV_32F);
	right_disp_u.create(rows, cols, CV_32F);
	const float gamma = 1.0f/50.f;
	const float c = 10000.0f;
	const int min_cc_size = 200;

	segment_image_other_init(cvLeftBGR_v[2], cvLeftBGR_v[1], cvLeftBGR_v[0], 
				 //cvLeftR_f, cvLeftG_f, cvLeftB_f, 
				 left_mst_vec, left_mst_vertices_vec, left_tree_g, 
				 left_abc_map, left_cost_norm_factor, left_bfs_order_vec, 
				 c, min_cc_size, Dmax, gamma); 

/*	segment_image_other_init(cvLeftBGR_v[2], cvLeftBGR_v[1], cvLeftBGR_v[0], 
				 //cvLeftR_f, cvLeftG_f, cvLeftB_f, 
				 left_mst_vec5, left_mst_vertices_vec5, left_tree_g5, 
				 left_abc_map5, left_cost_norm_factor5, left_bfs_order_vec5, 
				 c, min_cc_size, Dmax, gamma, 5); 
*/
#ifdef right
	segment_image_other_init(cvRightBGR_v[2], cvRightBGR_v[1], cvRightBGR_v[0],
				 //cvRightR_f, cvRightG_f, cvRightB_f, 
				 right_mst_vec, right_mst_vertices_vec, right_tree_g, 
				 right_abc_map, right_cost_norm_factor, right_bfs_order_vec,
				 c, min_cc_size, Dmax, gamma); 

/*	segment_image_other_init(cvRightBGR_v[2], cvRightBGR_v[1], cvRightBGR_v[0],
				 //cvRightR_f, cvRightG_f, cvRightB_f, 
				 right_mst_vec5, right_mst_vertices_vec5, right_tree_g5, 
				 right_abc_map5, right_cost_norm_factor5, right_bfs_order_vec5,
				 c, min_cc_size, Dmax, gamma); */
#endif

	//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	//std::default_random_engine generator(seed);
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

	const int num_iter = 100;

	for(int i=0; i<num_iter; i++)
	{
//		std::cout<<"iter: "<<i+1<<"\n";
	
		MST_PMS(left_mst_vec, left_mst_vertices_vec, left_bfs_order_vec, left_tree_g, left_abc_map, left_min_cost.ptr<double>(0), 
			left_agg_cost.ptr<double>(0), left_cost_norm_factor.ptr<double>(0), left_cost_vol_mccnn_w, left_disp_u.ptr<float>(0),
			left_L.ptr<float>(0), 0.0f, lambda_d, Dmax, cols, rows, imgSize, gamma, generator, distribution);

//		MST_PMS(left_mst_vec5, left_mst_vertices_vec5, left_bfs_order_vec5, left_tree_g5, left_abc_map5, left_min_cost5.ptr<double>(0), 
//			left_agg_cost.ptr<double>(0), left_cost_norm_factor5.ptr<double>(0), left_cost_vol_mccnn_w, left_disp_u.ptr<float>(0),
//			left_L.ptr<float>(0), 0.0f, lambda_d, Dmax, cols, rows, imgSize, gamma, generator, distribution, true);

		if(i%4 == 0 || i == num_iter-1)
		{
			LabelToDisp(left_abc_map, leftDisp, rows, cols, Dmax);	
			cv::imshow("left a", leftDisp); cv::waitKey(50);

//			LabelToDisp(left_abc_map5, leftDisp5, rows, cols, Dmax);	
//			cv::imshow("left5 a", leftDisp5); cv::waitKey(50);						
		}
	}

	
	

#ifdef right
	for(int i=0; i<num_iter; i++)
	{
	//	std::cout<<"iter: "<<i+1<<"\n";
	
		MST_PMS(right_mst_vec, right_mst_vertices_vec, right_bfs_order_vec, right_tree_g, right_abc_map, right_min_cost.ptr<double>(0), 
			right_agg_cost.ptr<double>(0), right_cost_norm_factor.ptr<double>(0), right_cost_vol_mccnn_w, right_disp_u.ptr<float>(0),
			left_L.ptr<float>(0), 0.0f, lambda_d, Dmax, cols, rows, imgSize, gamma, generator, distribution, true);

//		MST_PMS(right_mst_vec5, right_mst_vertices_vec5, right_bfs_order_vec5, right_tree_g5, right_abc_map5, right_min_cost5.ptr<double>(0), 
//			right_agg_cost.ptr<double>(0), right_cost_norm_factor5.ptr<double>(0), right_cost_vol_mccnn_w, right_disp_u.ptr<float>(0),
//			left_L.ptr<float>(0), 0.0f, lambda_d, Dmax, cols, rows, imgSize, gamma, generator, distribution, true);

		if(i%4 == 0 || i == num_iter-1)
		{
			LabelToDisp(right_abc_map, rightDisp, rows, cols, Dmax);	
			cv::imshow("right a", rightDisp); cv::waitKey(50);

//			LabelToDisp(right_abc_map5, rightDisp5, rows, cols, Dmax);	
//			cv::imshow("right5 a", rightDisp5); cv::waitKey(50);
		}
	}
#endif

#if 0
	cv::Mat diff_mask;
	diff_mask = left_min_cost.mul(1.0/left_min_cost5) <= 1.0;

	cv::imshow("diff", diff_mask); cv::waitKey(0);

	for(int i=0; i<imgSize; i++)
	{
		if(diff_mask.ptr<uchar>(0)[i] == 0)
		{
			left_abc_map[i].a = left_abc_map5[i].a;
			left_abc_map[i].b = left_abc_map5[i].b;
			left_abc_map[i].c = left_abc_map5[i].c;
		}
	}

	LabelToDisp(left_abc_map, leftDisp, rows, cols, Dmax);


	diff_mask = right_min_cost.mul(1.0/right_min_cost5) <= 1.0;

	cv::imshow("diff", diff_mask); cv::waitKey(0);

	for(int i=0; i<imgSize; i++)
	{
		if(diff_mask.ptr<uchar>(0)[i] == 0)
		{
			right_abc_map[i].a = right_abc_map5[i].a;
			right_abc_map[i].b = right_abc_map5[i].b;
			right_abc_map[i].c = right_abc_map5[i].c;
		}
	}

	LabelToDisp(right_abc_map, rightDisp, rows, cols, Dmax);
#endif


#if 0
	NonlocalTGVDenoise(leftDisp, d_left_w_NL2TGV, d_left_p_NL2TGV, d_left_q_NL2TGV, d_left_alpha1_NL2TGV, d_left_L, 
				support_radius_NL2TGV, lambda_s, lambda_a, lambda_d, cols, rows, blockSize, gridSize);

	NonlocalTGVDenoise(rightDisp, d_right_w_NL2TGV, d_right_p_NL2TGV, d_right_q_NL2TGV, d_right_alpha1_NL2TGV, d_right_L, 
				support_radius_NL2TGV, lambda_s, lambda_a, lambda_d, cols, rows, blockSize, gridSize);
#endif

#if 1
	leftDisp *= (Dmax-1.f);
	rightDisp *= (Dmax-1.f); 
	leftRightConsistencyCheck(leftDisp.ptr<float>(0), rightDisp.ptr<float>(0), cols, rows, Dmax, false);
	//return;
#endif

#if 0	// disable until free memory
	leftDisp.copyTo(left_disp_u); 
	rightDisp.copyTo(right_disp_u); 

	theta_inv = 1.0f;

	float* d_right_u = dRDispV;
	float* d_left_u = dLDispV;
	float* d_right_a = dRDisp;
	float* d_left_a = dLDisp;

	cudaMemcpy(d_left_a, leftDisp.ptr<float>(0), sizeof(float)*imgSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_right_a, rightDisp.ptr<float>(0), sizeof(float)*imgSize, cudaMemcpyHostToDevice);

	for(int iter=0; iter<10; iter++)
	{
		std::cout<<"iter: "<<iter+1<<"\n";

		cudaMemcpy(d_left_u, left_disp_u.ptr<float>(0), sizeof(float)*imgSize, cudaMemcpyHostToDevice);

		cudaMemcpy(d_right_u, right_disp_u.ptr<float>(0), sizeof(float)*imgSize, cudaMemcpyHostToDevice);


		for(int j=0; j<nIterSmooth; j++)
		{
			NL2TGV_dualUpdate<<<gridSize, blockSize>>>(d_left_u, d_left_w_NL2TGV, d_left_p_NL2TGV, d_left_q_NL2TGV, d_left_alpha1_NL2TGV, 
									support_radius_NL2TGV, lambda_s, lambda_a, cols, rows, TGV);


			NL2TGV_primalUpdate<<<gridSize, blockSize>>>(d_left_u, d_left_w_NL2TGV, d_left_p_NL2TGV, d_left_q_NL2TGV, d_left_a, d_left_L, 
									theta_inv, support_radius_NL2TGV, cols, rows, TGV);

#ifdef right
			NL2TGV_dualUpdate<<<gridSize, blockSize>>>(d_right_u, d_right_w_NL2TGV, d_right_p_NL2TGV, d_right_q_NL2TGV, d_right_alpha1_NL2TGV, 
									support_radius_NL2TGV, lambda_s, lambda_a, cols, rows, TGV);


			NL2TGV_primalUpdate<<<gridSize, blockSize>>>(d_right_u, d_right_w_NL2TGV, d_right_p_NL2TGV, d_right_q_NL2TGV, d_right_a, d_right_L, 
									theta_inv, support_radius_NL2TGV, cols, rows, TGV);
#endif

		}

		cudaMemcpy(left_disp_u.ptr<float>(0), d_left_u, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(left_L.ptr<float>(0), d_left_L, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);

		cv::imshow("left u", left_disp_u);
		cv::waitKey(50);

#ifdef right
		cudaMemcpy(right_disp_u.ptr<float>(0), d_right_u, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(right_L.ptr<float>(0), d_right_L, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
		cv::imshow("right u", right_disp_u);
		cv::waitKey(50);
#endif

		if(iter == 0)	
		{
			cv::Mat u_a;
			u_a = (left_disp_u - leftDisp);
			cv::multiply(u_a, u_a, u_a);
			left_min_cost = lambda_d*left_min_cost.mul(left_cost_norm_factor); 
			for(int p=0; p<imgSize; p++) left_min_cost.ptr<double>(0)[p] += 0.5*u_a.ptr<float>(0)[p];
		
#ifdef right
			u_a = (right_disp_u - rightDisp);
			cv::multiply(u_a, u_a, u_a);
			right_min_cost = lambda_d*right_min_cost.mul(right_cost_norm_factor);
			for(int p=0; p<imgSize; p++) right_min_cost.ptr<double>(0)[p] += 0.5*u_a.ptr<float>(0)[p];
#endif
		}

		for(int j=0; j<5; j++)
		{
			MST_PMS(left_mst_vec, left_mst_vertices_vec, left_bfs_order_vec, left_tree_g, left_abc_map, left_min_cost.ptr<double>(0), 
				left_agg_cost.ptr<double>(0), left_cost_norm_factor.ptr<double>(0), left_cost_vol_mccnn_w, left_disp_u.ptr<float>(0),
				left_L.ptr<float>(0), theta_inv, lambda_d, Dmax, cols, rows, imgSize, gamma, generator, distribution, true);

#ifdef right
			MST_PMS(right_mst_vec, right_mst_vertices_vec, right_bfs_order_vec, right_tree_g, right_abc_map, right_min_cost.ptr<double>(0), 
				right_agg_cost.ptr<double>(0), right_cost_norm_factor.ptr<double>(0), right_cost_vol_mccnn_w, right_disp_u.ptr<float>(0),
				right_L.ptr<float>(0), theta_inv, lambda_d, Dmax, cols, rows, imgSize, gamma, generator, distribution, true);
#endif 
		}

		LabelToDisp(left_abc_map, leftDisp, rows, cols, Dmax);

#ifdef right
		LabelToDisp(right_abc_map, rightDisp, rows, cols, Dmax);		
#endif


		cudaMemcpy(d_left_a, leftDisp.ptr<float>(0), sizeof(float)*imgSize, cudaMemcpyHostToDevice);

#ifdef right
		cudaMemcpy(d_right_a, rightDisp.ptr<float>(0), sizeof(float)*imgSize, cudaMemcpyHostToDevice);
#endif

		AL_TGV_augmentedLagranianUpdate<<<gridSize, blockSize>>>(cols, rows, d_left_L, d_left_u, d_left_a, theta_inv);

#ifdef right
		AL_TGV_augmentedLagranianUpdate<<<gridSize, blockSize>>>(cols, rows, d_right_L, d_right_u, d_right_a, theta_inv);
#endif
		

		cudaDeviceSynchronize();
		theta_inv = 1.0f/( (1.0f/theta_inv)*(1.0f-beta*iter) );

		if( 1.0f/theta_inv < 1e-5f ) break;

#ifdef SHOW_INTERMEDIATE_RESULT
		//if(i%10==0 || i==iteration-1)
		{
			cv::Mat diff;
			cv::absdiff(leftDisp, left_disp_u, diff);
			std::cout<<"	left disp diff: "<<cv::sum(diff)[0]<<"\n";
			cv::imshow("left a", leftDisp);
			cv::waitKey(50);

#ifdef right
			cv::absdiff(rightDisp, right_disp_u, diff);
			std::cout<<"	right disp diff: "<<cv::sum(diff)[0]<<"\n";
			cv::imshow("right a", rightDisp);
			cv::waitKey(50);
#endif


		}
#endif
	}

//	leftRightCheckHuber<<<gridSize, blockSize>>>(dRDispV, dLDispV, dRPlanesV, dLPlanesV, dROMask, dLOMask, cols, rows, 0.0f, (float)Dmax, 1.0f, 10.0f, true);
//	fillInOccludedHuber<<<gridSize, blockSize>>>(dLDispV, dLPlanesV, dLOMask, cols, rows, (float)Dmax);
	//fillInOccludedHuber<<<gridSize, blockSize>>>(dRDispV, dRPlanesV, dROMask, cols, rows, (float)Dmax);

	cudaMemcpy(leftDisp.ptr<float>(0), d_left_u, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(rightDisp.ptr<float>(0), d_right_u, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);

	leftDisp *= (Dmax-1.0f);
	rightDisp *= (Dmax-1.0f);
	leftRightConsistencyCheck(leftDisp.ptr<float>(0), rightDisp.ptr<float>(0), cols, rows, Dmax);

	goto FREE_RESOURCE;
	return;

#if 0
	if(window_based)
	{
		for(int i=0; i<PMS_init_iter; i++)
		{
		
			stereoMatching_AL_TGV<<<gridSize, blockSize>>>( d_left_ambiguity, d_right_ambiguity,
									d_left_cost_vol, d_right_cost_vol, dRDispV, dLDispV, dRPlanesV, dLPlanesV,
									dRDisp, dRPlanes, dLDisp, dLPlanes,
									dLCost, dRCost, d_right_L, d_left_L, theta, lambda_d,
									cols, rows, winRadius,
									states, (float)Dmax, lR_to, lG_to, lB_to,
									lGray_to, lGradX_to, lGradY_to, lGradXY_to,lGradYX_to,
									rR_to, rG_to, rB_to,
									rGray_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to,
									theta_sigma_d, theta_sigma_n, i);

	#if 1
			cudaDeviceSynchronize();

#if 0
			if(i>0)
			{
				leftRightCheckHuber<<<gridSize, blockSize>>>(dRDisp, dLDisp, dRPlanes, dLPlanes, dROMask, dLOMask, cols, rows, 0.0f, (float)Dmax, 0.5f, 5.0f, false);
				fillInOccludedHuber<<<gridSize, blockSize>>>(dLDisp, dLPlanes, dLOMask, cols, rows, (float)Dmax);
				fillInOccludedHuber<<<gridSize, blockSize>>>(dRDisp, dRPlanes, dROMask, cols, rows, (float)Dmax);		
			}
#endif

			//std::cout<<"initial PMS iteration: "<<i+1<<std::endl;

			cudaMemcpy(cvRightDisp_f.ptr<float>(0), dRDisp, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
			cudaMemcpy(cvLeftDisp_f.ptr<float>(0), dLDisp, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
			cv::imshow("Left a", cvLeftDisp_f);
			cv::imshow("Right a", cvRightDisp_f);
			cv::waitKey(1000);
	#endif


		}

	}
	else
	{
		MCCNN_ALTV_CostVolumeWTA<<<gridSize, blockSize>>>(d_left_cost_vol, dLDisp, d_left_L, dLDispV, 1.0f/theta_inv, lambda_d, Dmax, cols, rows);
		MCCNN_ALTV_CostVolumeWTA<<<gridSize, blockSize>>>(d_right_cost_vol, dRDisp, d_right_L, dRDispV, 1.0f/theta_inv, lambda_d, Dmax, cols, rows);

	#if 1
			cudaDeviceSynchronize();

			//std::cout<<"initial PMS iteration: "<<i+1<<std::endl;

			cudaMemcpy(cvRightDisp_f.ptr<float>(0), dRDisp, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
			cudaMemcpy(cvLeftDisp_f.ptr<float>(0), dLDisp, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
			cv::imshow("Left a", cvLeftDisp_f);
			cv::imshow("Right a", cvRightDisp_f);
			cv::waitKey(1000);
	#endif
	}

#endif

	theta_inv = 1.0f;



	cudaMemcpy(dRDispV, dRDisp, imgSize*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dLDispV, dLDisp, imgSize*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dRPlanesV, dRPlanes, imgSize*2*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(dLPlanesV, dLPlanes, imgSize*2*sizeof(float), cudaMemcpyDeviceToDevice);



	
	if(smoothness_prior != "NONE")
	for(int i=0; i<iteration; i++)
	{
		//StartTimer();

		if(smoothness_prior == "2TGV" || smoothness_prior == "TV")
		{
			for(int j=0; j<nIterSmooth; j++)
			{
				AL_TGV_dualUpdate<<<gridSize, blockSize>>>(cols, rows, d_right_G, d_right_p, d_right_u, d_right_v, d_right_q, tau_p, tau_q, lambda_s, lambda_a, TGV);	
				AL_TGV_dualUpdate<<<gridSize, blockSize>>>(cols, rows, d_left_G, d_left_p, d_left_u, d_left_v, d_left_q, tau_p, tau_q, lambda_s, lambda_a, TGV);

				AL_TGV_computeGp<<<gridSize, blockSize>>>(cols, rows, d_right_p, d_right_G, d_right_Gp);
				AL_TGV_computeGp<<<gridSize, blockSize>>>(cols, rows, d_left_p, d_left_G, d_left_Gp);

				AL_TGV_primalUpdate<<<gridSize, blockSize>>>(cols, rows, d_right_u, d_right_G, d_right_p, d_right_L, d_right_a, d_right_v, d_right_q, d_right_Gp, tau_u, tau_v, theta_inv, TGV);
				AL_TGV_primalUpdate<<<gridSize, blockSize>>>(cols, rows, d_left_u, d_left_G, d_left_p, d_left_L, d_left_a, d_left_v, d_left_q, d_left_Gp, tau_u, tau_v, theta_inv, TGV);

	#if 0
				//std::cout<<"	inner iteration: "<<j<<"\n";
				cudaDeviceSynchronize();
				cudaMemcpy(cvRightDisp_f.ptr<float>(0), d_right_u, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
				cudaMemcpy(cvLeftDisp_f.ptr<float>(0), d_left_u, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
				cv::imshow("Left u", cvLeftDisp_f);
				cv::imshow("Right u", cvRightDisp_f);
				cv::waitKey(1000);
	#endif
		
			}
		}
		else if(smoothness_prior == "NL2TGV" || smoothness_prior == "NLTV")
		{

			for(int j=0; j<nIterSmooth; j++)
			{
				NL2TGV_dualUpdate<<<gridSize, blockSize>>>(d_left_u, d_left_w_NL2TGV, d_left_p_NL2TGV, d_left_q_NL2TGV, d_left_alpha1_NL2TGV, 
										support_radius_NL2TGV, lambda_s, lambda_a, cols, rows, TGV);
				NL2TGV_dualUpdate<<<gridSize, blockSize>>>(d_right_u, d_right_w_NL2TGV, d_right_p_NL2TGV, d_right_q_NL2TGV, d_right_alpha1_NL2TGV, 
										support_radius_NL2TGV, lambda_s, lambda_a, cols, rows, TGV);

				NL2TGV_primalUpdate<<<gridSize, blockSize>>>(d_left_u, d_left_w_NL2TGV, d_left_p_NL2TGV, d_left_q_NL2TGV, d_left_a, d_left_L, 
										theta_inv, support_radius_NL2TGV, cols, rows, TGV);
				NL2TGV_primalUpdate<<<gridSize, blockSize>>>(d_right_u, d_right_w_NL2TGV, d_right_p_NL2TGV, d_right_q_NL2TGV, d_right_a, d_right_L, 
										theta_inv, support_radius_NL2TGV, cols, rows, TGV);

	#if 0
				std::cout<<"	inner iteration: "<<j<<"\n";
				cudaDeviceSynchronize();
				cudaMemcpy(cvRightDisp_f.ptr<float>(0), d_right_u, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
				cudaMemcpy(cvLeftDisp_f.ptr<float>(0), d_left_u, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
				cv::imshow("Left u", cvLeftDisp_f);
				cv::imshow("Right u", cvRightDisp_f);
				cv::waitKey(1000);
	#endif
			}
		}


		//cudaDeviceSynchronize();
		//std::cout<<GetTimer()<<std::endl;

#ifdef SHOW_INTERMEDIATE_RESULT
		//if(i%10==0 || i==iteration-1)
		{
			//std::cout<<"AL TGV iteration: "<<i<<"\n";
			cudaDeviceSynchronize();
			cudaMemcpy(cvRightDisp_f.ptr<float>(0), d_right_u, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
			cudaMemcpy(cvLeftDisp_f.ptr<float>(0), d_left_u, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
			cv::imshow("Left u", cvLeftDisp_f);
			cv::imshow("Right u", cvRightDisp_f);
			cv::waitKey(1000);

			//std::cout<<"PMS update\n";
		}
#endif

		//cudaDeviceSynchronize();
		//StartTimer();
		
		if (window_based)
		{
#if 0
			stereoMatching_AL_TGV<<<gridSize, blockSize>>>( d_left_ambiguity, d_right_ambiguity,
									d_left_cost_vol, d_right_cost_vol, dRDispV, dLDispV, dRPlanesV, dLPlanesV,
									dRDisp, dRPlanes, dLDisp, dLPlanes,
									dLCost, dRCost, d_right_L, d_left_L, 1.0f/theta_inv, lambda_d,
									cols, rows, winRadius,
									states, (float)Dmax, lR_to, lG_to, lB_to,
									lGray_to, lGradX_to, lGradY_to, lGradXY_to,lGradYX_to,
									rR_to, rG_to, rB_to,
									rGray_to, rGradX_to, rGradY_to, rGradXY_to, rGradYX_to,
									theta_sigma_d, theta_sigma_n, i, true, true);
#endif
		}
		else
		{
			MCCNN_ALTV_CostVolumeWTA<<<gridSize, blockSize>>>(d_left_cost_vol, d_left_a, d_left_L, d_left_u, 1.0f/theta_inv, lambda_d, Dmax, cols, rows);
			MCCNN_ALTV_CostVolumeWTA<<<gridSize, blockSize>>>(d_right_cost_vol, d_right_a, d_right_L, d_right_u, 1.0f/theta_inv, lambda_d, Dmax, cols, rows);
		}
		//cudaDeviceSynchronize();
		//std::cout<<"WTA: "<<GetTimer()<<std::endl;

#if 0
		if(i>0)
		{
			leftRightCheckHuber<<<gridSize, blockSize>>>(dRDisp, dLDisp, dRPlanes, dLPlanes, dROMask, dLOMask, cols, rows, 0.0f, (float)Dmax, 0.5f, 5.0f, false);
			fillInOccludedHuber<<<gridSize, blockSize>>>(dLDisp, dLPlanes, dLOMask, cols, rows, (float)Dmax);
			fillInOccludedHuber<<<gridSize, blockSize>>>(dRDisp, dRPlanes, dROMask, cols, rows, (float)Dmax);		
		}
#endif


		AL_TGV_augmentedLagranianUpdate<<<gridSize, blockSize>>>(cols, rows, d_right_L, d_right_u, d_right_a, theta_inv);
		AL_TGV_augmentedLagranianUpdate<<<gridSize, blockSize>>>(cols, rows, d_left_L, d_left_u, d_left_a, theta_inv);

		cudaDeviceSynchronize();
		theta_inv = 1.0f/( (1.0f/theta_inv)*(1.0f-beta*i) );

#ifdef SHOW_INTERMEDIATE_RESULT
		//if(i%10==0 || i==iteration-1)
		{
			cudaMemcpy(cvRightDispV_f.ptr<float>(0), d_right_a, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
			cudaMemcpy(cvLeftDispV_f.ptr<float>(0), d_left_a, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);

			cv::Mat diff;
			cv::absdiff(cvLeftDispV_f, cvLeftDisp_f, diff);
			std::cout<<"	left disp diff: "<<cv::sum(diff)[0]<<" ";
			cv::absdiff(cvRightDispV_f, cvRightDisp_f, diff);
			std::cout<<"right disp diff: "<<cv::sum(diff)[0]<<"\n";

			cv::imshow("Left a", cvLeftDispV_f);
			cv::imshow("Right a", cvRightDispV_f);
			cv::waitKey(1000);
		}
#endif
	}

	/********************************post processing********************************************/
	

#if 1
//	leftRightCheckHuber<<<gridSize, blockSize>>>(dRDisp, dLDisp, dRPlanes, dLPlanes, dROMask, dLOMask, cols, rows, 0.0f, (float)Dmax, 1.0f, 10.0f, true);
//	fillInOccludedHuber<<<gridSize, blockSize>>>(dLDisp, dLPlanes, dLOMask, cols, rows, (float)Dmax);
//	fillInOccludedHuber<<<gridSize, blockSize>>>(dRDisp, dRPlanes, dROMask, cols, rows, (float)Dmax);
	leftRightCheckHuber<<<gridSize, blockSize>>>(dRDispV, dLDispV, dRPlanesV, dLPlanesV, dROMask, dLOMask, cols, rows, 0.0f, (float)Dmax, 1.0f, 10.0f, true);
	fillInOccludedHuber<<<gridSize, blockSize>>>(dLDispV, dLPlanesV, dLOMask, cols, rows, (float)Dmax);
	fillInOccludedHuber<<<gridSize, blockSize>>>(dRDispV, dRPlanesV, dROMask, cols, rows, (float)Dmax);		
	//weightedMedianFilter<<<gridSize, blockSize>>>(dLDisp, dLOMask, dRDisp, dROMask, cols, rows, (float)Dmax, true, lR_to, lG_to, lB_to, lGray_to, rR_to, rG_to, rB_to, rGray_to);
	//weightedMedianFilter<<<gridSize, blockSize>>>(dLDispV, dLOMask, dRDispV, dROMask, cols, rows, (float)Dmax, true, lR_to, lG_to, lB_to, lGray_to, rR_to, rG_to, rB_to, rGray_to);
#endif


//	cudaDeviceSynchronize();
//	std::cout<<"Main loop:"<<GetTimer()<<std::endl;

#if 0	
        cudaMemcpy(dRDispV, dRDisp, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(dLDispV, dLDisp, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(dRPlanesV, dRPlanes, sizeof(float)*2*imgSize, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dLPlanesV, dLPlanes, sizeof(float)*2*imgSize, cudaMemcpyDeviceToDevice);
#endif


#if 0
	float* left_v = new float[imgSize*2];
	float* right_v = new float[imgSize*2];

        cudaMemcpy(left_v, d_left_v, sizeof(float)*imgSize*2, cudaMemcpyDeviceToHost);
        cudaMemcpy(right_v, d_right_v, sizeof(float)*imgSize*2, cudaMemcpyDeviceToHost);

	cv::Mat left_normal, right_normal;
	left_normal.create(rows, cols, CV_8UC3);
	right_normal.create(rows, cols, CV_8UC3);

	for(int i=0; i<imgSize; i++)
	{
		float nx = left_v[i*2];
		float ny = left_v[i*2+1];

		float norm = sqrt(nx*nx+ny*ny);

		nx /= norm;
		ny /= norm;

		float nz = sqrt(1.0f-nx*nx-ny*ny);

		left_normal.ptr<uchar>(0)[3*i] = (uchar)(nx*255.0f);
		left_normal.ptr<uchar>(0)[3*i+1] = (uchar)(ny*255.0f);
		left_normal.ptr<uchar>(0)[3*i+2] = (uchar)(nz*255.0f);
	}

	cv::imshow("left norm", left_normal);
	cv::waitKey(0);
#endif

	
        // copy disparity map from global memory on device to host
        cudaMemcpy(cvRightDisp_f.ptr<float>(0), dRDispV, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(cvLeftDisp_f.ptr<float>(0), dLDispV, sizeof(float)*imgSize, cudaMemcpyDeviceToHost);

	leftDisp = cvLeftDisp_f.clone();
	rightDisp = cvRightDisp_f.clone();

	leftDisp *= Dmax;
	rightDisp *= Dmax;

	if(data_cost == "MC_CNN_fst" || data_cost == "MC_CNN_acrt")
	{
		leftDisp += 1.0f;
		rightDisp += 1.0f;
	}

	if(showLeftDisp)
	{
		cv::imshow("Right Disp", cvRightDisp_f);
		cv::imshow("Left Disp", cvLeftDisp_f);
		cv::waitKey(0);
	}

#if USE_PCL
	float *normal_xy = new float[2*imgSize];

	cudaMemcpy(normal_xy, dRPlanesV, imgSize*2*sizeof(float), cudaMemcpyDeviceToHost);

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));

	struct callback_args cb_args;
	cb_args.cloud = cloud;
	cb_args.normals = normals;
        cb_args.viewerPtr = viewer;
	viewer->registerPointPickingCallback(pp_callback, (void*)&cb_args);

	float scale_along_z = 10.0f;
	float norm;

	for(int y=0; y<rows; y++)
	{
		for(int x=0; x<cols; x++)
		{
			int idx = y*cols+x;
			
			if(rightDisp.ptr<float>(0)[idx] < 1.f) continue;


			pcl::PointXYZRGB p;
			p.x = x; p.y = y;
			p.z = (Dmax - rightDisp.ptr<float>(0)[idx])*scale_along_z;
			//p.z = rightDisp.ptr<float>(0)[idx];
			cv::Vec3b bgr = rightImg.at<cv::Vec3b>(y, x);
			p.b = bgr.val[0];
			p.g = bgr.val[1];
			p.r = bgr.val[2];
			cloud->push_back(p);
			pcl::Normal n;
			n.normal_x = normal_xy[2*idx];
			n.normal_y = normal_xy[2*idx+1];
			n.normal_z = -sqrtf(1.0f-n.normal_x*n.normal_x-n.normal_y*n.normal_y)/scale_along_z;

			norm = sqrtf(n.normal_x*n.normal_x+n.normal_y*n.normal_y+n.normal_z*n.normal_z);
			n.normal_x /= norm;
			n.normal_y /= norm;
			n.normal_z /= norm;

			normals->push_back(n);
		}
	}


/*	viewer->addPointCloud(cloud,"cloud", 0);
	viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals, 10, 10.0f, "normals");
	viewer->spin();
	viewer->removeAllPointClouds(0);
	viewer->removeAllShapes(0);

	cloud->points.clear();
	normals->points.clear();
*/
	cudaMemcpy(normal_xy, dLPlanesV, imgSize*2*sizeof(float), cudaMemcpyDeviceToHost);


	for(int y=0; y<rows; y++)
	{
		for(int x=0; x<cols; x++)
		{
			int idx = y*cols+x;
			pcl::PointXYZRGB p;
			p.x = x-1.1f*cols; 
			p.y = y;
			p.z = (Dmax - leftDisp.ptr<float>(0)[idx])*scale_along_z;
			//p.z = leftDisp.ptr<float>(0)[idx];
			cv::Vec3b bgr = leftImg.at<cv::Vec3b>(y, x);
			p.b = bgr.val[0];
			p.g = bgr.val[1];
			p.r = bgr.val[2];
			cloud->push_back(p);
			pcl::Normal n;
			n.normal_x = normal_xy[2*idx];
			n.normal_y = normal_xy[2*idx+1];
			n.normal_z = -sqrtf(1.0f-n.normal_x*n.normal_x-n.normal_y*n.normal_y)/scale_along_z;

			norm = sqrtf(n.normal_x*n.normal_x+n.normal_y*n.normal_y+n.normal_z*n.normal_z);
			n.normal_x /= norm;
			n.normal_y /= norm;
			n.normal_z /= norm;

			normals->push_back(n);
		}

	}

	viewer->addPointCloud(cloud,"cloud", 0);
	viewer->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(cloud, normals, 5, 10.0f, "normals");
	viewer->spin();

	delete[] normal_xy;
#endif	

#endif

FREE_RESOURCE:
        // Free device memory

//	delete[] left_cost_vol_mccnn;
//	delete[] right_cost_vol_mccnn;
	delete[] left_cost_vol_mccnn_w;
	delete[] right_cost_vol_mccnn_w;
	cudaFree(dRDisp);
	cudaFree(dRPlanes);
	cudaFree(dLDisp);
	cudaFree(dLPlanes);
	cudaFree(states);

	cudaFree(dLCost);
	cudaFree(dRCost);
	cudaFree(lR_ca);
	cudaFree(lG_ca);
	cudaFree(lB_ca);
	cudaFree(lGray_ca);
	cudaFree(lGradX_ca);
	cudaFree(lGradY_ca);
	cudaFree(lGradXY_ca);
	cudaFree(lGradYX_ca);
	cudaFree(rR_ca);
	cudaFree(rG_ca);
	cudaFree(rB_ca);
	cudaFree(rGray_ca);
	cudaFree(rGradX_ca);
	cudaFree(rGradY_ca);
	cudaFree(rGradXY_ca);
	cudaFree(rGradYX_ca);
	cudaFree(dLPlanesV); 
	cudaFree(dRPlanesV);
	cudaFree(dLDispV);
	cudaFree(dRDispV);
	cudaFree(dLPlanesPn);
	cudaFree(dRPlanesPn);
	cudaFree(dLDispPd);
	cudaFree(dRDispPd);
	cudaFree(dLWeight);
	cudaFree(dRWeight);
	cudaFree(dRGradX);
	cudaFree(dLGradX);
	cudaFree(dLGradY);
	cudaFree(dRGradY);
	cudaFree(dLGradXY);
	cudaFree(dRGradXY);
	cudaFree(dLGradYX);
	cudaFree(dRGradYX);
	cudaFree(dLOMask);
	cudaFree(dROMask);
	cudaFree(d_left_ls_mask);
	cudaFree(d_right_ls_mask);
	cudaFree(d_left_cost_vol);
	cudaFree(d_right_cost_vol);
	cudaFree(d_left_w_NL2TGV);
	cudaFree(d_right_w_NL2TGV);
	cudaFree(d_left_p_NL2TGV);
	cudaFree(d_right_p_NL2TGV);
	cudaFree(d_left_q_NL2TGV);
	cudaFree(d_right_q_NL2TGV);
	cudaFree(d_left_alpha1_NL2TGV);
	cudaFree(d_right_alpha1_NL2TGV);

	curandDestroyGenerator(gen);
	cudaDestroyTextureObject(lR_to);
	cudaDestroyTextureObject(lG_to);
	cudaDestroyTextureObject(lB_to);
	cudaDestroyTextureObject(lGray_to);
	cudaDestroyTextureObject(lGradX_to);
	cudaDestroyTextureObject(rR_to);
	cudaDestroyTextureObject(rG_to);
	cudaDestroyTextureObject(rB_to);
	cudaDestroyTextureObject(rGray_to);
	cudaDestroyTextureObject(rGradX_to);

	cudaDeviceReset();
	std::cout<<"GPU resource freed\n";
}


void variationalDisparityDenoise(float* d_denoise, float* d_gray, float* d_disp, const int rows, const int cols, const int min_disp, const int max_disp,
				const float lambda = 1.0f, const int denoise_iter = 100)
{
	if(d_denoise == NULL) 
	{
		std::cout<<"d_denoise NULL\n";
		return;
	}

	const int img_size = rows*cols;
	const int img_size_in_byte = img_size*sizeof(float);
	float* d_weight = NULL;
	float* d_data_dual = NULL;
	float* d_regu_dual_x = NULL;
	float* d_regu_dual_y = NULL;


	cudaMalloc(&d_weight, img_size_in_byte);
	cudaMalloc(&d_data_dual, img_size_in_byte);
	cudaMalloc(&d_regu_dual_x, img_size_in_byte);
	cudaMalloc(&d_regu_dual_y, img_size_in_byte);

	cudaMemset(d_data_dual, 0, img_size_in_byte);
	cudaMemset(d_regu_dual_x, 0, img_size_in_byte);
	cudaMemset(d_regu_dual_y, 0, img_size_in_byte);

	cudaMemcpy(d_denoise, d_disp, img_size_in_byte, cudaMemcpyDeviceToDevice);


	dim3 block_size_de(16,16);
	dim3 grid_size_de((cols + block_size_de.x - 1)/block_size_de.x, (rows + block_size_de.y - 1)/block_size_de.y);
	

	const float alpha = 10.0f;
	const float beta = 1.0f;
	perPixelWeightPlusNormalizeImg<<<grid_size_de, block_size_de>>>(d_gray, d_weight, alpha, beta, rows, cols, d_disp, d_denoise, min_disp, max_disp);
	cudaDeviceSynchronize();

	cv::Mat h_weight_cv32f, h_denoise_cv32f, h_disp_cv32f;
	h_weight_cv32f.create(rows, cols, CV_32F);
	h_denoise_cv32f.create(rows, cols, CV_32F);
	h_disp_cv32f.create(rows, cols, CV_32F);

	cudaMemcpy(h_weight_cv32f.ptr<float>(0), d_weight, sizeof(float)*img_size, cudaMemcpyDeviceToHost);

	cudaMemcpy(h_disp_cv32f.ptr<float>(0), d_disp, sizeof(float)*img_size, cudaMemcpyDeviceToHost);

	cv::Mat weight_cv8u, denoise_cv8u;
	weight_cv8u.create(rows, cols, CV_8UC1);
	denoise_cv8u.create(rows, cols, CV_8UC1);

	h_weight_cv32f.convertTo(weight_cv8u, CV_8UC1, 255.0);
	cv::imshow("weight", weight_cv8u);
	//cv::waitKey(0);
	
	StartTimer();
	const float delta = 0.00159f;	//regu huber
	const float gamma = 0.00159f;	//data huber
	const float sigma = 1.0f/(8.0f*0.02f);	// sigma > 1, tau < 1 
	const float tau = 0.02f;

//	const float sigma = 1.0f/sqrtf(12.0f);	// Fast and Accurate Large-scale Stereo Reconstruction using Variational Methods
//	const float tau = sigma;

	for(int iter=0; iter<denoise_iter; iter++)
	{
		weightedHuberDenoiseDualUpdate<<<grid_size_de, block_size_de>>>(d_disp, d_denoise/*primal*/, d_weight, d_data_dual, 
										d_regu_dual_x, d_regu_dual_y, rows, cols, lambda, delta, gamma, sigma);

		weightedHuberDenoisePrimalUpdate<<<grid_size_de, block_size_de>>>(d_disp, d_denoise/*primal*/, d_weight, d_data_dual, d_regu_dual_x, d_regu_dual_y, 
											rows, cols, lambda, tau);

		if(0 && iter%100 == 0)
		{
			cudaMemcpy(h_denoise_cv32f.ptr<float>(0), d_denoise, sizeof(float)*img_size, cudaMemcpyDeviceToHost);

			//compute energy
			float data_cost = 0.f;
			for(int i=0; i<img_size; i++)
			{
				float tmp = fabs(h_denoise_cv32f.ptr<float>(0)[i]-h_disp_cv32f.ptr<float>(0)[i]);
				data_cost += lambda*(tmp<=gamma ? tmp*tmp*0.5f/gamma : tmp-gamma*0.5f);	//huber
				//data_cost += 0.5f*lambda*tmp*tmp;	// ROF
				//data_cost += lambda*tmp;	//L1
			}

			float regu_cost = 0.f;
			for(int y=0; y<rows; y++)
			{
				for(int x=0; x<cols; x++)
				{
					int idx = y*cols + x;

					float dx = x==cols-1 ? 0.f : h_denoise_cv32f.ptr<float>(0)[idx+1] - h_denoise_cv32f.ptr<float>(0)[idx];

					float dy = y==rows-1 ? 0.f : h_denoise_cv32f.ptr<float>(0)[idx+cols] - h_denoise_cv32f.ptr<float>(0)[idx];

					float d = sqrt(dx*dx+dy*dy);
					
					regu_cost += d<=delta ? d*d*0.5f/delta : d-delta*0.5f;	//huber
					//regu_cost += d;	//TV
					
				}
			}

			std::cout<<"iter: "<<iter<<" cost: "<<data_cost+regu_cost<<" data: "<<data_cost<<" regu:"<<regu_cost<<"\n";
			
			h_denoise_cv32f.convertTo(denoise_cv8u, CV_8UC1, 4.0*max_disp);
			//h_denoise_cv32f.convertTo(denoise_cv8u, CV_8UC1, 4.0);	

			cv::Mat color_map;

			cv::applyColorMap(denoise_cv8u, color_map, cv::COLORMAP_JET);

			cv::imshow("denoise", color_map);
			cv::waitKey(0);
		}
	}

	
	//scale disparity back
	scaleDisparityBack<<<grid_size_de, block_size_de>>>(d_disp, min_disp, max_disp, rows, cols);
	cudaDeviceSynchronize();
	scaleDisparityBack<<<grid_size_de, block_size_de>>>(d_denoise, min_disp, max_disp, rows, cols);
	cudaDeviceSynchronize();
	std::cout<<"denoise:"<<GetTimer()<<std::endl;

	cudaFree(d_weight);	
	cudaFree(d_data_dual);
	cudaFree(d_regu_dual_x);
	cudaFree(d_regu_dual_y);
}


//d_a = (d_corr_gi - d_mean_guide*d_mean_input)/(d_var_g+eps)
//d_b = d_mean_input - d_a*d_mean_guide
__global__ void guidedFilterComputation1(float* d_a, float* d_corr_gi, float* d_mean_guide, float* d_mean_input, float* d_var_g, float* d_b, 
					const int rows, const int cols, const float eps)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x >= cols || y>= rows)
		return;		

	const int idx = y*cols+x;

	float mean_guide = d_mean_guide[idx];
	float mean_input = d_mean_input[idx];
	float a = (d_corr_gi[idx] - mean_guide*mean_input)/(d_var_g[idx] + eps);

	d_b[idx] = mean_input-a*mean_guide;
	d_a[idx] = a;
}

// d_input = d_a*d_guide-d_b
__global__ void guidedFilterComputation2(float* d_input, float* d_a, float* d_b, float* d_guide, const int rows, const int cols)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x >= cols || y>= rows) return;		

	const int idx = y*cols+x;

	d_input[idx] = d_a[idx]*d_guide[idx]+d_b[idx];
}

void costVolumeGuidedFilterCUDA(const int cols, const int rows, const int img_size_pad_rows, const int num_disp, const int win_rad_bf, const float eps)
{
	float *d_guide, *d_tmp, *d_mean_guide, *d_corr_g, *d_var_g;
	float *d_input, *d_mean_input, *d_corr_gi, *d_a, *d_b;

	d_guide = d_guide_global;
	d_tmp = d_tmp_global;
	d_mean_guide = d_mean_guide_global;
	d_corr_g = d_corr_g_global;
	d_var_g = d_var_g_global;
	d_input = d_input_global;
	d_mean_input = d_mean_input_global;
	d_corr_gi = d_corr_gi_global;
	d_a = d_a_global;
	d_b = d_b_global;

	dim3 block_size_mul(32, 8);
	dim3 grid_size_mul((cols + block_size_mul.x - 1)/block_size_mul.x, (rows + block_size_mul.y - 1)/block_size_mul.y); 
	dim3 grid_size( (cols+num_threads-1)/num_threads, (rows+num_threads-1)/num_threads);


	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float)>>>(d_guide, d_tmp, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads>>>(d_tmp, d_mean_guide, rows, cols, win_rad_bf);

	pointWiseMul_global<<< grid_size_mul, block_size_mul>>>(d_guide, d_guide, d_corr_g, rows, cols);
	
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float)>>>(d_corr_g, d_tmp, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads>>>(d_tmp, d_corr_g, rows, cols, win_rad_bf);

	pointWiseMul_global<<< grid_size_mul, block_size_mul>>>(d_mean_guide, d_mean_guide, d_var_g, rows, cols);

	pointWiseSub_global<<<grid_size_mul, block_size_mul>>>(d_corr_g, d_var_g, d_var_g, rows, cols);


/*	cudaStream_t streams[num_disp];
	for(int i=0; i<num_disp; i++) cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);

	for(int i=0; i<num_disp; i++)
	//{
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), streams[i]>>>(d_input+i*img_size_pad_rows, d_tmp+i*img_size_pad_rows, rows, cols, win_rad_bf);

	for(int i=0; i<num_disp; i++)
	boxFilter_y_global<<<grid_size, num_threads, 0, streams[i]>>>(d_tmp+i*img_size_pad_rows, d_mean_input+i*img_size_pad_rows, rows, cols, win_rad_bf);

	for(int i=0; i<num_disp; i++)
	pointWiseMul_global<<< grid_size_mul, block_size_mul, 0, streams[i]>>>(d_guide, d_input+i*img_size_pad_rows, d_corr_gi+i*img_size_pad_rows, rows, cols);

	for(int i=0; i<num_disp; i++)
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), streams[i]>>>(d_corr_gi+i*img_size_pad_rows, d_tmp+i*img_size_pad_rows, rows, cols, win_rad_bf);

	for(int i=0; i<num_disp; i++)
	boxFilter_y_global<<<grid_size, num_threads, 0, streams[i]>>>(d_tmp+i*img_size_pad_rows, d_corr_gi+i*img_size_pad_rows, rows, cols, win_rad_bf);

	//d_a = (d_corr_gi - d_mean_guide*d_mean_input)/(d_var_g+eps)
	//d_b = d_mean_input - d_a*d_mean_guide
	for(int i=0; i<num_disp; i++)
	guidedFilterComputation1<<<grid_size_mul, block_size_mul, 0, streams[i]>>>(d_a+i*img_size_pad_rows, d_corr_gi+i*img_size_pad_rows, d_mean_guide, d_mean_input+i*img_size_pad_rows, 
											d_var_g, d_b+i*img_size_pad_rows, rows, cols, eps);

	for(int i=0; i<num_disp; i++)
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), streams[i]>>>(d_a+i*img_size_pad_rows, d_tmp+i*img_size_pad_rows, rows, cols, win_rad_bf);
	for(int i=0; i<num_disp; i++)
	boxFilter_y_global<<<grid_size, num_threads, 0, streams[i]>>>(d_tmp+i*img_size_pad_rows, d_a+i*img_size_pad_rows, rows, cols, win_rad_bf);

	for(int i=0; i<num_disp; i++)
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), streams[i]>>>(d_b+i*img_size_pad_rows, d_tmp+i*img_size_pad_rows, rows, cols, win_rad_bf);
	for(int i=0; i<num_disp; i++)
	boxFilter_y_global<<<grid_size, num_threads, 0, streams[i]>>>(d_tmp+i*img_size_pad_rows, d_b+i*img_size_pad_rows, rows, cols, win_rad_bf);

	// d_input = d_a*d_guide-d_b
	for(int i=0; i<num_disp; i++)
	guidedFilterComputation2<<< grid_size_mul, block_size_mul, 0, streams[i]>>>(d_input+i*img_size_pad_rows, d_a+i*img_size_pad_rows, d_b+i*img_size_pad_rows, d_guide, rows, cols);
//	}

	for(int i=0; i<num_disp; i++) cudaStreamDestroy(streams[i]);
*/

	for(int d=0; d<num_disp; d++, d_input += img_size_pad_rows)
	{
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float)>>>(d_input, d_tmp, rows, cols, win_rad_bf);
		//boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float)>>>(d_tmp, d_input, rows, cols, win_rad_bf);	continue;	//boxfilter

		boxFilter_y_global<<<grid_size, num_threads>>>(d_tmp, d_mean_input, rows, cols, win_rad_bf);

		pointWiseMul_global<<< grid_size_mul, block_size_mul>>>(d_guide, d_input, d_corr_gi, rows, cols);

		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float)>>>(d_corr_gi, d_tmp, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads>>>(d_tmp, d_corr_gi, rows, cols, win_rad_bf);

		
		//d_a = (d_corr_gi - d_mean_guide*d_mean_input)/(d_var_g+eps)
		//d_b = d_mean_input - d_a*d_mean_guide
		guidedFilterComputation1<<<grid_size_mul, block_size_mul>>>(d_a, d_corr_gi, d_mean_guide, d_mean_input, d_var_g, d_b, rows, cols, eps);

		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float)>>>(d_a, d_tmp, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads>>>(d_tmp, d_a, rows, cols, win_rad_bf);

		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float)>>>(d_b, d_tmp, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads>>>(d_tmp, d_b, rows, cols, win_rad_bf);

		// d_input = d_a*d_guide-d_b
		guidedFilterComputation2<<< grid_size_mul, block_size_mul>>>(d_input, d_a, d_b, d_guide, rows, cols);
	}
}



void costVolumeGuidedFilterCUDA2Streams(const int cols, const int rows, const int img_size_pad_rows, const int num_disp, const int win_rad_bf, const float eps)
{
	cudaStream_t stream1, stream2;
	cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

	dim3 block_size_mul(32, 8);
	dim3 grid_size_mul((cols + block_size_mul.x - 1)/block_size_mul.x, (rows + block_size_mul.y - 1)/block_size_mul.y); 
	dim3 grid_size((cols+num_threads-1)/num_threads, (rows+num_threads-1)/num_threads);


	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_right_gray, d_tmp_global, rows, cols, win_rad_bf);
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_left_gray, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);

	boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_mean_guide_global, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_mean_guide_global+img_size_pad_rows, rows, cols, win_rad_bf);

	pointWiseMul_global<<< grid_size_mul, block_size_mul, 0, stream1>>>(d_right_gray, d_right_gray, d_corr_g_global, rows, cols);
	pointWiseMul_global<<< grid_size_mul, block_size_mul, 0, stream2>>>(d_left_gray, d_left_gray, d_corr_g_global+img_size_pad_rows, rows, cols);
	
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_corr_g_global, d_tmp_global, rows, cols, win_rad_bf);
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_corr_g_global+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);

	boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_corr_g_global, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_corr_g_global+img_size_pad_rows, rows, cols, win_rad_bf);

	pointWiseMul_global<<< grid_size_mul, block_size_mul, 0, stream1>>>(d_mean_guide_global, d_mean_guide_global, d_var_g_global, rows, cols);
	pointWiseMul_global<<< grid_size_mul, block_size_mul, 0, stream2>>>(d_mean_guide_global+img_size_pad_rows, d_mean_guide_global+img_size_pad_rows, d_var_g_global+img_size_pad_rows, rows, cols);

	pointWiseSub_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_corr_g_global, d_var_g_global, d_var_g_global, rows, cols);
	pointWiseSub_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_corr_g_global+img_size_pad_rows, d_var_g_global+img_size_pad_rows, d_var_g_global+img_size_pad_rows, rows, cols);

	float* d_right_input = d_right_cost_vol;
	float* d_left_input = d_left_cost_vol;
	for(int d=0; d<num_disp; d++, d_right_input+=img_size_pad_rows, d_left_input+=img_size_pad_rows)
	{
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_right_input, d_tmp_global, rows, cols, win_rad_bf);
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_left_input, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);

		boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_mean_input_global, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_mean_input_global+img_size_pad_rows, rows, cols, win_rad_bf);

		pointWiseMul_global<<< grid_size_mul, block_size_mul, 0, stream1>>>(d_right_gray, d_right_input, d_corr_gi_global, rows, cols);
		pointWiseMul_global<<< grid_size_mul, block_size_mul, 0, stream2>>>(d_left_gray, d_left_input, d_corr_gi_global+img_size_pad_rows, rows, cols);

		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_corr_gi_global, d_tmp_global, rows, cols, win_rad_bf);
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_corr_gi_global+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);

		boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_corr_gi_global, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_corr_gi_global+img_size_pad_rows, rows, cols, win_rad_bf);

		
		//d_a = (d_corr_gi - d_mean_guide*d_mean_input)/(d_var_g+eps)
		//d_b = d_mean_input - d_a*d_mean_guide
		guidedFilterComputation1<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_a_global, d_corr_gi_global, d_mean_guide_global, d_mean_input_global, d_var_g_global, d_b_global, rows, cols, eps);
		guidedFilterComputation1<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_a_global+img_size_pad_rows, d_corr_gi_global+img_size_pad_rows, d_mean_guide_global+img_size_pad_rows, d_mean_input_global+img_size_pad_rows, d_var_g_global+img_size_pad_rows, d_b_global+img_size_pad_rows, rows, cols, eps);

		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_a_global, d_tmp_global, rows, cols, win_rad_bf);
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_a_global+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);

		boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_a_global, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_a_global+img_size_pad_rows, rows, cols, win_rad_bf);

		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_b_global, d_tmp_global, rows, cols, win_rad_bf);
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_b_global+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);

		boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_b_global, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_b_global+img_size_pad_rows, rows, cols, win_rad_bf);

		// d_input = d_a*d_guide-d_b
		guidedFilterComputation2<<< grid_size_mul, block_size_mul, 0, stream1>>>(d_right_input, d_a_global, d_b_global, d_right_gray, rows, cols);
		guidedFilterComputation2<<< grid_size_mul, block_size_mul, 0, stream2>>>(d_left_input, d_a_global+img_size_pad_rows, d_b_global+img_size_pad_rows, d_left_gray, rows, cols);
	}

	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
}

__global__ void pointWiseAddScalar_global(float* d_src, float* d_dst, float scalar, const int cols, const int rows)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if(x>=cols || y>=rows) return;
	
	const int idx = y*cols + x;
	d_dst[idx] = d_src[idx] + scalar;
}


__global__ void colorGuidedFilterHelper0_global(float* d_dst, float* d_src_1, float* d_src_2, float eps, bool add_eps, const int cols, const int rows)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if(x>=cols || y>=rows) return;
	
	const int idx = y*cols + x;
	if(add_eps) d_dst[idx] = d_src_1[idx] - d_src_2[idx] + eps;
	else d_dst[idx] = d_src_1[idx] - d_src_2[idx];
}

//var_I_gg.mul(var_I_bb) - var_I_gb.mul(var_I_gb);
__global__ void colorGuidedFilterHelper1_global(float* d_dst, float* d_src_1, float* d_src_2, float* d_src_3, float* d_src_4, const int cols, const int rows)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if(x>=cols || y>=rows) return;
	
	const int idx = y*cols + x;
	d_dst[idx] = d_src_1[idx]*d_src_2[idx] - d_src_3[idx]*d_src_4[idx];
}

// cv::Mat covDet = invrr.mul(var_I_rr) + invrg.mul(var_I_rg) + invrb.mul(var_I_rb);
__global__ void colorGuidedFilterHelper2_global(float* d_dst, float* d_src_1, float* d_src_2, float* d_src_3, float* d_src_4, float* d_src_5, float* d_src_6, const int cols, const int rows)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if(x>=cols || y>=rows) return;
	
	const int idx = y*cols + x;
	d_dst[idx] = d_src_1[idx]*d_src_2[idx] + d_src_3[idx]*d_src_4[idx] + d_src_5[idx]*d_src_6[idx];
}


__global__ void pointWiseDivison_global(float* d_dst, float* d_src_1, float* d_src_2, const int cols, const int rows)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if(x>=cols || y>=rows) return;
	
	const int idx = y*cols + x;
	d_dst[idx] = d_src_1[idx]/d_src_2[idx];
}

// cv::Mat cov_Ip_r = mean_Ip_r - mean_I_r.mul(mean_p);
__global__ void colorGuidedFilterHelper3_global(float* d_dst, float* d_src_1, float* d_src_2, float* d_src_3, const int cols, const int rows)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if(x>=cols || y>=rows) return;
	
	const int idx = y*cols + x;
	d_dst[idx] = d_src_1[idx]-d_src_2[idx]*d_src_3[idx];
}

//cv::Mat b = mean_p - a_r.mul(mean_I_r) - a_g.mul(mean_I_g) - a_b.mul(mean_I_b);
__global__ void colorGuidedFilterHelper4_global(float* d_dst, float* d_src_1, float* d_src_2, float* d_src_3, float* d_src_4, float* d_src_5, float* d_src_6, float* d_src_7, const int cols, const int rows)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if(x>=cols || y>=rows) return;
	
	const int idx = y*cols + x;
	d_dst[idx] = d_src_1[idx] - d_src_2[idx]*d_src_3[idx] - d_src_4[idx]*d_src_5[idx] - d_src_6[idx]*d_src_7[idx];
}


__global__ void colorGuidedFilterHelper5_global(float* d_dst, float* d_src_1, float* d_src_2, float* d_src_3, float* d_src_4, float* d_src_5, float* d_src_6, float* d_src_7, const int cols, const int rows)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if(x>=cols || y>=rows) return;
	
	const int idx = y*cols + x;
	d_dst[idx] = d_src_1[idx] + d_src_2[idx]*d_src_3[idx] + d_src_4[idx]*d_src_5[idx] + d_src_6[idx]*d_src_7[idx];
}


void costVolumeColorGuidedFilterCUDA2Streams(const int cols, const int rows, const int img_size_pad_rows, const int num_disp, const int win_rad_bf, const float eps)
{
	cudaStream_t stream1, stream2;
	cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

	dim3 block_size_mul(32, 8);
	dim3 grid_size_mul((cols + block_size_mul.x - 1)/block_size_mul.x, (rows + block_size_mul.y - 1)/block_size_mul.y); 
	dim3 grid_size((cols+num_threads-1)/num_threads, (rows+num_threads-1)/num_threads);

	//mean_r, mean_g, mean_b
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_r, d_tmp_global, rows, cols, win_rad_bf);
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_r+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_mean_r, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_mean_r+img_size_pad_rows, rows, cols, win_rad_bf);
	
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_g, d_tmp_global, rows, cols, win_rad_bf);
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_g+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_mean_g, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_mean_g+img_size_pad_rows, rows, cols, win_rad_bf);

	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_b, d_tmp_global, rows, cols, win_rad_bf);
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_b+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_mean_b, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_mean_b+img_size_pad_rows, rows, cols, win_rad_bf);


	//variance rr = boxfilter(r.*r) -mean_r.*mean_r + eps
	// use d_corr_gi_global as temporary container
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_r, d_r, d_corr_gi_global, rows, cols);
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_r+img_size_pad_rows, d_r+img_size_pad_rows, d_corr_gi_global+img_size_pad_rows, rows, cols);

	// use d_mean_input_global as temporary container
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_mean_r, d_mean_r, d_mean_input_global, rows, cols);
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_mean_r+img_size_pad_rows, d_mean_r+img_size_pad_rows, d_mean_input_global+img_size_pad_rows, rows, cols);

	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_corr_gi_global, d_tmp_global, rows, cols, win_rad_bf);
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_corr_gi_global+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_var_rr, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_var_rr+img_size_pad_rows, rows, cols, win_rad_bf);

	colorGuidedFilterHelper0_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_var_rr, d_var_rr, d_mean_input_global, eps, true, cols, rows);
	colorGuidedFilterHelper0_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_var_rr+img_size_pad_rows, d_var_rr+img_size_pad_rows, d_mean_input_global+img_size_pad_rows, eps, true, cols, rows);

	//variance gg
	// use d_corr_gi_global as temporary container
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_g, d_g, d_corr_gi_global, rows, cols);
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_g+img_size_pad_rows, d_g+img_size_pad_rows, d_corr_gi_global+img_size_pad_rows, rows, cols);

	// use d_mean_input_global as temporary container
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_mean_g, d_mean_g, d_mean_input_global, rows, cols);
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_mean_g+img_size_pad_rows, d_mean_g+img_size_pad_rows, d_mean_input_global+img_size_pad_rows, rows, cols);

	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_corr_gi_global, d_tmp_global, rows, cols, win_rad_bf);
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_corr_gi_global+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_var_gg, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_var_gg+img_size_pad_rows, rows, cols, win_rad_bf);

	colorGuidedFilterHelper0_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_var_gg, d_var_gg, d_mean_input_global, eps, true, cols, rows);
	colorGuidedFilterHelper0_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_var_gg+img_size_pad_rows, d_var_gg+img_size_pad_rows, d_mean_input_global+img_size_pad_rows, eps, true, cols, rows);

	//variance bb
	// use d_corr_gi_global as temporary container
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_b, d_b, d_corr_gi_global, rows, cols);
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_b+img_size_pad_rows, d_b+img_size_pad_rows, d_corr_gi_global+img_size_pad_rows, rows, cols);

	// use d_mean_input_global as temporary container
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_mean_b, d_mean_b, d_mean_input_global, rows, cols);
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_mean_b+img_size_pad_rows, d_mean_b+img_size_pad_rows, d_mean_input_global+img_size_pad_rows, rows, cols);

	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_corr_gi_global, d_tmp_global, rows, cols, win_rad_bf);
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_corr_gi_global+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_var_bb, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_var_bb+img_size_pad_rows, rows, cols, win_rad_bf);

	colorGuidedFilterHelper0_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_var_bb, d_var_bb, d_mean_input_global, eps, true, cols, rows);
	colorGuidedFilterHelper0_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_var_bb+img_size_pad_rows, d_var_bb+img_size_pad_rows, d_mean_input_global+img_size_pad_rows, eps, true, cols, rows);


	//variance rg
	// use d_corr_gi_global as temporary container
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_r, d_g, d_corr_gi_global, rows, cols);
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_r+img_size_pad_rows, d_g+img_size_pad_rows, d_corr_gi_global+img_size_pad_rows, rows, cols);

	// use d_mean_input_global as temporary container
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_mean_r, d_mean_g, d_mean_input_global, rows, cols);
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_mean_r+img_size_pad_rows, d_mean_g+img_size_pad_rows, d_mean_input_global+img_size_pad_rows, rows, cols);

	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_corr_gi_global, d_tmp_global, rows, cols, win_rad_bf);
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_corr_gi_global+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_var_rg, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_var_rg+img_size_pad_rows, rows, cols, win_rad_bf);

	colorGuidedFilterHelper0_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_var_rg, d_var_rg, d_mean_input_global, eps, false, cols, rows);
	colorGuidedFilterHelper0_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_var_rg+img_size_pad_rows, d_var_rg+img_size_pad_rows, d_mean_input_global+img_size_pad_rows, eps, false, cols, rows);

	//variance rb
	// use d_corr_gi_global as temporary container
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_r, d_b, d_corr_gi_global, rows, cols);
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_r+img_size_pad_rows, d_b+img_size_pad_rows, d_corr_gi_global+img_size_pad_rows, rows, cols);

	// use d_mean_input_global as temporary container
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_mean_r, d_mean_b, d_mean_input_global, rows, cols);
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_mean_r+img_size_pad_rows, d_mean_b+img_size_pad_rows, d_mean_input_global+img_size_pad_rows, rows, cols);

	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_corr_gi_global, d_tmp_global, rows, cols, win_rad_bf);
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_corr_gi_global+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_var_rb, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_var_rb+img_size_pad_rows, rows, cols, win_rad_bf);

	colorGuidedFilterHelper0_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_var_rb, d_var_rb, d_mean_input_global, eps, false, cols, rows);
	colorGuidedFilterHelper0_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_var_rb+img_size_pad_rows, d_var_rb+img_size_pad_rows, d_mean_input_global+img_size_pad_rows, eps, false, cols, rows);

	//variance gb
	// use d_corr_gi_global as temporary container
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_g, d_b, d_corr_gi_global, rows, cols);
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_g+img_size_pad_rows, d_b+img_size_pad_rows, d_corr_gi_global+img_size_pad_rows, rows, cols);

	// use d_mean_input_global as temporary container
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_mean_g, d_mean_b, d_mean_input_global, rows, cols);
	pointWiseMul_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_mean_g+img_size_pad_rows, d_mean_b+img_size_pad_rows, d_mean_input_global+img_size_pad_rows, rows, cols);

	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_corr_gi_global, d_tmp_global, rows, cols, win_rad_bf);
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_corr_gi_global+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_var_gb, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_var_gb+img_size_pad_rows, rows, cols, win_rad_bf);

	colorGuidedFilterHelper0_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_var_gb, d_var_gb, d_mean_input_global, eps, false, cols, rows);
	colorGuidedFilterHelper0_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_var_gb+img_size_pad_rows, d_var_gb+img_size_pad_rows, d_mean_input_global+img_size_pad_rows, eps, false, cols, rows);


	//inverse rr
	colorGuidedFilterHelper1_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_inv_rr, d_var_gg, d_var_bb, d_var_gb, d_var_gb, cols, rows);
	colorGuidedFilterHelper1_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_inv_rr+img_size_pad_rows, d_var_gg+img_size_pad_rows, d_var_bb+img_size_pad_rows, d_var_gb+img_size_pad_rows, d_var_gb+img_size_pad_rows, cols, rows);

	//inverse gg
	colorGuidedFilterHelper1_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_inv_gg, d_var_rr, d_var_bb, d_var_rb, d_var_rb, cols, rows);
	colorGuidedFilterHelper1_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_inv_gg+img_size_pad_rows, d_var_rr+img_size_pad_rows, d_var_bb+img_size_pad_rows, d_var_rb+img_size_pad_rows, d_var_rb+img_size_pad_rows, cols, rows);

	//inverse bb
	colorGuidedFilterHelper1_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_inv_bb, d_var_rr, d_var_gg, d_var_rg, d_var_rg, cols, rows);
	colorGuidedFilterHelper1_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_inv_bb+img_size_pad_rows, d_var_rr+img_size_pad_rows, d_var_gg+img_size_pad_rows, d_var_rg+img_size_pad_rows, d_var_rg+img_size_pad_rows, cols, rows);

	//inverse rg
	colorGuidedFilterHelper1_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_inv_rg, d_var_gb, d_var_rb, d_var_rg, d_var_bb, cols, rows);
	colorGuidedFilterHelper1_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_inv_rg+img_size_pad_rows, d_var_gb+img_size_pad_rows, d_var_rb+img_size_pad_rows, d_var_rg+img_size_pad_rows, d_var_bb+img_size_pad_rows, cols, rows);

	//inverse rb
	colorGuidedFilterHelper1_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_inv_rb, d_var_rg, d_var_gb, d_var_gg, d_var_rb, cols, rows);
	colorGuidedFilterHelper1_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_inv_rb+img_size_pad_rows, d_var_rg+img_size_pad_rows, d_var_gb+img_size_pad_rows, d_var_gg+img_size_pad_rows, d_var_rb+img_size_pad_rows, cols, rows);

	//inverse gb
	colorGuidedFilterHelper1_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_inv_gb, d_var_rb, d_var_rg, d_var_rr, d_var_gb, cols, rows);
	colorGuidedFilterHelper1_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_inv_gb+img_size_pad_rows, d_var_rb+img_size_pad_rows, d_var_rg+img_size_pad_rows, d_var_rr+img_size_pad_rows, d_var_gb+img_size_pad_rows, cols, rows);

	//cov_det
	colorGuidedFilterHelper2_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_cov_det, d_inv_rr, d_var_rr, d_inv_rg, d_var_rg, d_inv_rb, d_var_rb, cols, rows);
	colorGuidedFilterHelper2_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_cov_det+img_size_pad_rows, d_inv_rr+img_size_pad_rows, d_var_rr+img_size_pad_rows, d_inv_rg+img_size_pad_rows, d_var_rg+img_size_pad_rows, d_inv_rb+img_size_pad_rows, d_var_rb+img_size_pad_rows, cols, rows);

	// inv rr
	pointWiseDivison_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_inv_rr, d_inv_rr, d_cov_det, cols, rows);
	pointWiseDivison_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_inv_rr+img_size_pad_rows, d_inv_rr+img_size_pad_rows, d_cov_det+img_size_pad_rows, cols, rows);

	// inv rg
	pointWiseDivison_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_inv_rg, d_inv_rg, d_cov_det, cols, rows);
	pointWiseDivison_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_inv_rg+img_size_pad_rows, d_inv_rg+img_size_pad_rows, d_cov_det+img_size_pad_rows, cols, rows);

	// inv rb
	pointWiseDivison_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_inv_rb, d_inv_rb, d_cov_det, cols, rows);
	pointWiseDivison_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_inv_rb+img_size_pad_rows, d_inv_rb+img_size_pad_rows, d_cov_det+img_size_pad_rows, cols, rows);

	// inv gg
	pointWiseDivison_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_inv_gg, d_inv_gg, d_cov_det, cols, rows);
	pointWiseDivison_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_inv_gg+img_size_pad_rows, d_inv_gg+img_size_pad_rows, d_cov_det+img_size_pad_rows, cols, rows);

	// inv gb
	pointWiseDivison_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_inv_gb, d_inv_gb, d_cov_det, cols, rows);
	pointWiseDivison_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_inv_gb+img_size_pad_rows, d_inv_gb+img_size_pad_rows, d_cov_det+img_size_pad_rows, cols, rows);

	// inv bb
	pointWiseDivison_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_inv_bb, d_inv_bb, d_cov_det, cols, rows);
	pointWiseDivison_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_inv_bb+img_size_pad_rows, d_inv_bb+img_size_pad_rows, d_cov_det+img_size_pad_rows, cols, rows);


	float* d_right_input = d_right_cost_vol;
	float* d_left_input = d_left_cost_vol;
	for(int d=0; d<num_disp; d++, d_right_input+=img_size_pad_rows, d_left_input+=img_size_pad_rows)
	{
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_right_input, d_tmp_global, rows, cols, win_rad_bf);
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_left_input, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_mean_input_global, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_mean_input_global+img_size_pad_rows, rows, cols, win_rad_bf);
	
		//use d_corr_gi_global as a temporary container
		pointWiseMul_global<<< grid_size_mul, block_size_mul, 0, stream1>>>(d_r, d_right_input, d_corr_gi_global, rows, cols);
		pointWiseMul_global<<< grid_size_mul, block_size_mul, 0, stream2>>>(d_r+img_size_pad_rows, d_left_input, d_corr_gi_global+img_size_pad_rows, rows, cols);
		
		//mean_I_r
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_corr_gi_global, d_tmp_global, rows, cols, win_rad_bf);
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_corr_gi_global+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_mean_I_r, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_mean_I_r+img_size_pad_rows, rows, cols, win_rad_bf);

	
		//use d_corr_gi_global as a temporary container
		pointWiseMul_global<<< grid_size_mul, block_size_mul, 0, stream1>>>(d_g, d_right_input, d_corr_gi_global, rows, cols);
		pointWiseMul_global<<< grid_size_mul, block_size_mul, 0, stream2>>>(d_g+img_size_pad_rows, d_left_input, d_corr_gi_global+img_size_pad_rows, rows, cols);
		
		//mean_I_g
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_corr_gi_global, d_tmp_global, rows, cols, win_rad_bf);
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_corr_gi_global+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_mean_I_g, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_mean_I_g+img_size_pad_rows, rows, cols, win_rad_bf);


		//use d_corr_gi_global as a temporary container
		pointWiseMul_global<<< grid_size_mul, block_size_mul, 0, stream1>>>(d_b, d_right_input, d_corr_gi_global, rows, cols);
		pointWiseMul_global<<< grid_size_mul, block_size_mul, 0, stream2>>>(d_b+img_size_pad_rows, d_left_input, d_corr_gi_global+img_size_pad_rows, rows, cols);
		
		//mean_I_b
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_corr_gi_global, d_tmp_global, rows, cols, win_rad_bf);
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_corr_gi_global+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_mean_I_b, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_mean_I_b+img_size_pad_rows, rows, cols, win_rad_bf);


		//covariance
		colorGuidedFilterHelper3_global<<< grid_size_mul, block_size_mul, 0, stream1>>>(d_cov_I_r, d_mean_I_r, d_mean_r, d_mean_input_global, cols, rows);
		colorGuidedFilterHelper3_global<<< grid_size_mul, block_size_mul, 0, stream2>>>(d_cov_I_r+img_size_pad_rows, d_mean_I_r+img_size_pad_rows, d_mean_r+img_size_pad_rows, d_mean_input_global+img_size_pad_rows, cols, rows);

		colorGuidedFilterHelper3_global<<< grid_size_mul, block_size_mul, 0, stream1>>>(d_cov_I_g, d_mean_I_g, d_mean_g, d_mean_input_global, cols, rows);
		colorGuidedFilterHelper3_global<<< grid_size_mul, block_size_mul, 0, stream2>>>(d_cov_I_g+img_size_pad_rows, d_mean_I_g+img_size_pad_rows, d_mean_g+img_size_pad_rows, d_mean_input_global+img_size_pad_rows, cols, rows);

		colorGuidedFilterHelper3_global<<< grid_size_mul, block_size_mul, 0, stream1>>>(d_cov_I_b, d_mean_I_b, d_mean_b, d_mean_input_global, cols, rows);
		colorGuidedFilterHelper3_global<<< grid_size_mul, block_size_mul, 0, stream2>>>(d_cov_I_b+img_size_pad_rows, d_mean_I_b+img_size_pad_rows, d_mean_b+img_size_pad_rows, d_mean_input_global+img_size_pad_rows, cols, rows);
	
		//a
		colorGuidedFilterHelper2_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_a_r, d_inv_rr, d_cov_I_r, d_inv_rg, d_cov_I_g, d_inv_rb, d_cov_I_b, cols, rows);
		colorGuidedFilterHelper2_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_a_r+img_size_pad_rows, d_inv_rr+img_size_pad_rows, d_cov_I_r+img_size_pad_rows, d_inv_rg+img_size_pad_rows, d_cov_I_g+img_size_pad_rows, d_inv_rb+img_size_pad_rows, d_cov_I_b+img_size_pad_rows, cols, rows);

		colorGuidedFilterHelper2_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_a_g, d_inv_rg, d_cov_I_r, d_inv_gg, d_cov_I_g, d_inv_gb, d_cov_I_b, cols, rows);
		colorGuidedFilterHelper2_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_a_g+img_size_pad_rows, d_inv_rg+img_size_pad_rows, d_cov_I_r+img_size_pad_rows, d_inv_gg+img_size_pad_rows, d_cov_I_g+img_size_pad_rows, d_inv_gb+img_size_pad_rows, d_cov_I_b+img_size_pad_rows, cols, rows);

		colorGuidedFilterHelper2_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_a_b, d_inv_rb, d_cov_I_r, d_inv_gb, d_cov_I_g, d_inv_bb, d_cov_I_b, cols, rows);
		colorGuidedFilterHelper2_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_a_b+img_size_pad_rows, d_inv_rb+img_size_pad_rows, d_cov_I_r+img_size_pad_rows, d_inv_gb+img_size_pad_rows, d_cov_I_g+img_size_pad_rows, d_inv_bb+img_size_pad_rows, d_cov_I_b+img_size_pad_rows, cols, rows);

	
		//b
		colorGuidedFilterHelper4_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_b_global, d_mean_input_global, d_a_r, d_mean_r, d_a_g, d_mean_g, d_a_b, d_mean_b, cols, rows);
		colorGuidedFilterHelper4_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_b_global+img_size_pad_rows, d_mean_input_global+img_size_pad_rows, d_a_r+img_size_pad_rows, d_mean_r+img_size_pad_rows, d_a_g+img_size_pad_rows, d_mean_g+img_size_pad_rows, d_a_b+img_size_pad_rows, d_mean_b+img_size_pad_rows, cols, rows);


		//boxfilter a_r
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_a_r, d_tmp_global, rows, cols, win_rad_bf);
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_a_r+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_a_r, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_a_r+img_size_pad_rows, rows, cols, win_rad_bf);

		//boxfilter a_g
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_a_g, d_tmp_global, rows, cols, win_rad_bf);
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_a_g+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_a_g, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_a_g+img_size_pad_rows, rows, cols, win_rad_bf);		

		//boxfilter a_b
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_a_b, d_tmp_global, rows, cols, win_rad_bf);
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_a_b+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_a_b, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_a_b+img_size_pad_rows, rows, cols, win_rad_bf);				


		//boxfilter b
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_b_global, d_tmp_global, rows, cols, win_rad_bf);
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_b_global+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads, 0, stream1>>>(d_tmp_global, d_b_global, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_b_global+img_size_pad_rows, rows, cols, win_rad_bf);		

		//final
		colorGuidedFilterHelper5_global<<<grid_size_mul, block_size_mul, 0, stream1>>>(d_right_input, d_b_global, d_a_r, d_r, d_a_g, d_g, d_a_b, d_b, cols, rows);
		colorGuidedFilterHelper5_global<<<grid_size_mul, block_size_mul, 0, stream2>>>(d_left_input, d_b_global+img_size_pad_rows, d_a_r+img_size_pad_rows, d_r+img_size_pad_rows, d_a_g+img_size_pad_rows, d_g+img_size_pad_rows, d_a_b+img_size_pad_rows, d_b+img_size_pad_rows, cols, rows);

	}

	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
}


void *costVolumeGuidedFilterConcurrentCUDA(void *param)
{
	float *d_guide, *d_tmp, *d_mean_guide, /*d_corr_g,*/ *d_var_g;
	float *d_input, *d_mean_input, *d_corr_gi, *d_a, *d_b;

	HostThreadData *host_thread_data;

	host_thread_data = (HostThreadData*) param;

	const int rows = host_thread_data->rows;
	const int cols = host_thread_data->cols;
	const int num_disp = host_thread_data->num_disp;
	const int img_size_pad_rows = host_thread_data->img_size_pad_rows;
	const int win_rad_bf = host_thread_data->win_rad_bf;
	const float eps = host_thread_data->eps;


	d_guide = host_thread_data->d_guide;
	d_input = host_thread_data->d_input;
	d_tmp = host_thread_data->d_tmp;
	d_mean_guide = host_thread_data->d_mean_guide;
//	d_corr_g = host_thread_data->d_corr_g;
	d_var_g = host_thread_data->d_var_g;
	d_mean_input = host_thread_data->d_mean_input;
	d_corr_gi = host_thread_data->d_corr_gi;
	d_a = host_thread_data->d_a;
	d_b = host_thread_data->d_b;

	dim3 block_size_mul(32, 8);
	dim3 grid_size_mul((cols + block_size_mul.x - 1)/block_size_mul.x, (rows + block_size_mul.y - 1)/block_size_mul.y); 
	dim3 grid_size((cols+num_threads-1)/num_threads, (rows+num_threads-1)/num_threads);
	
	//mean guide
/*	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float)>>>(d_guide, d_tmp, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads>>>(d_tmp, d_mean_guide, rows, cols, win_rad_bf);

	//mean guide*guide
	pointWiseMul_global<<< grid_size_mul, block_size_mul>>>(d_guide, d_guide, d_corr_g, rows, cols);
	boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float)>>>(d_corr_g, d_tmp, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size, num_threads>>>(d_tmp, d_corr_g, rows, cols, win_rad_bf);

	//var guide
	pointWiseMul_global<<< grid_size_mul, block_size_mul>>>(d_mean_guide, d_mean_guide, d_var_g, rows, cols);
	pointWiseSub_global<<<grid_size_mul, block_size_mul>>>(d_corr_g, d_var_g, d_var_g, rows, cols);
*/
	for(int d=0; d<num_disp; d++, d_input += img_size_pad_rows)
	{
		// mean input
		//boxFilterCUDABaseline<<<grid_size_mul, block_size_mul>>>(d_input, d_mean_input, cols, rows, win_rad_bf);
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float)>>>(d_input, d_tmp, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads>>>(d_tmp, d_mean_input, rows, cols, win_rad_bf);

		//boxFilter_y_global<<<grid_size, num_threads>>>(d_tmp, d_input, rows, cols, win_rad_bf); continue;	//simple box filter cost volume

		// mean input*guide
		pointWiseMul_global<<< grid_size_mul, block_size_mul>>>(d_guide, d_input, d_corr_gi, rows, cols);
		//boxFilterCUDABaseline<<<grid_size_mul, block_size_mul>>>(d_corr_gi, d_tmp, cols, rows, win_rad_bf);
		//cudaMemcpy(d_corr_gi, d_tmp, sizeof(float)*img_size_pad_rows, cudaMemcpyDeviceToDevice);
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float)>>>(d_corr_gi, d_tmp, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads>>>(d_tmp, d_corr_gi, rows, cols, win_rad_bf);

		// a, b
		guidedFilterComputation1<<<grid_size_mul, block_size_mul>>>(d_a, d_corr_gi, d_mean_guide, d_mean_input, d_var_g, d_b, rows, cols, eps);

		// mean a
		//boxFilterCUDABaseline<<<grid_size_mul, block_size_mul>>>(d_a, d_tmp, cols, rows, win_rad_bf);
		//cudaMemcpy(d_a, d_tmp, sizeof(float)*img_size_pad_rows, cudaMemcpyDeviceToDevice);
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float)>>>(d_a, d_tmp, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads>>>(d_tmp, d_a, rows, cols, win_rad_bf);

		// mean b
		//boxFilterCUDABaseline<<<grid_size_mul, block_size_mul>>>(d_b, d_tmp, cols, rows, win_rad_bf);
		//cudaMemcpy(d_b, d_tmp, sizeof(float)*img_size_pad_rows, cudaMemcpyDeviceToDevice);
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float)>>>(d_b, d_tmp, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads>>>(d_tmp, d_b, rows, cols, win_rad_bf);

		// output
		guidedFilterComputation2<<<grid_size_mul, block_size_mul>>>(d_input, d_a, d_b, d_guide, rows, cols);
	}
	return NULL;
}


__global__ void guideMulInput(float* d_input, float* d_dest, const int rows, const int cols, const int side=0)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if(x>=cols || y>=rows) return;
	
	const int idx = y*cols + x;
	
	if(side == 0) d_dest[idx] = tex2D(tex_right_guide, x, y)*d_input[idx];
	else d_dest[idx] = tex2D(tex_left_guide, x, y)*d_input[idx];
}

//d_a = (d_corr_gi - d_mean_guide*d_mean_input)/(d_var_g+eps)
//d_b = d_mean_input - d_a*d_mean_guide
__global__ void guidedFilterComputation1Texture(float* d_a, float* d_corr_gi, float* d_mean_guide, float* d_mean_input, float* d_var_g, float* d_b, 
					const int rows, const int cols, const float eps, const int side=0)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x >= cols || y>= rows)
		return;		

	const int idx = y*cols+x;

	float mean_guide = side == 0 ? tex2D(tex_right_mean_guide, x, y) : tex2D(tex_left_mean_guide, x, y);
	float cov_g = side == 0 ? tex2D(tex_right_var_g, x, y) : tex2D(tex_left_var_g, x, y);

	float mean_input = d_mean_input[idx];
	float a = (d_corr_gi[idx] - mean_guide*mean_input)/(cov_g+eps);

	d_b[idx] = mean_input-a*mean_guide;
	d_a[idx] = a;
}

__global__ void guidedFilterComputation2Texture(float* d_input, float* d_a, float* d_b, float* d_guide, const int rows, const int cols, const int side =0)
{
	const int x = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	
	if(x >= cols || y>= rows)
		return;		

	const int idx = y*cols+x;

	if(side == 0)
		d_input[idx] = d_a[idx]*tex2D(tex_right_guide, x, y)+d_b[idx];
	else
		d_input[idx] = d_a[idx]*tex2D(tex_left_guide, x, y)+d_b[idx];
}

void costVolumeGuidedFilterNonConcurrentTextureCUDA(const int cols, const int rows, const int img_size_pad_rows, const int num_disp, const int win_rad_bf, const float eps, const int side=0/*right*/)
{

	float *d_guide, *d_tmp, *d_mean_guide, *d_var_g;
	float *d_input, *d_mean_input, *d_corr_gi, *d_a, *d_b;

	d_guide = d_guide_global;
	d_tmp = d_tmp_global;
	d_mean_guide = d_mean_guide_global;
	d_var_g = d_var_g_global;
	d_input = d_input_global;
	d_mean_input = d_mean_input_global;
	d_corr_gi = d_corr_gi_global;
	d_a = d_a_global;
	d_b = d_b_global;

	dim3 block_size_mul(32, 8);
	dim3 grid_size_mul((cols + block_size_mul.x - 1)/block_size_mul.x, (rows + block_size_mul.y - 1)/block_size_mul.y); 
	dim3 grid_size( (cols+num_threads-1)/num_threads, (rows+num_threads-1)/num_threads);

	for(int d=0; d<num_disp; d++, d_input += img_size_pad_rows)
	{
		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float)>>>(d_input, d_tmp, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads>>>(d_tmp, d_mean_input, rows, cols, win_rad_bf);

		guideMulInput<<< grid_size_mul, block_size_mul>>>(d_input, d_corr_gi, rows, cols, side);

		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float)>>>(d_corr_gi, d_tmp, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads>>>(d_tmp, d_corr_gi, rows, cols, win_rad_bf);

		//d_a = (d_corr_gi - d_mean_guide*d_mean_input)/(d_var_g+eps)
		//d_b = d_mean_input - d_a*d_mean_guide
		guidedFilterComputation1Texture<<<grid_size_mul, block_size_mul>>>(d_a, d_corr_gi, d_mean_guide, d_mean_input, d_var_g, d_b, rows, cols, eps, side);

		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float)>>>(d_a, d_tmp, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads>>>(d_tmp, d_a, rows, cols, win_rad_bf);

		boxFilter_x_global_shared<<<grid_size, num_threads, num_threads*(num_threads+1)*sizeof(float)>>>(d_b, d_tmp, rows, cols, win_rad_bf);
		boxFilter_y_global<<<grid_size, num_threads>>>(d_tmp, d_b, rows, cols, win_rad_bf);

		// d_input = d_a*d_guide-d_b
		guidedFilterComputation2Texture<<< grid_size_mul, block_size_mul>>>(d_input, d_a, d_b, d_guide, rows, cols, side);
	}
}

void costVolumeGuidedFilterOMP(const cv::Mat & guide, float* d_cost_vol, const int cols, const int rows, const int img_size_pad_rows, 
				const int num_disp, double eps = pow(0.01,2.0)*255*255, const int gfsize = 9)
{
//	StartTimer();
	cv::Mat meanguide;
	cv::blur(guide, meanguide, cv::Size(gfsize,gfsize));
	cv::Mat corr_g;
	cv::multiply(guide,guide,corr_g);
	cv::blur(corr_g,corr_g, cv::Size(gfsize,gfsize));
	cv::Mat var_g;
	cv::multiply(meanguide, meanguide,var_g);
	var_g = var_g*(-1) + corr_g;
//	std::cout<<"guided filter prepare:"<<GetTimer()<<std::endl;

	StartTimer();
#pragma omp parallel for
	for(int d=0; d<num_disp; d++)
	{
		cv::Mat p;
		p.create(rows, cols, CV_32F);
		int pitch = d*img_size_pad_rows;
		//cudaMemcpy(p.ptr<float>(0), d_cost_vol+pitch, sizeof(float)*rows*cols, cudaMemcpyDeviceToHost);
		std::memcpy(p.ptr<float>(0), d_cost_vol+pitch, sizeof(float)*rows*cols);

	/*	cv::Mat input = p;
		cv::Mat meaninput;
		cv::blur(input, meaninput, cv::Size(gfsize,gfsize));

		cv::Mat corr_gi;
		cv::multiply(input, guide, corr_gi);
		cv::blur(corr_gi, corr_gi, cv::Size(gfsize,gfsize));
		
		cv::Mat cov_gi;
		cv::multiply(meanguide, meaninput, cov_gi);
		cov_gi = cov_gi*(-1) + corr_gi;
		
		cv::Mat a = cov_gi/(var_g+eps);
		cv::Mat b;
		cv::multiply(a, meanguide, b);
		b = b*(-1) + meaninput;
		
		cv::blur(a, a, cv::Size(gfsize,gfsize));
		cv::blur(b, b, cv::Size(gfsize,gfsize));
		
		cv::multiply(a, guide, input);
		input += b;*/
		
		cv::Mat b = guidedFilter(p, guide, gfsize, eps); //b.copyTo(p);

		//cudaMemcpy( d_cost_vol+pitch, p.ptr<float>(0), sizeof(float)*rows*cols, cudaMemcpyHostToDevice);
		std::memcpy( d_cost_vol+pitch, b.ptr<float>(0), sizeof(float)*rows*cols);	//change p to b
	}

//	std::cout<<"cost volume filter:"<<GetTimer()<<std::endl;
}



void costVolumeStereoPlusVariationalDenoise(const cv::Mat& left_img, const cv::Mat& right_img, const int min_disp, const int max_disp)
{
	const int cols = left_img.cols;
	const int rows = left_img.rows;

	const size_t num_disp = max_disp-min_disp+1;

	double eps;	//guided filter
	int win_rad_bf;

	// variantional denoising
	float lambda;

	// split channels
	std::vector<cv::Mat> cvLeftBGR_v;
	std::vector<cv::Mat> cvRightBGR_v;

	cv::split(left_img, cvLeftBGR_v);
	cv::split(right_img, cvRightBGR_v);

	// convert to float
	cv::Mat cvLeftB_f;
	cv::Mat cvLeftG_f;
	cv::Mat cvLeftR_f;
	cv::Mat cvRightB_f;
	cv::Mat cvRightG_f;
	cv::Mat cvRightR_f;

	cvLeftBGR_v[0].convertTo(cvLeftB_f, CV_32F);
	cvLeftBGR_v[1].convertTo(cvLeftG_f, CV_32F);
	cvLeftBGR_v[2].convertTo(cvLeftR_f, CV_32F);	
	cvRightBGR_v[0].convertTo(cvRightB_f, CV_32F);	
	cvRightBGR_v[1].convertTo(cvRightG_f, CV_32F);	
	cvRightBGR_v[2].convertTo(cvRightR_f, CV_32F);	


	cv::Mat h_left_denoise_cv32f, h_right_denoise_cv32f;
	float* d_left_denoise = NULL;
	float* d_right_denoise = NULL;

	// BGR to grayscale
	cv::Mat cvLeftGray, cvRightGray;
	cv::cvtColor(left_img, cvLeftGray, CV_BGR2GRAY);
	cv::cvtColor(right_img, cvRightGray, CV_BGR2GRAY);	

	// convert to float
	cv::Mat cvLeftGray_f, cvRightGray_f;

	cv::Mat left_bgr_cv_f, right_bgr_cv_f;	

	const double img_norm_scale_factor = 1.0;
	
	cvLeftGray.convertTo(cvLeftGray_f, CV_32F, img_norm_scale_factor);
	cvRightGray.convertTo(cvRightGray_f, CV_32F, img_norm_scale_factor);

	left_img.convertTo(left_bgr_cv_f, CV_32FC3, img_norm_scale_factor);
	right_img.convertTo(right_bgr_cv_f, CV_32FC3, img_norm_scale_factor);
	

	const size_t img_size = cols*rows;
	const int rows_pad = (rows+num_threads-1)/num_threads*num_threads;
	const int img_size_pad_rows = cols*rows_pad;

	cv::Mat h_left_disp_cv32f, h_right_disp_cv32f;
	h_left_disp_cv32f.create(rows, cols, CV_32F);
	h_right_disp_cv32f.create(rows, cols, CV_32F);

	float *d_left_bgr = NULL;
	float *d_right_bgr = NULL;

	cudaMalloc(&d_left_bgr, img_size_pad_rows*3*sizeof(float));
	cudaMalloc(&d_right_bgr, img_size_pad_rows*3*sizeof(float));
	cudaMalloc(&d_b, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_g, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_r, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_mean_b, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_mean_g, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_mean_r, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_var_rr, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_var_rg, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_var_rb, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_var_gg, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_var_gb, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_var_bb, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_inv_rr, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_inv_rg, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_inv_rb, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_inv_gg, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_inv_gb, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_inv_bb, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_cov_det, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_a_b, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_a_g, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_a_r, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_mean_I_b, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_mean_I_g, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_mean_I_r, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_cov_I_b, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_cov_I_g, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_cov_I_r, 2*img_size_pad_rows*sizeof(float));


	cudaMemcpy(d_left_bgr, left_bgr_cv_f.ptr<float>(0), sizeof(float)*img_size*3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_right_bgr, right_bgr_cv_f.ptr<float>(0), sizeof(float)*img_size*3, cudaMemcpyHostToDevice);

	cudaMemcpy(d_b, cvRightB_f.ptr<float>(0), sizeof(float)*img_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_g, cvRightG_f.ptr<float>(0), sizeof(float)*img_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, cvRightR_f.ptr<float>(0), sizeof(float)*img_size, cudaMemcpyHostToDevice);

	cudaMemcpy(d_b+img_size_pad_rows, cvLeftB_f.ptr<float>(0), sizeof(float)*img_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_g+img_size_pad_rows, cvLeftG_f.ptr<float>(0), sizeof(float)*img_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_r+img_size_pad_rows, cvLeftR_f.ptr<float>(0), sizeof(float)*img_size, cudaMemcpyHostToDevice);

	// allocate floating disparity map(global memory)
	float* d_left_disp = NULL;
	float* d_right_disp = NULL;

	float* d_left_gray_zero_mean = NULL;
	float* d_right_gray_zero_mean = NULL;

	cudaMalloc(&d_left_disp, img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_right_disp, img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_left_gray_zero_mean, img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_right_gray_zero_mean, img_size_pad_rows*sizeof(float));

	cudaMalloc(&d_left_gray, img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_right_gray, img_size_pad_rows*sizeof(float));

	cudaMalloc(&d_left_cost_vol, img_size_pad_rows*num_disp*sizeof(float));
	cudaMalloc(&d_right_cost_vol, img_size_pad_rows*num_disp*sizeof(float));

	cudaMalloc(&d_mean_guide_global, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_corr_g_global, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_var_g_global, 2*img_size_pad_rows*sizeof(float));
	
	cudaMalloc(&d_tmp_global, 2*img_size_pad_rows*sizeof(float));	
	cudaMalloc(&d_mean_input_global, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_corr_gi_global, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_a_global, 2*img_size_pad_rows*sizeof(float));
	cudaMalloc(&d_b_global, 2*img_size_pad_rows*sizeof(float));

	//dst, src, size
	cudaMemcpy(d_left_gray, cvLeftGray_f.ptr<float>(0), sizeof(float)*img_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_right_gray, cvRightGray_f.ptr<float>(0), sizeof(float)*img_size, cudaMemcpyHostToDevice);

	dim3 block_size_mul(32, 8);
	dim3 grid_size_mul((cols + block_size_mul.x - 1)/block_size_mul.x, (rows + block_size_mul.y - 1)/block_size_mul.y); 
	dim3 grid_size_g( (cols+num_threads-1)/num_threads, (rows+num_threads-1)/num_threads);


	/************************preprocess stereo images by subtracting local mean*********************/
	StartTimer();
	dim3 block_size_global(32, 8);
	dim3 grid_size_global((cols + block_size_global.x - 1)/block_size_global.x, (rows + block_size_global.y - 1)/block_size_global.y); 
	dim3 grid_size_box( (cols+num_threads-1)/num_threads, (rows+num_threads-1)/num_threads );
	const int win_rad_lm = 1;

	//nonconcurrent
#if 1
#if 0
	boxFilter_x_global_shared<<<grid_size_box, num_threads, num_threads*(num_threads+1)*sizeof(float)>>>(d_right_gray, d_tmp_global, rows, cols, win_rad_lm);
	boxFilter_y_global<<<grid_size_box, num_threads>>>(d_tmp_global, d_right_gray_zero_mean, rows, cols, win_rad_lm);
	pointWiseSub_global<<<grid_size_global, block_size_global>>>(d_right_gray, d_right_gray_zero_mean, d_right_gray_zero_mean, rows, cols);

	boxFilter_x_global_shared<<<grid_size_box, num_threads, num_threads*(num_threads+1)*sizeof(float)>>>(d_left_gray, d_tmp_global, rows, cols, win_rad_lm);
	boxFilter_y_global<<<grid_size_box, num_threads>>>(d_tmp_global, d_left_gray_zero_mean, rows, cols, win_rad_lm);
	pointWiseSub_global<<<grid_size_global, block_size_global>>>(d_left_gray, d_left_gray_zero_mean, d_left_gray_zero_mean, rows, cols);
	cudaStreamSynchronize(0);

#else	
	//concurrent
	cudaStream_t stream1, stream2;
	cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);

	boxFilter_x_global_shared<<<grid_size_box, num_threads, num_threads*(num_threads+1)*sizeof(float), stream1>>>(d_right_gray, d_tmp_global, rows, cols, win_rad_lm);
	boxFilter_x_global_shared<<<grid_size_box, num_threads, num_threads*(num_threads+1)*sizeof(float), stream2>>>(d_left_gray, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_lm);

	boxFilter_y_global<<<grid_size_box, num_threads, 0, stream1>>>(d_tmp_global, d_right_gray_zero_mean, rows, cols, win_rad_lm);
	boxFilter_y_global<<<grid_size_box, num_threads, 0, stream2>>>(d_tmp_global+img_size_pad_rows, d_left_gray_zero_mean, rows, cols, win_rad_lm);

	pointWiseSub_global<<<grid_size_global, block_size_global, 0, stream1>>>(d_right_gray, d_right_gray_zero_mean, d_right_gray_zero_mean, rows, cols);
	pointWiseSub_global<<<grid_size_global, block_size_global, 0, stream2>>>(d_left_gray, d_left_gray_zero_mean, d_left_gray_zero_mean, rows, cols);

	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);
#endif
	std::cout<<"subtract local mean:"<<GetTimer()<<std::endl;
#endif	//end of substract local mean	


	/**************************build cost volume*********************************/
	dim3 grid_size_gm((cols + block_size_global.x - 1)/block_size_global.x, (rows + block_size_global.y - 1)/block_size_global.y, num_disp); 
	dim3 block_size_sm(cols, 1);
	dim3 grid_size_sm(rows,1,1);


	//build cost volume
	StartTimer();
#if 1
	
/*	const int win_radius = 0;
	buildCostVolumeSharedMemory<<<grid_size_sm, block_size_sm, 2*cols*(2*win_radius+1)*sizeof(float)>>>
					(d_left_gray_zero_mean, d_right_gray_zero_mean, 
					//(d_left_gray, d_right_gray,
					d_left_cost_vol, d_right_cost_vol, rows, cols, min_disp, max_disp, win_radius, img_size_pad_rows);
*/
	buildCostVolumeSharedMemoryBGR<<<grid_size_sm, block_size_sm, 6*cols*sizeof(float)>>>
					(d_left_bgr, d_right_bgr, d_left_cost_vol, d_right_cost_vol, rows, cols, min_disp, max_disp, img_size_pad_rows);


	cudaStreamSynchronize(0);
	std::cout<<"build cost volume shared memory:"<<GetTimer()<<std::endl;

#else
	buildCostVolume<<<grid_size_gm, block_size_global>>>(d_left_gray_zero_mean, d_right_gray_zero_mean, d_left_cost_vol, d_right_cost_vol, rows, cols, 
								min_disp, max_disp, win_radius, img_size_pad_rows);
	cudaStreamSynchronize(0);
	std::cout<<"build cost volume global memory:"<<GetTimer()<<std::endl;
#endif


	/************************************cost volume filtering*************************************/
	//guided filter cost volume CUDA
	eps = pow(0.01, 2.0)*255*255;	// SUPER IMPORTANT, too large = box filter, too small = guide image
	win_rad_bf = 9;
#if 1	//start of cost volume filter

	//2 threads generating 2 streams
#if 0
	//precompute 
/*	cudaChannelFormatDesc channelDesc;
        channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaMallocArray(&d_array, &channelDesc, cols, rows);
	cudaMallocArray(&d_array_lh, &channelDesc, cols, rows);
	cudaMallocArray(&d_array_rh, &channelDesc, cols, rows);

	cudaMallocArray(&d_array_left_guide, &channelDesc, cols, rows_pad);
	cudaMallocArray(&d_array_left_mean_guide, &channelDesc, cols, rows_pad);
	cudaMallocArray(&d_array_left_var_g, &channelDesc, cols, rows_pad);

	cudaMallocArray(&d_array_right_guide, &channelDesc, cols, rows_pad);
	cudaMallocArray(&d_array_right_mean_guide, &channelDesc, cols, rows_pad);
	cudaMallocArray(&d_array_right_var_g, &channelDesc, cols, rows_pad);

    	// set texture parameters
    	tex.addressMode[0] = cudaAddressModeClamp;
    	tex.addressMode[1] = cudaAddressModeClamp;
    	tex.filterMode = cudaFilterModePoint;
    	tex.normalized = false;

    	tex_lh.addressMode[0] = cudaAddressModeClamp;
    	tex_lh.addressMode[1] = cudaAddressModeClamp;
    	tex_lh.filterMode = cudaFilterModePoint;
    	tex_lh.normalized = false;

    	tex_rh.addressMode[0] = cudaAddressModeClamp;
    	tex_rh.addressMode[1] = cudaAddressModeClamp;
    	tex_rh.filterMode = cudaFilterModePoint;
    	tex_rh.normalized = false;

    	tex_left_guide.addressMode[0] = cudaAddressModeClamp;
    	tex_left_guide.addressMode[1] = cudaAddressModeClamp;
    	tex_left_guide.filterMode = cudaFilterModePoint;
    	tex_left_guide.normalized = false;

    	tex_left_mean_guide.addressMode[0] = cudaAddressModeClamp;
    	tex_left_mean_guide.addressMode[1] = cudaAddressModeClamp;
    	tex_left_mean_guide.filterMode = cudaFilterModePoint;
    	tex_left_mean_guide.normalized = false;

    	tex_left_var_g.addressMode[0] = cudaAddressModeClamp;
    	tex_left_var_g.addressMode[1] = cudaAddressModeClamp;
    	tex_left_var_g.filterMode = cudaFilterModePoint;
    	tex_left_var_g.normalized = false;

    	tex_right_guide.addressMode[0] = cudaAddressModeClamp;
    	tex_right_guide.addressMode[1] = cudaAddressModeClamp;
    	tex_right_guide.filterMode = cudaFilterModePoint;
    	tex_right_guide.normalized = false;

    	tex_right_mean_guide.addressMode[0] = cudaAddressModeClamp;
    	tex_right_mean_guide.addressMode[1] = cudaAddressModeClamp;
    	tex_right_mean_guide.filterMode = cudaFilterModePoint;
    	tex_right_mean_guide.normalized = false;

    	tex_right_var_g.addressMode[0] = cudaAddressModeClamp;
    	tex_right_var_g.addressMode[1] = cudaAddressModeClamp;
    	tex_right_var_g.filterMode = cudaFilterModePoint;
    	tex_right_var_g.normalized = false;

	cudaBindTextureToArray(tex_right_guide, d_array_right_guide);
	cudaBindTextureToArray(tex_right_mean_guide, d_array_right_mean_guide);
	cudaBindTextureToArray(tex_right_var_g, d_array_right_var_g);

	cudaBindTextureToArray(tex_left_guide, d_array_left_guide);
	cudaBindTextureToArray(tex_left_mean_guide, d_array_left_mean_guide);
	cudaBindTextureToArray(tex_left_var_g, d_array_left_var_g);
*/
	StartTimer();
	// prepare guide, mean_guide, var_g
	cudaStream_t stream3;
	cudaStream_t stream4;
	cudaStreamCreateWithFlags(&stream3, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&stream4, cudaStreamNonBlocking);

	boxFilter_x_global_shared<<<grid_size_g, num_threads, num_threads*(num_threads+1)*sizeof(float), stream3>>>(d_right_gray, d_tmp_global, rows, cols, win_rad_bf);
	boxFilter_x_global_shared<<<grid_size_g, num_threads, num_threads*(num_threads+1)*sizeof(float), stream4>>>(d_left_gray, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);

	boxFilter_y_global<<<grid_size_g, num_threads, 0, stream3>>>(d_tmp_global, d_mean_guide_global, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size_g, num_threads, 0, stream4>>>(d_tmp_global+img_size_pad_rows, d_mean_guide_global+img_size_pad_rows, rows, cols, win_rad_bf);

	pointWiseMul_global<<< grid_size_mul, block_size_mul, 0, stream3>>>(d_right_gray, d_right_gray, d_corr_g_global, rows, cols);
	pointWiseMul_global<<< grid_size_mul, block_size_mul, 0, stream4>>>(d_left_gray, d_left_gray, d_corr_g_global+img_size_pad_rows, rows, cols);

	boxFilter_x_global_shared<<<grid_size_g, num_threads, num_threads*(num_threads+1)*sizeof(float), stream3>>>(d_corr_g_global, d_tmp_global, rows, cols, win_rad_bf);
	boxFilter_x_global_shared<<<grid_size_g, num_threads, num_threads*(num_threads+1)*sizeof(float), stream4>>>(d_corr_g_global+img_size_pad_rows, d_tmp_global+img_size_pad_rows, rows, cols, win_rad_bf);

	boxFilter_y_global<<<grid_size_g, num_threads, 0, stream3>>>(d_tmp_global, d_corr_g_global, rows, cols, win_rad_bf);
	boxFilter_y_global<<<grid_size_g, num_threads, 0, stream4>>>(d_tmp_global+img_size_pad_rows, d_corr_g_global+img_size_pad_rows, rows, cols, win_rad_bf);

	pointWiseMul_global<<< grid_size_mul, block_size_mul, 0, stream3>>>(d_mean_guide_global, d_mean_guide_global, d_var_g_global, rows, cols);
	pointWiseMul_global<<< grid_size_mul, block_size_mul, 0, stream4>>>(d_mean_guide_global+img_size_pad_rows, d_mean_guide_global+img_size_pad_rows, d_var_g_global+img_size_pad_rows, rows, cols);

	pointWiseSub_global<<<grid_size_mul, block_size_mul, 0, stream3>>>(d_corr_g_global, d_var_g_global, d_var_g_global, rows, cols);
	pointWiseSub_global<<<grid_size_mul, block_size_mul, 0, stream4>>>(d_corr_g_global+img_size_pad_rows, d_var_g_global+img_size_pad_rows, d_var_g_global+img_size_pad_rows, rows, cols);


/*	cudaMemcpyToArrayAsync(d_array_right_guide, 0, 0, d_right_gray, img_size_pad_rows*sizeof(float), cudaMemcpyDeviceToDevice, stream3);
	cudaMemcpyToArrayAsync(d_array_left_guide, 0, 0, d_left_gray, img_size_pad_rows*sizeof(float), cudaMemcpyDeviceToDevice, stream4);

	cudaMemcpyToArrayAsync(d_array_right_mean_guide, 0, 0, d_mean_guide_global, img_size_pad_rows*sizeof(float), cudaMemcpyDeviceToDevice, stream3);	
	cudaMemcpyToArrayAsync(d_array_left_mean_guide, 0, 0, d_mean_guide_global+img_size_pad_rows, img_size_pad_rows*sizeof(float), cudaMemcpyDeviceToDevice, stream4);	

	cudaMemcpyToArrayAsync(d_array_right_var_g, 0, 0, d_var_g_global, img_size_pad_rows*sizeof(float), cudaMemcpyDeviceToDevice, stream3);	
	cudaMemcpyToArrayAsync(d_array_left_var_g, 0, 0, d_var_g_global+img_size_pad_rows, img_size_pad_rows*sizeof(float), cudaMemcpyDeviceToDevice, stream4);	
*/

	cudaStreamDestroy(stream3);
	cudaStreamDestroy(stream4);

	const int num_host_threads = 2;

	HostThreadData host_thread_data_array[num_host_threads];

	for(int i=0; i<num_host_threads; i++)
	{
		host_thread_data_array[i].id = i;
		host_thread_data_array[i].cols = cols;
		host_thread_data_array[i].rows = rows;
		host_thread_data_array[i].img_size_pad_rows = img_size_pad_rows;
		host_thread_data_array[i].num_disp = num_disp;
		host_thread_data_array[i].eps = eps;
		host_thread_data_array[i].win_rad_bf = win_rad_bf;
		host_thread_data_array[i].d_guide = i==0 ? d_right_gray : d_left_gray;
		host_thread_data_array[i].d_input = i==0 ? d_right_cost_vol : d_left_cost_vol;

		host_thread_data_array[i].d_tmp = d_tmp_global + i*img_size_pad_rows;
		host_thread_data_array[i].d_mean_guide = d_mean_guide_global + i*img_size_pad_rows;
		host_thread_data_array[i].d_corr_g = d_corr_g_global + i*img_size_pad_rows;
		host_thread_data_array[i].d_var_g = d_var_g_global + i*img_size_pad_rows;
		host_thread_data_array[i].d_mean_input = d_mean_input_global + i*img_size_pad_rows;
		host_thread_data_array[i].d_corr_gi = d_corr_gi_global + i*img_size_pad_rows;
		host_thread_data_array[i].d_a = d_a_global + i*img_size_pad_rows;
		host_thread_data_array[i].d_b = d_b_global + i*img_size_pad_rows;
	}

	StartTimer();

	pthread_t host_threads[num_host_threads];

	for(int i=0; i<num_host_threads; i++)
		if(pthread_create(&host_threads[i], NULL, costVolumeGuidedFilterConcurrentCUDA, (void*) &host_thread_data_array[i]))
			std::cout<<"Error creating thread "<<i<<"\n";

	for(int i=0; i<num_host_threads; i++)
		if(pthread_join(host_threads[i], NULL)) std::cout<<"Error joining thread "<<i<<"\n";

#else	//2 concurrent streams
	cudaDeviceSynchronize();
	//costVolumeGuidedFilterCUDA2Streams(cols, rows, img_size_pad_rows, num_disp, win_rad_bf, eps);
	costVolumeColorGuidedFilterCUDA2Streams(cols, rows, img_size_pad_rows, num_disp, win_rad_bf, eps);
#endif
	cudaDeviceSynchronize();
	std::cout<<"cost volume filter CUDA:"<<GetTimer()<<std::endl;

	cudaFree(d_tmp_global);
	cudaFree(d_mean_guide_global);
	cudaFree(d_corr_g_global);
	cudaFree(d_var_g_global);
	cudaFree(d_mean_input_global);
	cudaFree(d_corr_gi_global);
	cudaFree(d_a_global);
	cudaFree(d_b_global);
	cudaFree(d_b);
	cudaFree(d_g);
	cudaFree(d_r);
	cudaFree(d_mean_b);
	cudaFree(d_mean_g);
	cudaFree(d_mean_r);
	cudaFree(d_mean_I_b);
	cudaFree(d_mean_I_g);
	cudaFree(d_mean_I_r);
	cudaFree(d_cov_I_b);
	cudaFree(d_cov_I_g);
	cudaFree(d_cov_I_r);
	cudaFree(d_var_rr);
	cudaFree(d_var_rg);
	cudaFree(d_var_rb);
	cudaFree(d_var_gg);
	cudaFree(d_var_gb);
	cudaFree(d_var_bb);
	cudaFree(d_inv_rr);
	cudaFree(d_inv_rg);
	cudaFree(d_inv_rb);
	cudaFree(d_inv_gg);
	cudaFree(d_inv_gb);
	cudaFree(d_inv_bb);
	cudaFree(d_cov_det);
	cudaFree(d_a_r);
	cudaFree(d_a_g);
	cudaFree(d_a_b);

	//freeTextures();

#else
	// guided filter cost volume, the larger, the more blurry
	int gfsize = 2*win_rad_bf+1;
	costVolumeGuidedFilterOMP(cvRightGray_f, d_right_cost_vol, cols, rows, img_size_pad_rows, num_disp, eps, gfsize);
	costVolumeGuidedFilterOMP(cvLeftGray_f, d_left_cost_vol, cols, rows, img_size_pad_rows, num_disp, eps, gfsize);
#endif	//end of cost volume filtering


	/******************************select disparity from cost volume***********************/
	StartTimer();
	selectDisparity<<<grid_size_global, block_size_global>>>(d_left_cost_vol, d_left_disp, rows, cols, min_disp, max_disp, img_size_pad_rows);
	selectDisparity<<<grid_size_global, block_size_global>>>(d_right_cost_vol, d_right_disp, rows, cols, min_disp, max_disp, img_size_pad_rows);
	cudaStreamSynchronize(0);
	std::cout<<"select disparity: "<<GetTimer()<<std::endl;

#define denoise

#ifdef denoise
	// variational denoising weighted Huber
	h_left_denoise_cv32f.create(rows, cols, CV_32F);
	h_right_denoise_cv32f.create(rows, cols, CV_32F);
	cudaMalloc(&d_left_denoise, img_size*sizeof(float));
	cudaMalloc(&d_right_denoise, img_size*sizeof(float));	
	lambda = 0.5f;
	variationalDisparityDenoise(d_left_denoise, d_left_gray, d_left_disp, rows, cols, min_disp, max_disp, lambda);
	variationalDisparityDenoise(d_right_denoise, d_right_gray, d_right_disp, rows, cols, min_disp, max_disp, lambda);
#endif


	/**********************************occlusion handling and holes filling*********************************/
#if 0
	StartTimer();
	bool set_occlusion_to_zero = true;
	handleOcclusionSharedMemory<<<grid_size_sm, block_size_sm, 4*cols*sizeof(float)>>>(d_left_disp, d_right_disp, rows, cols, min_disp, max_disp, 1.0f, set_occlusion_to_zero);
	handleOcclusionSharedMemory<<<grid_size_sm, block_size_sm, 4*cols*sizeof(float)>>>(d_left_denoise, d_right_denoise, rows, cols, min_disp, max_disp, 1.0f, set_occlusion_to_zero);
	cudaStreamSynchronize(0);
	std::cout<<"occlusion: "<<GetTimer()<<std::endl;
#endif


	// copy disparity map back
	cudaMemcpy(h_left_disp_cv32f.ptr<float>(0), d_left_disp, sizeof(float)*img_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_right_disp_cv32f.ptr<float>(0), d_right_disp, sizeof(float)*img_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_left_denoise_cv32f.ptr<float>(0), d_left_denoise, sizeof(float)*img_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_right_denoise_cv32f.ptr<float>(0), d_right_denoise, sizeof(float)*img_size, cudaMemcpyDeviceToHost);

/*
#if USE_PCL && 0
	StartTimer();
	pcl::PointCloud<pcl::PointXYZ>::Ptr disparity_cloud(new pcl::PointCloud<pcl::PointXYZ>);

	for(int y=0; y<rows; y++)
	{
		for(int x=0; x<cols; x++)
		{
			pcl::PointXYZ p;
			p.x = (float)x;
			p.y = (float)y;
			p.z = 10.f*h_right_disp_cv32f.at<float>(y,x);
			disparity_cloud->push_back(p);
		}
	}

	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);	
	pcl::PointCloud<pcl::PointNormal> mls_points;

  	// Init object (second point type is for the normals, even if unused)
  	pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
 
  	mls.setComputeNormals (true);

	// Set parameters
	mls.setInputCloud (disparity_cloud);
	mls.setPolynomialFit (true);
	mls.setSearchMethod (tree);
	mls.setSearchRadius (7);
	mls.setPolynomialOrder(1);

	// Reconstruct
	mls.process (mls_points);

	for(int i=0; i<img_size; i++)
	{
		h_right_disp_cv32f.ptr<float>(0)[i] = disparity_cloud->points[i].z*0.1f;
	}
	std::cout<<"mls:"<<GetTimer()<<std::endl;
#endif
*/
	cv::Mat left_disp_cv8u, right_disp_cv8u, left_denoise_cv8u, right_denoise_cv8u;
	left_disp_cv8u.create(rows, cols, CV_8UC1);
	right_disp_cv8u.create(rows, cols, CV_8UC1);
	left_denoise_cv8u.create(rows, cols, CV_8UC1);
	right_denoise_cv8u.create(rows, cols, CV_8UC1);

	h_left_denoise_cv32f.convertTo(left_denoise_cv8u, CV_8UC1, 4.0);
	h_right_denoise_cv32f.convertTo(right_denoise_cv8u, CV_8UC1, 4.0);

	h_left_disp_cv32f.convertTo(left_disp_cv8u, CV_8UC1, 4.0);
	h_right_disp_cv32f.convertTo(right_disp_cv8u, CV_8UC1, 4.0);


#if 1
	cv::Mat color_map0, color_map1, color_map2, color_map3;
	cv::applyColorMap(left_disp_cv8u, color_map0, cv::COLORMAP_JET);
	cv::applyColorMap(right_disp_cv8u, color_map1, cv::COLORMAP_JET);	
	cv::imshow("left disp", color_map0);
	cv::imshow("right disp", color_map1);

#ifdef denoise
	cv::applyColorMap(left_denoise_cv8u, color_map2, cv::COLORMAP_JET);
	cv::applyColorMap(right_denoise_cv8u, color_map3, cv::COLORMAP_JET);
	cv::imshow("left denoise", color_map2);
	cv::imshow("right denoise", color_map3);
#endif

	cv::waitKey(0);
#endif

#if USE_PCL
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));

/*	for(int y=win_radius; y<rows-win_radius; y++)
	{
		for(int x=win_radius; x<cols-win_radius; x++)
		{
			int idx = y*cols+x;
			pcl::PointXYZRGB p;
			p.x = mls_points.points[idx].x; 
			p.y = mls_points.points[idx].y;
			p.z = mls_points.points[idx].z;
			cv::Vec3b bgr = right_img.at<cv::Vec3b>(y, x);
			p.b = bgr.val[0];
			p.g = bgr.val[1];
			p.r = bgr.val[2];
			cloud->push_back(p);
		}

	}


	viewer->addPointCloud(cloud, "mls", 0);
	viewer->spin();
	viewer->removePointCloud("mls", 0);
	cloud->points.clear();*/

	for(int y=0; y<rows; y++)
	{
		for(int x=0; x<cols; x++)
		{
			int idx = y*cols+x;
			pcl::PointXYZRGB p;
			p.x = x; p.y = y;
			p.z = 10.f*(max_disp - h_right_disp_cv32f.ptr<float>(0)[idx]);
			cv::Vec3b bgr = right_img.at<cv::Vec3b>(y, x);
			p.b = bgr.val[0];
			p.g = bgr.val[1];
			p.r = bgr.val[2];
			cloud->push_back(p);
		}

	}


	//viewer->addPointCloud(cloud,"cloud", 0);
	//viewer->spin();
	//viewer->removePointCloud("cloud",0);
	//cloud->points.clear();

	for(int y=0; y<rows; y++)
	{
		for(int x=0; x<cols; x++)
		{
			int idx = y*cols+x;
			pcl::PointXYZRGB p;
			p.x = x + 1.1f*cols; 
			p.y = y;
			p.z = 10.f*(max_disp - h_right_denoise_cv32f.ptr<float>(0)[idx]);
			cv::Vec3b bgr = right_img.at<cv::Vec3b>(y, x);
			p.b = bgr.val[0];
			p.g = bgr.val[1];
			p.r = bgr.val[2];
			cloud->push_back(p);
		}

	}

	viewer->addPointCloud(cloud,"cloud denoise", 0);
	viewer->spin();
	
#endif	


	cudaFree(d_left_disp);
	cudaFree(d_right_disp);
	cudaFree(d_left_gray);
	cudaFree(d_right_gray);
	cudaFree(d_left_gray_zero_mean);
	cudaFree(d_right_gray_zero_mean);
	cudaFree(d_left_bgr);
	cudaFree(d_right_bgr);
	cudaFree(d_left_cost_vol);
	cudaFree(d_right_cost_vol);
	cudaFree(d_left_denoise);
	cudaFree(d_right_denoise);

	cudaDeviceReset();
}


// load png image to a float image in RGB order 8 bit
void loadPNG(float* img_ptr, float* R, float* G, float* B, std::string file_name, int* cols, int* rows)
{
        std::vector<unsigned char> tmp_img;

        unsigned int width;
        unsigned int height;
        unsigned error = lodepng::decode(tmp_img, width, height, file_name);

	// how to save img 
        //error = lodepng::encode("new1.png", tmp_img, width, height);

        for(unsigned int y=0; y<height; y++)
        {
                for(unsigned int x=0; x<width; x++)
                {
                        unsigned int idx = x+y*width;
                        img_ptr[idx] = (float)(tmp_img[idx*4]+tmp_img[idx*4+1]+tmp_img[idx*4+2])/3.0f;
                //      img_ptr[idx] = (float)(tmp_img[idx*4+1]);
                        R[idx] = (float)tmp_img[idx*4];
                        G[idx] = (float)tmp_img[idx*4+1];
                        B[idx] = (float)tmp_img[idx*4+2];

                }
        }

//	*cols = (int)width;
//	*rows = (int)height;
}


// save png disparity image
void savePNG(unsigned char* disp, std::string fileName, int cols, int rows)
{
        std::vector<unsigned char> tmp_img;

	int imgSize = cols*rows;

        for(int i=0; i<imgSize; i++)
        {
                tmp_img.push_back(disp[i]);
                tmp_img.push_back(disp[i]);
                tmp_img.push_back(disp[i]);
                tmp_img.push_back(255);
        }

        unsigned error = lodepng::encode(fileName, tmp_img, (unsigned int)cols, (unsigned int)rows);
}

// convert char image to float image and normalize to [0,1]
// if reverse is true, convert float to char
int imgCharToFloat(unsigned char* imgCharPtr, float* imgFloatPtr, bool reverse, unsigned int imgSize, float scale)
{
        if(!reverse)
        {
                for(int i=0; i<imgSize; i++)
                        imgFloatPtr[i] = (float)imgCharPtr[i];
        }
        else
        {
                for(int i=0; i<imgSize; i++)
                        imgCharPtr[i] = (unsigned char)(round(imgFloatPtr[i]*scale));
        }
        return 0;
}

void StartTimer()
{
        gettimeofday(&timerStart, NULL);
}

// time elapsed in ms
double GetTimer()
{
        struct timeval timerStop, timerElapsed;
        gettimeofday(&timerStop, NULL);
        timersub(&timerStop, &timerStart, &timerElapsed);
        return timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
}

void timingStat(double* time, int nt, double* average, double* sd)
{

        *average = 0.0;
	*sd = 0.0;
	
	if(nt < 2)
	{
		*average = time[0];
		return;
	}

        for(int i=1; i<=nt; i++)
                *average += time[i];

        *average /= (double)nt;

      
        for(int i=1; i<=nt; i++)
                *sd += pow(time[i] - *average, 2);

        *sd = sqrt(*sd/(double)(nt-1));

        return;
}

void StartTimer_GPU(cudaEvent_t* start, cudaEvent_t* stop)
{
	cudaEventCreate(start);
	cudaEventCreate(stop);

	cudaEventRecord(*start, 0);
}

float GetTimer_GPU(cudaEvent_t* start, cudaEvent_t* stop)
{
	cudaEventRecord(*stop, 0);
	float time;
	cudaEventElapsedTime(&time, *start, *stop);
	cudaEventDestroy(*start);
	cudaEventDestroy(*stop);
	return time;
}


