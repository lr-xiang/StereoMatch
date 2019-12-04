#include "Stereo3DMST.h"
#include <iostream>

#include <sys/time.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <omp.h>

using namespace std;

struct timeval timer_start;

void startTimer() { gettimeofday(&timer_start, NULL); }

// time elapsed in ms
double getTimer()
{
        struct timeval timerStop, timerElapsed;
        gettimeofday(&timerStop, NULL);
        timersub(&timerStop, &timer_start, &timerElapsed);
        return timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
}

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
#include "boost/graph/iteration_macros.hpp"

// setS as edge list enforces no parallel edges, takes extra time to look up though
typedef boost::adjacency_list < boost::setS, boost::vecS, boost::undirectedS> tree_graph_t;
typedef tree_graph_t::vertex_descriptor tree_vertex_descriptor;
typedef tree_graph_t::edge_descriptor tree_edge_descriptor;
typedef std::pair<int, int> tree_edge;

struct EdgeWeight
{
	double weight;
};

struct VertexProperties
{
	int parent_idx;
	int num_children = 0;
	int children_indices[4];
	double weight;	//between self and parent
	double weight2;
};

typedef boost::adjacency_list < boost::vecS, boost::vecS, boost::undirectedS, VertexProperties/*boost::no_property*/, EdgeWeight> mst_graph_t;
typedef mst_graph_t::vertex_descriptor mst_vertex_descriptor;
typedef mst_graph_t::edge_descriptor mst_edge_descriptor;
typedef std::pair<int, int> mst_edge;


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

void aggregateCostFromChildren(mst_graph_t& mst, double* cur_agg_cost, std::vector<int>& vertices_vec,
				float* cost_vol, abc* abc_map, abc& test_label, const int max_disp, const int width, const int img_size) {
			
	const int size = boost::num_vertices(mst);
	
	for(int node_id = size-1; node_id>0; node_id--)	{
	
		int & parent_id = mst[node_id].parent_idx;
		int & pixel_id = vertices_vec[node_id];
		int & parent_pixel_id = vertices_vec[parent_id];	

		//self
		cur_agg_cost[pixel_id] += compute3DLabelCost(cost_vol, test_label, pixel_id, max_disp, width, img_size);

		cur_agg_cost[parent_pixel_id] += mst[node_id].weight*cur_agg_cost[pixel_id];
	}

	cur_agg_cost[vertices_vec[0]] += compute3DLabelCost(cost_vol, test_label, vertices_vec[0], max_disp, width, img_size);
}


void aggregateCostFromParent(double* agg_cost, mst_graph_t& mst, std::vector<int>& vertices_vec)
{
	const int size = boost::num_vertices(mst);
	
	for(int node_id = 0; node_id<size; node_id++) {
	
		int & node_pixel_id = vertices_vec[node_id];

		for(int i=0; i<mst[node_id].num_children; i++)	{
		
			int & child_id = mst[node_id].children_indices[i];
		
			int & child_pixel_id = vertices_vec[child_id];

			agg_cost[child_pixel_id] = mst[child_id].weight*agg_cost[node_pixel_id] + mst[child_id].weight2*agg_cost[child_pixel_id];
		}
	}
}

void MSTCostAggregationAndLabelUpdate(double* min_cost, double* agg_cost, mst_graph_t& mst, abc* abc_map, abc& test_label, 
				      std::vector<int>& vertices_vec, float* cost_vol, float* disp_u,
				      const int max_disp, const int width, const int height, const int img_size)
{
	// zero agg_cost in tree due to OpenMP
	for(auto& pixel_idx : vertices_vec) agg_cost[pixel_idx] = 0.0;

	// leaf to root cost aggregation
	aggregateCostFromChildren(mst, agg_cost, vertices_vec, cost_vol, abc_map, test_label, max_disp, width, img_size);

	aggregateCostFromParent(agg_cost, mst, vertices_vec);	

	// update 3d labels
	for(auto& pixel_idx : vertices_vec) {
	
		double & cost = agg_cost[pixel_idx];

		if( cost < min_cost[pixel_idx] ) {
		
			min_cost[pixel_idx] = cost;

			abc_map[pixel_idx].a = test_label.a; 
			abc_map[pixel_idx].b = test_label.b; 
			abc_map[pixel_idx].c = test_label.c;
		}			
	}
}


void LabelToDisp(abc* abc_map, std::vector<mst_graph_t>& mst_vec, std::vector<std::vector<int>>& mst_vertices_vec, cv::Mat& disp, const int height, const int width, const int max_disp)
{
//	startTimer();
#pragma omp parallel for
	for(int i=0; i<height*width; i++)
	{
		abc* abc_ptr = abc_map + i;
			
            disp.ptr<float>(0)[i] = MAX(0.0f, min(1.0f, ( (i % width)*abc_ptr->a + (i / width)*abc_ptr->b + abc_ptr->c )/(max_disp-1.0f) ) );
		//disp.ptr<float>(0)[i] = ( (i % width)*abc_ptr->a + (i / width)*abc_ptr->b + abc_ptr->c );			
	}
//	std::cout<<"label to disp time "<<getTimer()<<"\n";
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
		              tree_graph_t& tree_g, abc* abc_map, float c, int min_size, const int max_disp, float gamma, const int filter_size=3) 
{
	const int width = r.cols;
	const int height = r.rows;
	const int img_size = width*height;

	mst_vec.clear();
	mst_vertices_vec.clear();

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
	min_size = std::max(2, min_size);
	
	for (int i = 0; i < num; i++) {
	
		int a = u->find(edges[i].a);
		int b = u->find(edges[i].b);
		
		if ((a != b) && ((u->size(a) < min_size) || (u->size(b) < min_size))) {
		
			u->join(a, b);
			u->find(a);
			u->find(b);
			mst_edge_mask[i] = 1; //there are components of size 1
		}
	}
#endif


#if 1
	cv::Mat out;
	out.create(height, width, CV_8UC3);
	// pick random colors for each component
	rgb *colors = new rgb[img_size];
	for (int i = 0; i < img_size; i++) colors[i] = random_rgb();
	
	for (int y = 0; y < height; y++) {
	
		for (int x = 0; x < width; x++) {
		
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
	//startTimer();
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
			cc_ids[i] = std::distance(rep_id_vec.begin(), it);
		else {
			cc_ids[i] = rep_id_vec.size();
			rep_id_vec.push_back(u->find(i));
		}

		// partition pixel vertices
		id_in_tree[i] = mst_vertices_vec[ cc_ids[i] ].size();	

		mst_vertices_vec[ cc_ids[i] ].push_back(i);
		
		//std::cout<<cc_ids[i]<<" ";
	}

	//std::cout<<"vector time: "<<getTimer()<<"\n";
	//std::cout<<"id vector size: "<<rep_id_vec.size()<<"\n";

	//build tree adjacency list
	//startTimer();

	for(int i = 0; i < num; i++)
	{
		const int a = u->find(edges[i].a); 
		const int b = u->find(edges[i].b);

		//edge connects 2 components
		if (a != b) boost::add_edge(cc_ids[a], cc_ids[b], tree_g);
	}

	//std::cout<<"build tree graph time: "<<getTimer()<<"\n";
	
	//unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	//std::default_random_engine generator(seed);
	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
	auto dice = std::bind (distribution, generator);

	//startTimer();
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

	//startTimer();
	// build each MST
	mst_vec.resize( u->num_sets() );

	for(int i = 0; i < num; i++)
	{
		if(mst_edge_mask[i] == 1)	//MST edge
		{
			const int tree_id = cc_ids[edges[i].b];

			mst_edge_descriptor e = boost::add_edge(id_in_tree[edges[i].a], id_in_tree[edges[i].b], mst_vec[tree_id]).first;

			mst_vec[tree_id][e].weight = exp(-edges[i].w*gamma);	
		}
	}
	

	// BFS
	for(int t=0; t<mst_vec.size(); t++)
	{
		std::vector<int> new_mst_vertices(boost::num_vertices(mst_vec[t]));
		
		new_mst_vertices[0] = mst_vertices_vec[t][0]; 
		
		mst_graph_t new_mst(boost::num_vertices(mst_vec[t]));
		
		std::queue<int> new_vertices_queue;
		
		new_vertices_queue.push(0);
		
		std::queue<int> vertices_queue;

		//root node
		vertices_queue.push(0);

		mst_vec[t][0].parent_idx = 0;
		
		new_mst[0].parent_idx = 0;//new
		
		int node_id = 0;//new

		std::vector<uchar> color(boost::num_vertices(mst_vec[t]), 0); // white color

		color[0] = 1;	//gray color
	
		while( !vertices_queue.empty() )
		{
			// parent
			const int p = vertices_queue.front();	

			vertices_queue.pop();	
			
			const int new_p = new_vertices_queue.front();
			
			new_vertices_queue.pop();

			boost::graph_traits<mst_graph_t>::adjacency_iterator ai, a_end;		

			boost::tie(ai, a_end) = boost::adjacent_vertices(p, mst_vec[t]); 

			for (; ai != a_end; ++ai) 	
			{
				if(color[*ai] == 0) //white
				{
					color[*ai] = 1;	//gray
					
					vertices_queue.push(*ai);	
					
					new_vertices_queue.push(++node_id);
					
					boost::add_edge(new_p, node_id, new_mst);
					
					new_mst_vertices[node_id] = mst_vertices_vec[t][*ai];
					
					// edge between parent and child
					mst_edge_descriptor e = boost::edge(p, *ai, mst_vec[t]).first;
					
					new_mst[node_id].parent_idx = new_p;
					
					new_mst[node_id].weight = mst_vec[t][e].weight;	
										
					new_mst[node_id].weight2 = 1.0f - new_mst[node_id].weight*new_mst[node_id].weight;
					
					new_mst[new_p].children_indices[new_mst[new_p].num_children++] = node_id;
				}
			}
		}	
		
		mst_vec[t] = new_mst;
		mst_vertices_vec[t] = new_mst_vertices;	
	}


	//std::cout<<"build MSTs time: "<<getTimer()<<"\n";



#if 0
	MyData my_data;
	my_data.width = width;
	my_data.height = height;
	my_data.mst_vec = &mst_vec;
	my_data.mst_vertices_vec = &mst_vertices_vec;

	cv::setMouseCallback("segment", ShowWeights, (void*)&my_data);
	cv::waitKey(0);
#endif

	delete[] mst_edge_mask;
	delete [] edges;
	delete u;
}


void MST_PMS(std::vector<mst_graph_t>& mst_vec, std::vector<std::vector<int>>& mst_vertices_vec, /*std::vector<std::vector<int>>& bfs_order_vec,*/
	     tree_graph_t& tree_g, abc* abc_map, double* min_cost, double* agg_cost, float* cost_vol, float* disp_u, 
	     const int max_disp, const int width, const int height, 
	     const int img_size, std::default_random_engine& generator, 
	     std::uniform_real_distribution<float>& distribution)
{
	//startTimer();

	auto dice = std::bind (distribution, generator);

	//std::mutex mutex;

	// go through each tree
#pragma omp parallel for schedule(dynamic)
	for(int tree_id = 0; tree_id < mst_vec.size(); tree_id++)
	{
		// SPATIAL PROPAGATION
		boost::graph_traits<tree_graph_t>::adjacency_iterator ai, a_end;
		boost::tie(ai, a_end) = boost::adjacent_vertices(tree_id, tree_g);

		//std::cout<<"random sample label from each neighbor tree\n";
		for (; ai != a_end; ++ai) 
		{
			const int test_pixel_idx = mst_vertices_vec[*ai][(int)((dice()+1.0f)*0.5f*mst_vertices_vec[*ai].size())];
			
			abc test_label;
			//mutex.lock();
			test_label.a = abc_map[test_pixel_idx].a; test_label.b = abc_map[test_pixel_idx].b; test_label.c = abc_map[test_pixel_idx].c;
			//mutex.unlock();

			// test label and update
			MSTCostAggregationAndLabelUpdate(min_cost, agg_cost, mst_vec[tree_id], abc_map, test_label, 
							 mst_vertices_vec[tree_id], cost_vol, disp_u, max_disp, width, 
							 height, img_size);
		}

		// RANDOM REFINEMENT
		//std::cout<<"pick a random node in the tree\n";
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

			MSTCostAggregationAndLabelUpdate(min_cost, agg_cost, mst_vec[tree_id], abc_map, test_label, 
							 mst_vertices_vec[tree_id], cost_vol, disp_u, max_disp, 
							 width, height, img_size);
		}			
	}

	//std::cout<<"time: "<<getTimer()<<"\n";
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



void stereo3dmst(std::string left_name, std::string right_name, cv::Mat & leftImg, cv::Mat & rightImg, cv::Mat & leftDisp, 
		cv::Mat & rightDisp, std::string data_cost="MCCNN_acrt", int Dmax=100) {
	
		
	const int cols = leftImg.cols;
	const int rows = leftImg.rows;
	unsigned int imgSize = (unsigned int)cols*rows;
	
	leftDisp.create(rows, cols, CV_32F);
	rightDisp.create(rows, cols, CV_32F);
	
	if(data_cost == "MCCNN_fst") {
	
		if( chdir("mc-cnn-master") < 0 ) {
		
			std::cout<<"no mc-cnn-master folder\n";
			return;
		}

		std::string cmd = "./main.lua mb fast -a predict -net_fname net/net_mb_fast_-a_train_all.t7 -left ../" + left_name 
				+ " -right ../"+ right_name + " -disp_max " + std::to_string(Dmax)+ " -sm_terminate cnn";
	
		if( system(cmd.c_str()) < 0)
			return;

		if( chdir("..") < 0 )
			return;	
	}
	else if(data_cost == "MCCNN_acrt") {
	
		if( chdir("mc-cnn-master") < 0 )
			return;

		std::string cmd = "./main.lua mb slow -a predict -net_fname net/net_mb_slow_-a_train_all.t7 -left ../" + left_name 
				+ " -right ../"+ right_name + " -disp_max " + std::to_string(Dmax)+ " -sm_terminate cnn";
	
		if( system(cmd.c_str()) < 0)
			return;

		if( chdir("..") < 0 )
			return;	
	}
	else {
		std::cout<<"wrong data cost\n";
		return;
	}
	
	//startTimer();
	
	//load mc-cnn raw cost volume, origional range (-1,1)
	int fd;
	float *left_cost_vol_mccnn, *right_cost_vol_mccnn;

	if(data_cost == "MCCNN_fst" || data_cost == "MCCNN_acrt")
	{
		fd = open("mc-cnn-master/left.bin", O_RDONLY);
		left_cost_vol_mccnn = (float*)mmap(NULL, 1 * Dmax * rows * cols * sizeof(float), PROT_READ, MAP_SHARED, fd, 0);
		close(fd);
		fd = open("mc-cnn-master/right.bin", O_RDONLY);
		right_cost_vol_mccnn = (float*)mmap(NULL, 1 * Dmax * rows * cols * sizeof(float), PROT_READ, MAP_SHARED, fd, 0);
		close(fd);
	}
	

	float *left_cost_vol_mccnn_w = new float[imgSize*Dmax];
	float *right_cost_vol_mccnn_w = new float[imgSize*Dmax];

	std::memcpy(left_cost_vol_mccnn_w, left_cost_vol_mccnn, imgSize*Dmax*sizeof(float));
	std::memcpy(right_cost_vol_mccnn_w, right_cost_vol_mccnn, imgSize*Dmax*sizeof(float));


#pragma omp parallel for
	for(int i=0; i<imgSize*Dmax; i++) 
	{
        if(std::isnan(left_cost_vol_mccnn_w[i]))
			left_cost_vol_mccnn_w[i] = 0.5f;
		else
			left_cost_vol_mccnn_w[i] = min(0.5f, left_cost_vol_mccnn_w[i]);		//accurate (0,1)
//			left_cost_vol_mccnn_w[i] = min(0.5f, (left_cost_vol_mccnn_w[i]+1.0f)*0.5f);	//fast, (-1,1)
	}

#pragma omp parallel for
	for(int i=0; i<imgSize*Dmax; i++) 
	{
		if(isnan(right_cost_vol_mccnn_w[i])) 
			right_cost_vol_mccnn_w[i] = 0.5f;
		else
			right_cost_vol_mccnn_w[i] = min(0.5f, right_cost_vol_mccnn_w[i]);
//			right_cost_vol_mccnn_w[i] = min(0.5f, (right_cost_vol_mccnn_w[i]+1.0f)*0.5f);
	}
	
	std::vector<mst_graph_t> left_mst_vec;
	std::vector<mst_graph_t> right_mst_vec;

	std::vector<std::vector<int>> left_mst_vertices_vec;
	std::vector<std::vector<int>> right_mst_vertices_vec;

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

	cv::Mat left_disp_u, right_disp_u;
	left_disp_u.create(rows, cols, CV_32F);
	right_disp_u.create(rows, cols, CV_32F);
    const float gamma = 1.0f/12.f;
    const float c = 5000.0f;
	const int min_cc_size = 200;
	
	// split channels
	std::vector<cv::Mat> cvLeftBGR_v;
	std::vector<cv::Mat> cvRightBGR_v;

	cv::split(leftImg, cvLeftBGR_v);
	cv::split(rightImg, cvRightBGR_v);

	segment_image_other_init(cvLeftBGR_v[2], cvLeftBGR_v[1], cvLeftBGR_v[0], 
				 left_mst_vec, left_mst_vertices_vec, left_tree_g, 
				 left_abc_map, c, min_cc_size, Dmax, gamma); 
				 
	segment_image_other_init(cvRightBGR_v[2], cvRightBGR_v[1], cvRightBGR_v[0],
				 right_mst_vec, right_mst_vertices_vec, right_tree_g, 
				 right_abc_map, c, min_cc_size, Dmax, gamma); 
				 


	std::default_random_engine generator;
	std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

	const int num_iter = 100;
	
	//startTimer();
	
	for(int i=0; i<num_iter; i++)
	{
//		std::cout<<"iter: "<<i+1<<"\n";
	
		MST_PMS(left_mst_vec, left_mst_vertices_vec, left_tree_g, left_abc_map, left_min_cost.ptr<double>(0), 
			left_agg_cost.ptr<double>(0), left_cost_vol_mccnn_w, left_disp_u.ptr<float>(0),
			Dmax, cols, rows, imgSize, generator, distribution);
#if 1
		//if(i%4 == 0 || i == num_iter-1)
		{
			LabelToDisp(left_abc_map, left_mst_vec, left_mst_vertices_vec, leftDisp, rows, cols, Dmax);	
			cv::imshow("left a", leftDisp); cv::waitKey(50);
		}
#endif
	}

	for(int i=0; i<num_iter; i++)
	{
	//	std::cout<<"iter: "<<i+1<<"\n";
	
		MST_PMS(right_mst_vec, right_mst_vertices_vec, right_tree_g, right_abc_map, right_min_cost.ptr<double>(0), 
			right_agg_cost.ptr<double>(0), right_cost_vol_mccnn_w, right_disp_u.ptr<float>(0),
			Dmax, cols, rows, imgSize, generator, distribution);

#if 1
		//if(i%4 == 0 || i == num_iter-1)
		{
			LabelToDisp(right_abc_map, right_mst_vec, right_mst_vertices_vec, rightDisp, rows, cols, Dmax);	
			cv::imshow("right a", rightDisp); cv::waitKey(50);
		}
#endif
	}
	
	//std::cout<<"time: "<<getTimer()<<"\n";

#if 0
	LabelToDisp(left_abc_map, left_mst_vec, left_mst_vertices_vec, leftDisp, rows, cols, Dmax);	
	LabelToDisp(right_abc_map, right_mst_vec, right_mst_vertices_vec, rightDisp, rows, cols, Dmax);	
	cv::imshow("right a", rightDisp); 
	cv::imshow("left a", leftDisp); cv::waitKey(0);
#endif

	leftDisp *= (Dmax-1.f);
	
	rightDisp *= (Dmax-1.f); 
	
	leftRightConsistencyCheck(leftDisp.ptr<float>(0), rightDisp.ptr<float>(0), cols, rows, Dmax, false);

	delete[] left_cost_vol_mccnn_w;
	delete[] right_cost_vol_mccnn_w;
	delete[] left_abc_map;
	delete[] right_abc_map;
	
	//std::cout<<"time: "<<getTimer()<<"\n";
}
