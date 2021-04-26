/*
 * KMeanAlgo_parallel_omp4.h
 *
 *  Created on: Nov 4, 2020
 *      Author: gratienj
 */

#pragma once

#include <stdlib.h>
#include <math.h>
#include <map>
#include "tbb/tbb.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;
using namespace tbb;

namespace PPTP
{

	class KMeanAlgoParTbb
	{
	public:
		//constructor
		KMeanAlgoParTbb(int nb_channels, int nb_centroids, int nb_iterations, int nb_threads)
	    : nb_channels(nb_channels)
		, m_nb_centroid(nb_centroids)
		, nb_iterations(nb_iterations)
		, nb_threads(nb_threads)
	    {
		centroids_occurences.resize(m_nb_centroid*nb_threads);
		updated_centroids.resize(m_nb_centroid*nb_channels*nb_threads);
		m_centroids.resize(m_nb_centroid*nb_channels);
	    }
		//destructor
		virtual ~KMeanAlgoParTbb() {}
		
		// compute the new value of the centroids
		bool compute_centroids_tbb(cv::Mat const& image)
		{
			using namespace cv ;
	
			bool centroids_convergence=true;
		      	
			/*****COMPUTE NEAREST CENTROID*****/
			compteur=0;
			
			centroids_occurences.clear();
			updated_centroids.clear();

			centroids_occurences.resize(m_nb_centroid*nb_threads);
			updated_centroids.resize(m_nb_centroid*nb_channels*nb_threads);
				
			
			switch(nb_channels)
			{
				// if there is only one color channel	
				case 1:
				{
					//TO PARALLELISE
					tbb::parallel_for(size_t(0), (size_t)nb_threads, [&](std::size_t id_thread){
						
						/*****SPLIT THE IMAGE PER THREADS****/
						int begin=0; // id of the first row for each thread
						int nrows_local = image.rows/nb_threads; // nb of rows to be handeled per thread
						int reste = image.rows%nb_threads;
						if((int)id_thread<reste)
						{
							nrows_local++;
							begin=id_thread*nrows_local;
						}
						else
						{
							begin=id_thread*nrows_local + reste;
						}
						// each thread deal with it own part of the image
						for(int i=begin;i<begin + nrows_local;i++)
						{
							
							int centroid_id=0;
							for(int j=0;j<image.cols;j++)
							{
								// CALCUL NEAREST CENTROID
							
								centroid_id = calc_closest_centroid(image, i, j); // find the centroid id associated with the current pixel
								auto pixel_coord = std::make_tuple(i,j); // get the pixel coordinate
								if(collection_centroids_ids[i*image.cols+j]!= centroid_id)  // if the pixel is already associated with this centroid we skip the update
								{
									compteur++;	
									centroids_convergence=false; // the convergence is not reached since there is a centroid update
									collection_centroids_ids[i*image.cols+j]=centroid_id; // we update the centroid associated to the current pixel
									collection_pixel[i*image.cols+j]=pixel_coord; // not used
								}	
								{
									centroids_occurences[centroid_id+id_thread*m_nb_centroid]++;  //increase the occurence of that centroid in the current thread
									updated_centroids[centroid_id+id_thread*m_nb_centroid]+=image.at<uchar>(i,j); // add the value of the current pixel associated with the current centroid (used to compute the mean i.E the new color value of the current centroid)
								}						
							}	
						}
					});
					break;
				}
				// same as above but for 3 channel
				case 3:
				{			
					Mat_<Vec3b> _I = image;
					//TO PARALLELISE
					tbb::parallel_for(size_t(0), (size_t)nb_threads, [&](std::size_t id_thread){
						int begin=0;
						int nrows_local = image.rows/nb_threads;
						int reste = image.rows%nb_threads;
						if((int)id_thread<reste)
						{
							nrows_local++;
							begin=id_thread*nrows_local;
						}
						else
						{
							begin=id_thread*nrows_local + reste;
						}	
						for(int i=begin;i<begin + nrows_local;i++)
						{
							//cout<<"id thread: "<<id_thread<<endl;
							int centroid_id=0;
							for(int j=0;j<image.cols;j++)
							{
								// CALCUL NEAREST CENTROID
						
								centroid_id = calc_closest_centroid(image, i, j);
								auto pixel_coord = std::make_tuple(i,j);
								if(collection_centroids_ids[i*image.cols+j]!= centroid_id)
								{
									compteur++;	
									centroids_convergence=false;
									collection_centroids_ids[i*image.cols+j]=centroid_id;
									collection_pixel[i*image.cols+j]=pixel_coord;
								}	
								{
									for(int k=0;k<3;k++)
									{
										updated_centroids[nb_channels*(centroid_id+id_thread*m_nb_centroid)+k]+=_I(i,j)[k];
									}	
									centroids_occurences[centroid_id + id_thread*m_nb_centroid]++;
								}						
							}	
						}
					});
					
					break;
				}
			}

			return centroids_convergence; // return the state of the convergence
		}
		
		// update the values of the centroids 
		void update_centroids_tbb(cv::Mat& image)
		{
			using namespace cv;
	
			std::vector<double> updated_centroids_reduced; //reduce the value of the centroids across the different threads
			std::vector<double> centroids_occurences_reduced; //reduce the occurence value of the centroids across the different threads
			
			centroids_occurences_reduced.resize(m_nb_centroid);
			updated_centroids_reduced.resize(m_nb_centroid*nb_channels);
			
			switch(nb_channels)
			{
				// if 1 color channel
				case 1:
				{
					// for each centroid
					for(int c=0; c<m_nb_centroid; c++)
					{
						// for each thread
						for(int id=0;id<nb_threads;id++)
						{
							centroids_occurences_reduced[c]+=centroids_occurences[c+id*m_nb_centroid]; // reduce (sumup) the occurence values of each centroid across each thread
							updated_centroids_reduced[c]+=updated_centroids[c+id*m_nb_centroid]; // reduce (sumup) the values of each centroid across each thread
						}
						if(centroids_occurences_reduced[c] !=0)
							updated_centroids_reduced[c]/=centroids_occurences_reduced[c]; // compute the mean of centroid to get the real value of each centroid
					}
					break;
				}
				// same as above but for 3 channels
				case 3:
				{
					Mat_<Vec3b> _I = image;
					
					for(int c=0; c<m_nb_centroid; c++)
					{	
						for(int id=0;id<nb_threads;id++)
						{
							centroids_occurences_reduced[c]+=centroids_occurences[c+id*m_nb_centroid];
							for(int k=0;k<nb_channels;k++){
								updated_centroids_reduced[nb_channels*c+k]+=updated_centroids[nb_channels*(c+id*m_nb_centroid)+k];
							}
						}
						if(centroids_occurences_reduced[c] !=0)
						{
							for(int k=0;k<nb_channels;k++){
							updated_centroids_reduced[c*nb_channels+k]/=centroids_occurences_reduced[c];
							}
						}	
					}
					break;
				}

			}
			m_centroids.resize(updated_centroids_reduced.size());

			
			for(unsigned int i=0; i<updated_centroids_reduced.size(); i++)
			{
				//cout<<" delta centroid: "<<(uchar)updated_centroids[i]-m_centroids[i]<<endl;
				m_centroids[i]=(uchar)updated_centroids_reduced[i]; // change the type to uchar
			}

 		}
		
		// segmente the image
		void compute_segmentation_tbb(cv::Mat& image)
		{
                      using namespace cv ;


		      switch(nb_channels)
		      {
		        case 1:
			  	//TO PARALLELISE
		          	tbb::parallel_for(std::size_t(0), (size_t)image.rows, [&](std::size_t i)	
				{
					int cluster_id =0;
		            		for(int j=0;j<image.cols;++j)
		           	 	{
						cluster_id=collection_centroids_ids[i*image.cols+j];
			      			image.at<uchar>(i,j)=m_centroids[cluster_id]; // each pixel get the color of it's own centroid
		            		}
		          	});
		          	break ;
		        
			case 3:
		       		Mat_<Vec3b> _I = image;
		          	//TO PARALLELISE
		          	tbb::parallel_for(std::size_t(0), (size_t)image.rows, [&](std::size_t i)	
				{
		            		
		      			int cluster_id =0;
					for(int j=0;j<image.cols;++j)
		            		{
		              	
						cluster_id=collection_centroids_ids[i*image.cols+j];
						for(int k=0;k<3;++k)
		              			{
							_I(i,j)[k]=m_centroids[3*cluster_id+k];	
		              			}
		            		}
		          	});
		          	break ;
		      }
		}
		
		// clustering algo
		void process_tbb(cv::Mat& image)
		{
			std::cout<<"Start process"<<std::endl;
			std::cout<<"Centroid_init"<<std::endl;
			init_centroid(image);
			
			std::cout<<"compute centroids"<<std::endl;
			int iterations=0;
			bool convergence_centroids=false;
			
			while(iterations<nb_iterations && convergence_centroids==false){
				std::cout<<"iteration number: "<<iterations<<std::endl;
				convergence_centroids=compute_centroids_tbb(image) ;
				//std::cout<<"compteur: "<<compteur<<endl;
				update_centroids_tbb(image);
				iterations++;
			}
			std::cout<<"compute segmentation"<<std::endl; 
			compute_segmentation_tbb(image) ;
		}

	private :

		// generate random centroids
		void init_centroid(Mat& image){
		      	collection_centroids_ids.resize(image.rows*image.cols);
			collection_pixel.resize(image.rows*image.cols);
			//INIT INITTIAL CENTROID

			int row_id=0;
			int col_id=0;
			switch(nb_channels)
			{
				case 1:
				{
					
					row_id=(int)(rand()%image.rows);
					col_id=(int)(rand()%image.cols);
					m_centroids[0]=image.at<uchar>(row_id,col_id);
					for(int i=1;i<(int)m_centroids.size();i++)
					{
						do{
						row_id=(int)(rand()%image.rows);
						col_id=(int)(rand()%image.cols);
						m_centroids[i]=image.at<uchar>(row_id,col_id);
						}while(abs(m_centroids[i]-m_centroids[i-1])>5 && std::find(m_centroids.begin(), m_centroids.end(), m_centroids[i]) != m_centroids.end());
					}
					break;
				}
				case 3: 
				{
						
					row_id=(int)(rand()%image.rows);
					col_id=(int)(rand()%image.cols);
					
					Mat_<Vec3b> _I = image;
					double dist=0;
					
					for(int k=0;k<3;k++){
						m_centroids[k]=_I(row_id,col_id)[k];
					}
					for(int c=1; c<m_nb_centroid; c++)
					{	
						do{
							row_id=(int)(rand()%image.rows);
							col_id=(int)(rand()%image.cols);
							dist=0;
							for(int i=3*c;i<3*c+3;i++)
							{
							
								m_centroids[i]=_I(row_id,col_id)[i];

								dist+=pow(_I(row_id,col_id)[i]-m_centroids[i-3],2);
							}
						}while(sqrt(dist)>5 && std::find(m_centroids.begin(), m_centroids.end(), m_centroids[c]) != m_centroids.end());
						
					}
					break;
				}
			}

		}
		
		// calc the closest centroid to a pixel according to its coordinates 
		int calc_closest_centroid(cv::Mat const& image, int i, int j){
		
			
			double min_dist=0;
			int id_centroid=0;
			
			switch(nb_channels)
			{
				case 1:
					{	
						uchar pixel = image.at<uchar>(i,j);
						min_dist=abs(pixel-m_centroids[0]);
						for (int c=1; c<m_nb_centroid; c++){
							if(min_dist>abs(pixel-m_centroids[c]))
							{
								min_dist=abs(pixel-m_centroids[c]);
								id_centroid=c;	
							}
						}
						break;
					}
				case 3:
					{
						Mat_<Vec3b> _I = image;	
						for(int k=0;k<3;k++){
							min_dist+=pow(_I(i,j)[k]-m_centroids[k],2);
						}
					
						min_dist=sqrt(min_dist);
						
						double clust_dist=0;	

						for(int c=1; c<m_nb_centroid;c++)
						{
							for(int k=3*c;k<3*c+3;k++)
							{
								clust_dist+=pow(_I(i,j)[k]-m_centroids[k],2);
							}
							clust_dist=sqrt(clust_dist);
							if(min_dist>clust_dist)
							{
								min_dist=clust_dist;
								id_centroid=c;
							}
						}
					break;
					}
			}
			return id_centroid;
		}

	    	int nb_channels  =3  ; // nb of color channels (1 or 3)
		int m_nb_centroid = 0 ; // nb of centroids
		int nb_iterations = 50; // nb of iterations
		int compteur =0; //used only for debug
		int nb_threads=1; // nb of threads 
		std::vector<uchar> m_centroids ; // centroid color values (size= nb_centroids * nb_channels)
		std::vector<std::tuple<int,int>> collection_pixel; //not used (should be removed)
		std::vector<int> collection_centroids_ids; // contain the centroid ids associated to each pixel (the vector is of the size of the image i.e flatened image)

		std::vector<double> updated_centroids;
		std::vector<double> centroids_occurences;
	};
}

