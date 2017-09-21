/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  //Tunable parameter, number of particles:
	num_particles = 100;
	particles.resize( num_particles );
	weights.resize( num_particles );
	//Normal distribution (Gaussian)
	normal_distribution<double> dist_x( x, std[0] );
	normal_distribution<double> dist_y( y, std[1] );
	normal_distribution<double> dist_theta( theta, std[2] );
	//Engine used for generating particles
	random_device rd;
	default_random_engine generate(rd());
	//Iterate through particles, initializing them
	for (int i = 0; i < num_particles; ++i)
	{
		particles[i].id = i;
		particles[i].x = dist_x( generate );
		particles[i].y = dist_y( generate );
		particles[i].theta = dist_theta( generate );
		particles[i].weight = 1.0;
		//cout << "X: " << particles[i].x << endl << "Y: " << particles[i].y << endl << "Theta " << particles[i].theta << endl << endl;
	}
	//Set as initialized
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	//Normal distribution (Gaussian)
	normal_distribution<double> dist_x( 0, std_pos[0] );
	normal_distribution<double> dist_y( 0, std_pos[1] );
	normal_distribution<double> dist_theta( 0, std_pos[2] );
	
	default_random_engine generate;
	//Iterate through particles, checking whether yaw_rate is zero
	for ( int i = 0; i < num_particles; ++i )
	{
		if (abs( yaw_rate ) == 0)
		{
			particles[i].x += velocity * delta_t * cos( particles[i].theta );
			particles[i].y += velocity * delta_t * sin( particles[i].theta );
		} else
		{
			particles[i].x += ( velocity / yaw_rate ) * ( sin( particles[i].theta + ( yaw_rate * delta_t ) ) - sin( particles[i].theta ) );
			particles[i].y += ( velocity / yaw_rate ) * ( cos( particles[i].theta ) - cos( particles[i].theta + ( yaw_rate * delta_t ) ) );
			particles[i].theta += yaw_rate * delta_t;
    }
    particles[i].x += dist_x( generate );
    particles[i].y += dist_y( generate );
    particles[i].theta += dist_theta( generate );
  }
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	
	const double a = 1 / ( 2 * M_PI * std_landmark[0] * std_landmark[1] );
  // Run through each particle
  for ( int i = 0; i < num_particles; ++i ) {
    
    // Variable to hold multi variate gaussian distribution.
    double gaussDist = 1.0;
    
    // For each observation
    for ( int j = 0; j < observations.size(); ++j ) {
      
      // Vehicle to map coordinates
      double trans_obs_x = observations[j].x * cos( particles[i].theta ) - observations[j].y * sin( particles[i].theta ) + particles[i].x;
      double trans_obs_y = observations[j].x * sin( particles[i].theta ) + observations[j].y * cos( particles[i].theta ) + particles[i].y;
      
      // Find nearest landmark
      vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
      vector<double> landmark_obs_dist ( landmarks.size() );
      //Iterate through landmarks
      for ( int k = 0; k < landmarks.size(); ++k ) {
        
        //Check to see if particle landmark is within sensor range, and calculate distance between particle and landmark.
        double landmark_part_dist = sqrt( pow( particles[i].x - landmarks[k].x_f, 2 ) + pow( particles[i].y - landmarks[k].y_f, 2 ) );
        if (landmark_part_dist <= sensor_range) {
          landmark_obs_dist[k] = sqrt( pow( trans_obs_x - landmarks[k].x_f, 2 ) + pow( trans_obs_y - landmarks[k].y_f, 2 ) );

        } else {
          landmark_obs_dist[k] = 999999.0; //Fill with large number so algorithm doesn't think they are 0 and close.
          
        }
        
      }
      
      // Associate the observation point with its nearest landmark neighbor
      int min_pos = distance( landmark_obs_dist.begin(), min_element( landmark_obs_dist.begin(), landmark_obs_dist.end() ) );
      float nn_x = landmarks[min_pos].x_f;
      float nn_y = landmarks[min_pos].y_f;
      
      // Calculate multi-variate Gaussian distribution
      double x_diff = trans_obs_x - nn_x;
      double y_diff = trans_obs_y - nn_y;
      double b = ( ( x_diff * x_diff ) / ( 2 * std_landmark[0] * std_landmark[0] ) ) + ( ( y_diff * y_diff ) / ( 2 * std_landmark[1] * std_landmark[1] ) );
      gaussDist *= a * exp(-b);
      
    }
    
    // Update particle weights with combined multi-variate Gaussian distribution
    particles[i].weight = gaussDist;
    weights[i] = particles[i].weight;

  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	vector<Particle> new_particles ( num_particles );
  
  // Use discrete distribution to return particles by weight
  random_device rd;
  default_random_engine gen( rd() );
  for ( int i = 0; i < num_particles; ++i ) {
    discrete_distribution<int> index( weights.begin(), weights.end() );
    new_particles[i] = particles[index( gen )];
    
  }
  
  // Replace old particles with the resampled particles
  particles = new_particles;


}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
