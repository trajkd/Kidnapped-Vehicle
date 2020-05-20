/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 20;  // Set the number of particles
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  for (int i = 0; i < num_particles; ++i) {
    double sample_x, sample_y, sample_theta;
    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);
    Particle p = {
      i,
      sample_x,
      sample_y,
      sample_theta,
      1.0
    };
    particles.push_back(p);
    weights.push_back(1.0);
  }
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  for (int i=0; i<num_particles; i++) {
    Particle p = particles[i];
    double x_p, y_p, theta_p;
    if (yaw_rate == 0.0) {
      x_p = p.x + velocity * delta_t * cos(p.theta);
      y_p = p.y + velocity * delta_t * sin(p.theta);
      theta_p = p.theta;
    } else {
      x_p = p.x + (velocity/yaw_rate) 
      * (sin(p.theta + (yaw_rate * delta_t)) - sin(p.theta));
      y_p = p.y + (velocity/yaw_rate) 
      * (cos(p.theta) - cos(p.theta + (yaw_rate * delta_t)));
      theta_p = p.theta + (yaw_rate * delta_t);
    }
    std::normal_distribution<double> dist_x(x_p, std_pos[0]);
    std::normal_distribution<double> dist_y(y_p, std_pos[1]);
    std::normal_distribution<double> dist_theta(theta_p, std_pos[2]);
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    particles[i] = p;
  } 
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (int i = 0; i < observations.size(); i++) {
    int closest_id = 0;
    double closest_distance = std::numeric_limits<double>::max();
    for (int j = 0; j < predicted.size(); j++) {
      double current_distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (current_distance < closest_distance) {
        closest_id = j;
        closest_distance = current_distance;
      }
    }
    observations[i].id = closest_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a multi-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (int i = 0; i < particles.size(); i++) {
    Particle p = particles[i];

    vector<LandmarkObs> observations_mapcoords;
    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs o = observations[j];

      double x_map = p.x + cos(p.theta) * o.x - sin(p.theta) * o.y;
      double y_map = p.y + sin(p.theta) * o.x + cos(p.theta) * o.y;

      observations_mapcoords.push_back({o.id, x_map, y_map});
    }
    
    std::vector<LandmarkObs> predicted_mapcoords;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      double distance = dist(p.x, p.y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
      if (distance <= sensor_range) {
        predicted_mapcoords.push_back({map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f});
      }
    }
    
    if (predicted_mapcoords.size() > 0) {
      dataAssociation(predicted_mapcoords, observations_mapcoords);
      double particle_weight = 0;
      for (int j = 0; j < observations_mapcoords.size(); j++) {
        double sigma_x = std_landmark[0];
        double sigma_y = std_landmark[1];
        double x = observations_mapcoords[j].x;
        double y = observations_mapcoords[j].y;
        double mu_x = predicted_mapcoords[observations_mapcoords[j].id].x;
        double mu_y = predicted_mapcoords[observations_mapcoords[j].id].y;
        double weight = 1 / (2 * M_PI * sigma_x * sigma_y) * exp(-((pow(x - mu_x, 2) / (2 * pow(sigma_x, 2))) + (pow(y - mu_y, 2) / (2 * pow(sigma_y, 2)))));
        if (particle_weight == 0) {
          particle_weight = weight;
        } else {
          particle_weight *= weight;
        }
      }
      particles[i].weight = particle_weight;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::vector<double> weights;
  for (int i=0; i<particles.size(); i++) {
    weights.push_back(particles[i].weight);
  }

  std::default_random_engine gen;
  std::discrete_distribution<int> distribution (weights.begin(), weights.end());

  std::vector<Particle> resampled_particles;

  for (int i=0; i<particles.size(); i++) {
    int n = distribution(gen);
    resampled_particles.push_back(particles[n]);
  }  
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}