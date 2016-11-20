import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Generate the population distribution for a given district based on its political
# division in the 2012 election (latest data I found was in the google doc referenced at top). 
def gen_pop_distribution(district, scaled_ave_pop, percent_dem):

    # Get every 30th boundary point + form a district boundary
    contour = np.array([district.points[i] for i in xrange(0,len(district.points),10)],\
                       dtype=np.float32)
    
    n_people = 0

    affiliation = []
    coordinates = []
  
    # Bounding box of our district-contour
    min_x, min_y, max_x, max_y = district.bbox

    # Fill each district with N people 
    while( n_people < scaled_ave_pop ):
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        
        # Check if point generated in BB is in the district; if not, regenerate
        if( cv2.pointPolygonTest(contour,(x,y),False) >= 0 ):
      
            person = random.uniform(0,1) * 100
            
            # Is our new citizen dem or rep?
            if person < percent_dem:
                affiliation.append(1)
            else:
                affiliation.append(0)
            
            coordinates.append((x,y))
            n_people += 1

    return affiliation, coordinates

# First step: Copy functions written during DL course homework 7.
# Adjust to limit representation size per new district to the current average (scaled)
def findClosestCentroids(X,centroids,n_people):
    
    idx = []
    cluster_limits = np.zeros([len(centroids),1])
    
    for x in X:
        min_dist = 1.e10
        min_c_it = -1
        c_it = 0

        for c in centroids:
            
            if cluster_limits[c_it] >= n_people:
                c_it += 1
                continue
            sq_dist = np.sum( np.square(x - c) )
            
            if sq_dist < min_dist:
                min_dist = sq_dist
                min_c_it = c_it
            c_it += 1
        
        cluster_limits[min_c_it] += 1 
        
        idx.append(min_c_it)
    
    return idx

def computeCentroids(X,centroids,idx_v):
    
    new_cents = np.array([])
    
    for c in xrange(centroids.shape[0]):
   
        x_per_c = [ X[i] for i in xrange(len(idx_v)) if idx_v[i] == c ]
        
        sum_c = np.cumsum(x_per_c,axis=0)[-1] / len(x_per_c)
        new_cents = np.append(new_cents,sum_c)
      
    #print "Finishing up an iteration... ", new_cents.shape, centroids.shape[0], X.shape[1]
    new_cents = new_cents.reshape(centroids.shape[0],X.shape[1])
    
    return new_cents

def runkMeans(X,iterations,k,n_people=0):
    
    k_init = random.sample(range(0,X.shape[0]),k)
    centroids = np.array([X[i] for i in k_init])
    prev_centroids = centroids

    for i in xrange(iterations):

        idx_v = findClosestCentroids(X,centroids,n_people)
        centroids = computeCentroids(X,centroids,idx_v)  
        prev_centroids = np.append(prev_centroids,centroids,axis=0)
        
    return centroids, idx_v, prev_centroids



# Get coordinate info for your district + compile party affiliation info for plotting
# or further analysis
def grab_district_info(sf,df_cd,df_vote,state_districts,state_code,draw=False):

  tot_districts = 432
  n_people_per_dist = 720
  district_it = 0
  district_lines = sf.shapes()

  pop_coords = np.zeros([n_people_per_dist * state_districts,2])

  state_districts_v = []

  if draw:
    plt.figure(figsize=(11,7))

  for i in xrange(tot_districts):
      
      district_i = sf.records()[i]
      state_code_i = district_i[0]
      cd_code = district_i[1] # congressional district
      
      if int(state_code_i) != state_code: continue 

      state_districts_v.append(i)
      
      # Need to translate from state code in shape file to state name in 
      # nation_cd113 file; need this translation to build proper query name
      # for DKE file to get percent dem/rep for population
      state = df_cd[df_cd['STATEFP'] == int(state_code_i)]['STATE']
      state_name =  np.array(state)[0]
      
      DKE_id = str(state_name) + "-" + str(cd_code)
          
      percent_dem_row = df_vote[ df_vote['CD'] == str(DKE_id) ]['Obama 2012']
      percent_dem = np.array(percent_dem_row)[0]
      
      # Now that we have our party affiliation breakdown, fill the district!
      party_v, coord_v = gen_pop_distribution(district_lines[i],\
                                              n_people_per_dist, percent_dem)
      
      pop_coords[district_it*n_people_per_dist:(district_it+1)*n_people_per_dist] = np.array(coord_v)
      
      dem = np.array([coord_v[j] for j in xrange(len(party_v)) if party_v[j] == 1])
      rep = np.array([coord_v[j] for j in xrange(len(party_v)) if party_v[j] == 0])
      d = district_lines[i]
      state_lines = np.array(d.points).T

      if draw:
        plt.plot(dem[:,0],dem[:,1],'b.',ms=1,alpha=0.9)#,label='Democrats')
        plt.plot(rep[:,0],rep[:,1],'r.',ms=1,alpha=0.45)#,label='Republicans')   
        plt.plot(state_lines[0],state_lines[1],'k-',lw=0.5) #,label=DKE_id)
  
      # Keep track of how many districts we've filled for
      district_it += 1

  if draw:
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Congressional Districts in the Continental US")
    plt.show()

  return state_lines, pop_coords, state_districts_v

def plotKMeansData(X,sf,districts_v,centroids,idx_v):
    

    plt.figure(figsize=(11,7))

    for c in xrange(centroids.shape[0]):
        c_it = np.array([ X[i] for i in xrange(len(idx_v)) if idx_v[i] == c ])
        
	# If no entries, ignore
        if c_it.shape[0] == 0:  continue
	# If point at 0,0, ignore
	if c_it[0][0] == 0 and c_it[0][1] == 0: continue

        plt.plot(c_it[:,0],c_it[:,1],'.',ms=5,label="%i Clus"%c,alpha = 0.35)

    district_lines = sf.shapes()

    for i in districts_v:
        d = district_lines[i]
	pts = np.array(d.points).T
	plt.plot(pts[0],pts[1],'k-',lw=0.5)
    

    plt.grid(True)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Congressional Districts Redrawn")
    plt.show()

