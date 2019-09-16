library(reticulate)
source_python("topologizer_functions.py")

#Helper functions for plotting--replicates ggplot default qualitative color map
gg_color_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
hex_to_rgb = function(hex)
{
  require(stringr)
  rgb = matrix(0,nrow=length(hex),ncol=4)
  i = 0
  for (c in hex) {
    i = i + 1
    r_part = strtoi(str_sub(c,2,3),base=16)/255
    g_part = strtoi(str_sub(c,4,5),base=16)/255
    b_part = strtoi(str_sub(c,6,7),base=16)/255
    rgb[i,]=c(r_part,g_part,b_part,1)
  }
  return(rgb)
}

#' Computes the Mapper graph of datasets stored in a LIGER object.
#'
#' @param object liger object. Should have H.norm and tsne.coords.
#' @param path_html name of html output file
#' @param title title to display in html file
#' @param nr_cubes number of hypercubes to use in building Mapper graph
#' @param clust_thresh distance threshold for cutting hierarchical clustering dendrogram (higher = fewer clusters per hypercube)
#' @param overlap_perc percent overlap between adjacent hypercubes (higher = more connected Mapper graph)
#' @param cc_thresh Minimum connected component size in Mapper graph. Connected components with fewer nodes are filtered out.
#' @param color_by factor to use in coloring cells (e.g., cluster, time point, etc.)
#' @param node_size_func function to use in sizing Mapper graph nodes. Either none (radius is number of cells) or sqrt (square root of number of cells)
#' @return Mapper graph
run_topologizer = function(object,path_html,title="LIGER->Topologizer",nr_cubes=30,clust_thresh=0.8,overlap_perc=0.5,cc_thresh=3,color_by = NULL,node_size_func="none")
{
  km = import("kmapper")
  skl = import("sklearn")
  mapper = km$KeplerMapper(verbose=0)
  cat("Initializing filter\n")
  projected_data = mapper$fit_transform(object@H.norm, projection=DummyDR(object@tsne.coords),scaler=NULL)
  cat("Calculating Mapper graph\n")
  graph = mapper$map(projected_data, object@H.norm, nr_cubes=as.integer(nr_cubes),clusterer=AgglomerativeHierarchical(thresh=clust_thresh),overlap_perc=overlap_perc)
  graph = filter_mapper(graph,thresh=cc_thresh)
  if (is.null(color_by))
  {
    color_by=object@clusters
  }
  nbins=as.integer(length(levels(color_by)))
  bin_colors = hex_to_rgb(gg_color_hue(nbins))
  #add gray as an additional color
  #bin_colors = rbind(bin_colors,c(211/255,211/255,211/255,1))  
  set_km_color_map(bin_colors,as.integer(nbins))
  color_func = as.matrix(as.integer(color_by)-2,byrow=T,nrow=1)
  color_func[is.na(color_func)] = -1
  res = visualize(graph, color_function = color_func, path_html=path_html,title=title,node_size_func=node_size_func,nbins=nbins)  
  #res = mapper$visualize(graph, path_html=path_html,title=title)
  return(graph)
}

#' Reduces dimensionality for Mapper graph construction using diffusion maps followed by UMAP.
#'
#' @param object liger object. Should have H.norm.
#' @param num_dc number of diffusion components to use
#' @param n_neighbors number of neighbors to use in UMAP. Higher = more global structure preservation
#' @param min_dist minimum allowed distance among points in UMAP embedding. Higher = more local structure preservation.
#' @param k Number of UMAP dimensions.
#' @param distance metric to use in computing distance for UMAP
#' @return liger object with updated tsne.coords slot
run_dm_umap = function(object,num_dc=10,n_neighbors=20,min_dist=0.3,k=2, distance = 'euclidean')
{
  nmf_coords = data.frame(object@H.norm)
  cat("Running diffusion map\n")
  dm_res= run_diffusion_maps(nmf_coords, n_components=as.integer(num_dc))
  np = import("numpy")
  dm_res = np$array(dm_res$EigenVectors)
  UMAP<-import("umap")
  cat("Running UMAP\n")
  umapper = UMAP$UMAP(n_components=as.integer(k),metric = distance, n_neighbors = as.integer(n_neighbors),
                      min_dist = min_dist)
  Rumap = umapper$fit_transform
  object@tsne.coords = Rumap(dm_res)
  rownames(object@tsne.coords) = rownames(object@H.norm)
  return(object)
}

readNodes = function(filename="nodes.csv")
{
  test<-read.csv("nodes.csv",header=FALSE,sep=',')
  test = test[,c(3,4,27,28,29,30)]
  colnames(test)<-c('Name','Size','Index','Weight','X','Y')
  # Remove unnecessary part from values
  newname<-gsub(".*:","",test$Name)
  newsize<-gsub(".*:","",test$Size)
  newindex<-gsub(".*:","",test$Index)
  newweight<-gsub(".*:","",test$Weight)
  newx<-gsub(".*:","",test$X)
  newy<-gsub(".*:","",test$Y)

  # Create new dataframe in desired structure and format and output to csv file
  coords<-data.frame(index=newindex,name=newname,size=newsize,weight=newweight,x=newx,y=newy)
  coords_final<-subset(coords,index!='')
  write.csv(coords_final,"nodes_cleaned.csv",row.names = FALSE)  
}
