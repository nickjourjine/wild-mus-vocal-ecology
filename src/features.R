# extract features from selection table

check_selection_table=function(path.to.selectiontable, mar){
  #make a selection table by moth and deployment
  
  
  #get packages
  require(warbleR)
  library(tools)
  
  #load the file
  st.csv <- read.csv(path.to.selectiontable)
  
  
 #check if there are predictions
  if (nrow(st.csv) != 0){

  	#directory where the wav files are 
  	wavs.dir <- st.csv$wavs.dir[1]
	cat(wavs.dir)
  	#check it with warbleR
  	st <- selection_table(X = st.csv, path = wavs.dir, pb = F, mar = mar)  
	
  	return(st)

  } else{
  	cat('no vocalizations in this file...\n')
	return(st.csv)
  }
}

get_acoustic_features=function(path.to.selectiontable, save.dir, wl, ovlp, bp, mar, ncores, annotation = F){

  #check the selection table
  st <- check_selection_table(path.to.selectiontable, mar)
  
  #check if there are vocalizations
  if (nrow(st) == 0){
  return()
  }
  
  #show pup info (wav file without the extension)
  moth <- st$moth[1]
  deployment <- st$deployment[1]
  box <- st$box[1]
  cat("\n", paste0(deployment,':',moth,':box',box, "\n"))

  #calculate features if you haven't already
  if (!file.exists(file.path(save.dir,paste(deployment,moth,box,'features.csv', sep='_')))){ 
  
      #get packages
      require(warbleR)

      #set spec options
      warbleR_options(wl=wl, ovlp=ovlp, bp=bp, mar=mar)

      #directory where the wav files are 
      wavs.dir <- st$wavs.dir[1]

      # get the features at all time windows
      cat("\t...calculating acoustic features.\n")
      sp <- spectro_analysis(st, harmonicity = F, fast = T, pb = T, path = wavs.dir, parallel = ncores)
      spfeatures <- merge(st,sp)

      ## get the features just between Q.25 and Q.75 for each voc - this takes a very very long time so I'm not doing it
      
      ##make a modified selection table
      #st.Q25Q75 <- spfeatures
      #st.Q25Q75$start <- st.Q25Q75$start + st.Q25Q75$time.Q25
      #st.Q25Q75$end <- st.Q25Q75$start + st.Q25Q75$time.Q75
      #columns_to_keep <- c('sound.files', 'selec', 'start', 'end', 'full.path', 'wavs.dir')
      #st.Q25Q75 <- st.Q25Q75[, columns_to_keep, drop = FALSE]

      ## rerun spectro_analysis
      #sp.Q25Q75 <- spectro_analysis(st.Q25Q75, harmonicity = F, fast = T, pb = T, path = wavs.dir)
      #spfeatures.Q25Q75 <- merge(st.Q25Q75,sp.Q25Q75)
      
      ##update names
      #columns_to_keep <- c('sound.files', 'selec', 'full.path', 'wavs.dir')
      #columns_to_modify <- setdiff(names(spfeatures.Q25Q75), columns_to_keep)
      #for (i in colnames(spfeatures.Q25Q75)) {
    
    		#if (i %in% columns_to_modify){
        #		colnames(spfeatures.Q25Q75)[colnames(spfeatures.Q25Q75) == i] <- paste0(i,'.Q25Q75')
    		#}
	    #}

      ##merge with spfeatures
      #spfeatures <- merge(spfeatures, spfeatures.Q25Q75, by = columns_to_keep)

      # get sound pressure levels
      cat("\t...getting sound pressure levels.\n")
      splfeatures <- sound_pressure_level(spfeatures, pb = T, path = wavs.dir, parallel = ncores)
      
      tryCatch({
	      error_condition <- FALSE
	      SNR <- sig2noise(splfeatures, pb = T, path = wavs.dir, type=1)
	      SNRfeatures <- merge(splfeatures,SNR)
      }, error = function(err){
              error_condition <<- TRUE
	      cat("\t\tAn error occurred: ", conditionMessage(err), "\n")
      })
      #if you can't calculate SNR, fill in with NA and continue
      if (error_condition) {
	      splfeatures$SNR <- NA
              SNRfeatures <- splfeatures
      }

      # get clipping
      cat("\t...getting signal clipping.\n")
      clipping <- find_clipping(SNRfeatures, pb=T, path = wavs.dir, parallel = ncores)
      all.features <- merge(SNRfeatures, clipping)

      # add columns with parameter info
      all.features$param.wl <- wl
      all.features$param.ovlp <- ovlp
      all.features$param.bp_low <- bp[1]
      all.features$param.bp_high <- bp[2]
      all.features$param.mar <- mar

      #save
      box.string <- paste0("box", box)
      cat('\tsaving...\n')
      if (!annotation) {
			write.csv(all.features, file.path(save.dir, paste(deployment, moth, box.string, 'features.csv', sep = '_')))
		} else {
			file <- gsub('.wav', '', st$sound.files[1])
			write.csv(all.features, file.path(save.dir, paste(file, 'features.csv', sep = '_')))
		}
      cat("\tdone.\n")
    } else {
      cat("\t...already processed.\n")
    }
}
