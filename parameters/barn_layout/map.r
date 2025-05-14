setwd("/Users/nick_jourjine/Desktop/barn_map")
barriers=read.csv("newbarrierpos.csv")

plot(0,type="n",xlim=c(0,max(barriers$x)),ylim=c(max(barriers$y),0),xaxs="i",yaxs="i",
asp=-max(barriers$y)/max(barriers$x),xlab="",ylab="")
for(i in unique(barriers$name[barriers$type==1])){
	lines(barriers$x[barriers$name==i],barriers$y[barriers$name==i],lwd=4,col="blue")
}

for(i in unique(barriers$name[barriers$type==3])){
	lines(barriers$x[barriers$name==i],barriers$y[barriers$name==i],lwd=4,col="red")
}

for(i in unique(barriers$name[barriers$type==4])){
	lines(barriers$x[barriers$name==i],barriers$y[barriers$name==i],lwd=4,col="purple")
}

for(i in unique(barriers$name[barriers$type==2])){
	polygon(barriers$x[barriers$name==i],barriers$y[barriers$name==i],col="black")
}

boxes=read.csv("boxpos.csv")
boxes$x[boxes$box==39]=boxes$x[boxes$box==39]+0.1
points(y~x,data=boxes,pch=15,col="grey",cex=2.5)
text(boxes$x,boxes$y,boxes$box)
