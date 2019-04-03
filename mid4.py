from rx import Observable

o1 = Observable.interval(1000)
o2 = Observable.from_(["A","B","C"])

Observable.zip(o1,o2,lambda x,y:(x,y)) \
	.subscribe(lambda x: print(x))
