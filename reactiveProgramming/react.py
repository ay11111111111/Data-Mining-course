from rx import Observable
import signal

def read_lines(fn):
    file = open(fn)
    return Observable.from_(file)


def words_in_file(fn):
    file = open(fn)
    return Observable.from_(file)\
                    .flat_map(lambda line: Observable.from_(line.split()))\
                    .map(lambda b: b.lower())\
                    .group_by(lambda word: word)\
                    .map(lambda grp: grp.count().map(lambda pair: (grp.key, pair)))\
                    .merge_all()\
                    .to_dict(lambda pair: pair[0], lambda pair: pair[1])


fn = "mc.txt"
def obs(fn):
    Observable.interval(3000) \
            .map(lambda i: words_in_file(fn)) \
            .merge_all()\
            .distinct_until_changed()\
            .subscribe(lambda value: print(value))

obs(fn)
signal.pause()

# input(" Press  any key\n")

# source = Observable.from_(['A', 'B', 'G', 'D', 'E'])

# source.subscribe(on_next = lambda value: print("Received {0}!".format(value)),\
#                 on_completed = lambda: print("done!"),
#                 on_error=lambda error:print("Error Occurred: {0}".format(error))
#                 )
#
# Observable.from_ (["one two", "three  four", "five  six "]) \
#                 .take (3) \
#                 .flat_map(lambda s: s.split ()) \
#                 .subscribe(lambda  value: print ("{0}". format(value )))


#
# Observable.from_ (["A", "B", "C", "D"]) \
#     .to_list() \
#     .subscribe(lambda i: print(i))
