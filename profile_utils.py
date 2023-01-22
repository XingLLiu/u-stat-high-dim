import pstats
import io

sio = io.StringIO()
path = "profile.o"
s = pstats.Stats(path, stream=sio)
s.strip_dirs()
s.sort_stats("cumulative")
s.print_stats()
response = sio.getvalue()
sio.close()

text_file = open(path + "_formatted.txt", "w")
text_file.write(response)
text_file.close()
