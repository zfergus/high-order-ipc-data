import numpy

correct_order = numpy.array("""\
    -0.5      0.5      0.5
       0      0.5        0
     0.5      0.5      0.5
       0        0      0.5
    -0.5     -0.5      0.5
    -0.5        0        0
    -0.5     -0.5     -0.5
       0     -0.5        0
     0.5        0        0
     0.5     -0.5     -0.5
     0.5     -0.5      0.5
     0.5      0.5     -0.5
       0        0     -0.5
    -0.5      0.5     -0.5
-4.82963 0.660105        1
 4.82963 -1.92809        1
-4.82963 0.660105       -1
 4.82963 -1.92809       -1\
""".split()).reshape(-1, 3)

wrong_order = numpy.array("""\
     0.5     -0.5      0.5
       0        0      0.5
    -0.5     -0.5      0.5
     0.5      0.5      0.5
    -0.5      0.5      0.5
    -0.5        0        0
    -0.5     -0.5     -0.5
    -0.5      0.5     -0.5
       0        0     -0.5
     0.5      0.5     -0.5
     0.5     -0.5     -0.5
     0.5        0        0
       0      0.5        0
       0     -0.5        0
-4.82963 0.660105        1
 4.82963 -1.92809        1
-4.82963 0.660105       -1
 4.82963 -1.92809       -1\
""".split()).reshape(-1, 3)

assert(correct_order.shape == wrong_order.shape)

for line in wrong_order:
    indices = numpy.where(numpy.all(line == correct_order, axis=1))
    assert(len(indices) == 1)
    print(indices[0][0])
