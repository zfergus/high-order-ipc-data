def write_obj(filename, V, E=None, F=None):
    with open(filename, 'w') as f:
        for v in V:
            f.write("v {:.16f} {:.16f} {:.16f}\n".format(*v))
        if E is not None:
            for e in E:
                f.write("l {:d} {:d}\n".format(*(e + 1)))
        if F is not None:
            for face in F:
                f.write("f {:d} {:d} {:d}\n".format(*(face + 1)))
