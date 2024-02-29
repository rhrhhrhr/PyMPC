import traceback

try:
    from .poly import PolyException, Polyhedron, Rn, UnitCube, support_fun
except PolyException:
    traceback.print_exc()
