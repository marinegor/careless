import reciprocalspaceship as rs
import gemmi
from careless.utils.io import load_isomorphous_mtzs
from os.path import abspath,dirname

path = dirname(abspath(__file__)) + "/../merge/"

filenames = [
    "pyp_off.mtz",
    "pyp_2ms.mtz",
]
filenames = [path + i for i in filenames]

def test_load_isomorphous_mtzs():
    length = 0
    for inFN in filenames:
        length += len(rs.read_mtz(inFN))
    ds = load_isomorphous_mtzs(*filenames)
    assert isinstance(ds, rs.DataSet)
    assert length == len(ds)
    assert 'file_id' in ds
    assert isinstance(ds.spacegroup, gemmi.SpaceGroup)
    assert isinstance(ds.cell, gemmi.UnitCell)

