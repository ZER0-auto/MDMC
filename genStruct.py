from clease.settings import Concentration
from clease.settings import CEBulk
from clease import NewStructures
import sys
from ase.build import bulk

gen = 10
db_name = "AlLifcc.db"
a=4.04
conc = Concentration(basis_elements=[['Al', 'Li']])
#conc.set_conc_ranges(ranges=[[(0.75, 1), (0, 0.25)]])

settings = CEBulk(crystalstructure='fcc',
                  a=a,
                  concentration=conc,
                  max_cluster_dia=[12, 7, 5],
                  db_name=db_name)

ns = NewStructures(settings, generation_number=gen, struct_per_gen=20)
template = bulk('Al', crystalstructure='fcc', a=4.04, cubic=True) * (2, 2, 2)
ns.generate_probe_structure(atoms=template, init_temp=1000, final_temp=5)
#ns.generate_random_structures(atoms=template)
#ns.generate_initial_pool(atoms=template)