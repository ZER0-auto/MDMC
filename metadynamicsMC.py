from clease.calculator import attach_calculator
from clease.montecarlo.observers import ConcentrationObserver
from clease.montecarlo import SGCMonteCarlo, BinnedBiasPotential, MetaDynamicsSampler
import sys
import logging
from ase.build import bulk
from clease.settings import CEBulk

log_freq = 600
nbins = 251
mod_factors = range(6)
Tem = 500
n, alpha = int(sys.argv[1]), int(sys.argv[2])
settings = CEBulk(crystalstructure='fcc',
                  a=a,
                  concentration=conc,
                  max_cluster_dia=[12, 7, 5],
                  db_name=db_name)

logging.basicConfig(filename=f'meta{Tem}.log', level=logging.INFO)

ns = [12, 12, 12]
nn = 4 * ns[0] * ns[1] * ns[2]
eciName = f'{Tem}eci.json'
with open(eciName) as f:
    eci = json.load(f)

for mod_factor in mod_factors:
    atoms = bulk("Al", crystalstructure='fcc', a=a, cubic=True) * ns
    atoms = attach_calculator(settings, atoms=atoms, eci=eci)
    obs = ConcentrationObserver(atoms, element='Li')
    mc = SGCMonteCarlo(atoms, Tem, symbols=['Al', 'Li'])
    bias = BinnedBiasPotential(xmin=0.0, xmax=0.25, nbins=nbins, getter=obs)
    if mod_factor > 0:
        with open(f'AlLifccMetadyn{mod_factor - 1}_{n}_{alpha}.json', 'r') as f:
            data = json.load(f)
        bias.from_dict(data['bias_pot'])

    if mod_factor == 5:
        meta_dyn = MetaDynamicsSampler(mc=mc, bias=bias, flat_limit=0.95, mod_factor=10 ** -mod_factor,
                                       fname=f'{ase_script.getName()}Metadyn{mod_factor}_{n}_{alpha}.json')
    else:
        meta_dyn = MetaDynamicsSampler(mc=mc, bias=bias, flat_limit=0.9, mod_factor=10 ** -mod_factor,
                                       fname=f'{ase_script.getName()}Metadyn{mod_factor}_{n}_{alpha}.json')
    meta_dyn.log_freq = log_freq
    meta_dyn.run(max_sweeps=None)
    logging.getLogger(__name__).info('mod_factor: %s', mod_factor)
