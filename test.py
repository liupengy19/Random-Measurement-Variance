import opt_einsum as oe
import cotengra as ctg

eq, shapes = oe.helpers.rand_equation(3, 1)
eq="iab,jbc,kcd,lda->ijkl"
shapes=[(256,16,16),(256,16,16),(256,16,16),(256,16,16)]
opt = ctg.HyperOptimizer()
path, info = oe.contract_path(eq, *shapes, shapes=True, optimize=opt)
print(info)