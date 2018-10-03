from IPython.core.magic import  cell_magic, Magics, magics_class
execution_log = []
@magics_class
class CellLogger(Magics):
    @cell_magic
    def log(self, line='', cell=None):
        execution_log.append(cell)
        self.shell.ex(cell)
ip = get_ipython()
ip.register_magics(CellLogger)

def dump_history(file):
    for idx, cell in enumerate(execution_log):
        file.write("\n# Cell " + str(idx) + "\n")
        file.write(cell)