import jpype
from jpype import JInt, JArray, JString, JClass


class TTPWrapper:
    """
    This class is a caller of JAVA TTP class, aim to evaluate TTP solutions (tour & picking) in
    """
    def __init__(self,
                 caller_args,
                 jar_path='.'):
        """

        :param caller_args:
            Args to initialize TTPInstance(Java).
            The current sequence of parameters is
             * args[0]  folder with TTP files
             * args[1]  pattern to identify the TTP problems that should be solved
             * args[2]  optimisation approach chosen
             * args[3]  stopping criterion: number of evaluations without improvement
             * args[4]  stopping criterion: time in milliseconds (e.g., 60000 equals 1 minute)
            Setting to [] will initialize with default parameters.

        :param jar_path:
            TTPCaller(Java) root path.
        """
        jpype.startJVM(jpype.getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % jar_path)
        self.JavaCaller = JClass('TTPCaller')
        self.args = JArray(JString)(caller_args)
        info = self.JavaCaller.getTTPInfo(self.args)
        self.nodes, self.items = int(info[0]), int(info[1])

    def __call__(self, picking, tour):
        picking = JArray(JInt)(picking)
        tour = JArray(JInt)(tour)
        return self.JavaCaller.test(self.args, picking, tour)


if __name__ == "__main__":
    args = ["instances", "a280_n837_uncorr_08.ttp", "2", "10000", "60000"]
    problem = TTPWrapper(args)
