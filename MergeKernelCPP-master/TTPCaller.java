
import java.io.*;
import ttp.Optimisation.Optimisation;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import ttp.TTPInstance;
import ttp.TTPSolution;
import ttp.Utils.DeepCopy;
import ttp.Utils.Utils;


/**
 *
 * @author zikaix
 */

 public class TTPCaller {
    /* The current sequence of parameters is
     * args[0]  folder with TTP files
     * args[1]  pattern to identify the TTP problems that should be solved
     * args[2]  optimisation approach chosen
     * args[3]  stopping criterion: number of evaluations without improvement
     * args[4]  stopping criterion: time in milliseconds (e.g., 60000 equals 1 minute)
     */
    public static void main(String[] args) {
        if (args.length == 0)
            args = new String[]{"instances", "fnl4461_n4460_bounded-strongly-corr_01.ttp", "2",
             "10000", "60000"};
        System.out.print(Arrays.toString(getTTPInfo(args)));
    }

    public static double evaluate(String[] args, int[] tour, int[] picking) {
        File[] files = ttp.Utils.Utils.getFileList(args);
        double ob;
        ob = 1.0;
        for (File f:files) {
            TTPInstance instance = new TTPInstance(f);
            TTPSolution solution = new TTPSolution(tour, picking);
            instance.evaluate(solution);
            ob = solution.ob;
        }
        System.out.println(ob);
        return ob;
    }

    public static int[] getTTPInfo(String[] args) {
        int[] ttpInfo = new int[2];
        File[] files = ttp.Utils.Utils.getFileList(args);
        for (File f:files) {
            TTPInstance instance = new TTPInstance(f);
            ttpInfo[0] = instance.numberOfNodes;
            ttpInfo[1] = instance.numberOfItems;
        } 
        return ttpInfo;
    }

    public static double test(String[] args, int[] picking, int[] tour) {
        if (args.length == 0)
            args = new String[]{"instances", "fnl4461_n4460_bounded-strongly-corr_01.ttp", "2",
             "10000", "60000"}; /// Default TTP instance

        if (tour[0] == 1)
            for (int i=0; i<tour.length; i++){
                tour[i] -= 1;
            }

        return evaluate(args, tour, picking);
    }
 }