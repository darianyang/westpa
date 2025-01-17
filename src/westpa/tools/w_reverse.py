
import h5py
from tqdm.auto import tqdm
import os
import shutil
import argparse

from westpa.tools import (
    WESTTool,
    WESTDataReader,
    IterRangeSelection,
)

class W_Reverse(WESTTool):
    """
    w_reverse: a tool for taking a WE simulation facilitated 
    through WESTPA with successful recycling events and generating
    a new directory with the recycled restart files, which can then
    serve as the bstates for a subsequent WE simulation in the opposite
    direction, i.e. starting from the successfully recycled events and
    then going back to the original starting point; overall reversed.

    TODO:
        * option to use w_assign/assign.h5 output for successful bstate selection
        * adapt for WE simulation traj_segs that used the hdf5 framework
        * integrate into WESTPA, pull instance attributes and args from west.cfg
        * change printing to west logging
        * switch to WESTParallelTool?
    """

    def __init__(self, #h5="west.h5", first_iter=1, last_iter=None, 
                 #traj_segs="traj_segs", rst_name="seg.rst", 
                 max_n_bstates=10000,
                 output_bstates_dir="bstates_reverse", output_bstates_file="bstates.txt",
                 use_weights=True):
        """
        Parameters
        ----------
        h5 : str
            Path to west.h5 file
        first_iter : int
            By default start at iteration 1.
        last_iter : int
            Last iteration data to include, default is the last recorded iteration in the west.h5 file. 
        traj_segs : str
            Path to the traj_segs directory. Default current directory.
        rst_name : str
            Name of the restart file within each traj_segs/ subdirectory.
        max_n_bstates : int
            Max number of bstates to copy over. Adjust this if you prefer only the first
            n bstates found, default 10,000.
        output_bstates_dir : str
            Output directory for the bstates and output_bstates_file.
            Default './bstates_reverse'.'
        output_bstates_file : str
            Name of the output bstates file, default 'bstates.txt'.
        use_weights : bool
            By default, include the recycled event weight when making the bstates.txt file.
        """
        self.h5 = h5py.File(h5, mode="r")
        self.first_iter = int(first_iter)
        # default to last
        if last_iter is not None:
            self.last_iter = int(last_iter)
        elif last_iter is None:
            self.last_iter = self.h5.attrs["west_current_iteration"] - 1
        self.traj_segs = str(traj_segs)
        self.rst_name = str(rst_name)
        self.rst_extension = rst_name.rsplit('.', 1)[1]
        self.max_n_bstates = int(max_n_bstates)
        self.output_bstates_dir = str(output_bstates_dir)
        self.output_bstates_file = str(output_bstates_file)
        self.use_weights = use_weights

    def w_succ(self):
        """
        Find and return all successfully recycled (iter, seg) pairs.

        Returns
        -------
        succ : list of tuples (iter,wlk,weight)
        """
        succ = []

        for iter in tqdm(range(self.first_iter, self.last_iter + 1), 
                         desc="Running w_succ"):
            # if the new_weights group exists in the h5 file
            if f"iterations/iter_{iter:08d}/new_weights" in self.h5:
                prev_segs = self.h5[f"iterations/iter_{iter:08d}/new_weights/index"]["prev_seg_id"]
                recycled_weights = self.h5[f"iterations/iter_{iter:08d}/new_weights/index"]["weight"]
                # append the previous iter and previous seg id recycled and the weight
                for i in range(len(prev_segs)):
                    succ.append((iter-1, prev_segs[i], recycled_weights[i]))
        return succ

    @staticmethod
    def create_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")
        else:
            print(f"Directory '{directory}' already exists.")

    def w_reverse(self):
        """
        Main public method for running w_reverse.
        """
        # first generate list of successful events / iter,seg pairs and weights
        succ_pairs = self.w_succ()
        # make directory for bstates_reverse if it doesn't already exist
        self.create_dir(self.output_bstates_dir)

        # create bstates.txt file
        with open(f"{self.output_bstates_dir}/{self.output_bstates_file}", "w") as bstates_f:
    
            # control max number of reverse bstates created
            n_bstates = 0
            # different totals if the max is less than total succ_pairs to loop
            if self.max_n_bstates < len(succ_pairs):
                total_pairs = self.max_n_bstates
            else:
                total_pairs = len(succ_pairs)

            # then for each pair
            for idx, (it, wlk, weight) in tqdm(enumerate(succ_pairs), 
                                               total=total_pairs,
                                               desc="New bstates"):    
                if n_bstates < self.max_n_bstates:
                    # find the corresponding restart file
                    rst_file_path = f"{self.traj_segs}/{it:06d}/{wlk:06d}/{self.rst_name}"
                    rst_dest_name = f"{it:06d}_{wlk:06d}.{self.rst_extension}"
                    # if bstate file exists, skip
                    if os.path.exists(f"{self.output_bstates_dir}/{rst_dest_name}"):
                        continue
                    # copy to bstates_reverse directory (source, dest)
                    shutil.copyfile(rst_file_path, 
                                    f"{self.output_bstates_dir}/{rst_dest_name}")
                    # fill out the bstates.txt file with name and weight
                    # but only use weights if requested, otherwise use equal weights
                    # bstates.txt row format: bstate_n | weight | bstate_filename
                    if self.use_weights:
                        bstates_f.write(f"{idx} {weight:.3e} {rst_dest_name}\n") 
                    else:
                        bstates_f.write(f"{idx} 1 {rst_dest_name}\n") 
                    n_bstates += 1
                else:
                    break

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="w_reverse: a tool for generating bstates for a subsequent " + 
                    "steady-state WE simulation in the opposite direction."
    )
    parser.add_argument(
        "-W", "-w", "--west", "--west-data", "-h5", "--h5file",
        dest="h5",
        type=str,
        default="west.h5",
        help="Path to west.h5 file"
    )
    parser.add_argument(
        "--first-iter", "-fi",
        dest="first_iter",
        type=int,
        default=1,
        help="First iteration to consider (default: 1)"
    )
    parser.add_argument(
        "--last-iter", "-li",
        dest="last_iter",
        type=int,
        default=None,
        help="Last iteration to consider (default: last recorded iteration in west.h5)"
    )
    parser.add_argument(
        "--traj-segs",
        dest="traj_segs",
        type=str,
        default="traj_segs",
        help="Path to the traj_segs directory"
    )
    parser.add_argument(
        "--rst-name",
        dest="rst_name",
        type=str,
        default="seg.rst",
        help="Name of the restart file within each traj_segs/ subdirectory"
    )
    parser.add_argument(
        "--max-n-bstates", "-max", "-mnb",
        dest="max_n_bstates",
        type=int,
        default=10000,
        help="Max number of bstates to copy over. Adjust this if you prefer " +
             "a subset of the first bstates found. Default max of 10000."
    )
    parser.add_argument(
        "--output-bstates-dir", "-obd",
        dest="output_bstates_dir",
        type=str,
        default="bstates_reverse",
        help="Output directory for the bstates and output_bstates_file"
    )
    parser.add_argument(
        "--output-bstates-file", "-obf",
        dest="output_bstates_file",
        type=str,
        default="bstates.txt",
        help="Name of the output bstates file"
    )
    # TODO: may need to be adjusted to store False when included
    parser.add_argument(
        "--no-weights", "-nw",
        dest="use_weights",
        action="store_false",
        help="Don't include the recycled event weight when making the bstates.txt file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    reverse = W_Reverse(**vars(args))
    reverse.w_reverse()
