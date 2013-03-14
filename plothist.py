from __future__ import print_function, division; __metaclass__ = type
import logging
import re, os
from itertools import izip
from westtools.tool_classes import WESTTool
import numpy, h5py
from westtools import h5io, textio
from matplotlib import pyplot
from fasthist import normhistnd

log = logging.getLogger('westtools.plothist')

# Suppress divide-by-zero in log
numpy.seterr(divide='ignore', invalid='ignore')


def sum_except_along(array, axes):
    '''Reduce the given array by addition over all axes except those listed in the scalar or 
    iterable ``axes``.'''
    
    try:
        iter(axes)
    except TypeError:
        axes = [axes]
    
    output_array = array.copy()
    for axis in xrange(array.ndim):
        if axis not in axes:
            output_array = numpy.add.reduce(output_array, axis=axis)
    return output_array

class PlotHistTool(WESTTool):
    prog='plothist'
    description = '''\
Plot probability density functions (histograms)
generated by w_pcpdist or other programs conforming to the same output format.
Instantaneous, average, or evolution of PDFs may be plotted. HDF5 files of the
properly constructed histograms (i.e. what is plotted) may be produced
(--hdf5-output). For 1-D histograms, text files may also be produced
(--text-output). Plotting may be supressed (to generate only HDF5/text files)
by passing --plot-output=''.
'''
    
    def __init__(self):
        super(PlotHistTool,self).__init__()
        
        self.input_h5 = None
        self.opmode = None
        self.plotmode = None
        self.plotrange = None
        self.plottitle = None
        
        # Iteration range for average/evolution
        self.iter_start = None
        self.iter_stop = None
        self.iter_step = None
        
        # Iteration for single point
        self.n_iter = None
        
        # An array of dicts describing what dimensions to work with and
        # what their ranges should be for the plots.
        self.dimensions = []
        
        self.plot_output_filename = None
        self.text_output_filename = None
        self.hdf5_output_filename = None
        
    def _add_plot_options(self, parser):
        pgroup = parser.add_argument_group('plot options')
        pmgroup = pgroup.add_mutually_exclusive_group()
        pgroup.add_argument('--title', dest='title',
                             help='Include TITLE as the top-of-graph title') 
        pmgroup.add_argument('--linear', dest='plotmode', action='store_const', const='linear',
                             help='Plot the PDF on a linear scale.')
        pmgroup.add_argument('--energy', dest='plotmode', action='store_const', const='energy',
                             help='Plot the PDF on an inverted natural log scale, corresponding to (free) energy (default).')
        pmgroup.add_argument('--log10', dest='plotmode', action='store_const', const='log10',
                             help='Plot the PDF on a base-10 log scale.')
        pgroup.add_argument('--range',
                            help='''Plot PMF over the given RANGE, specified as LB,UB, where 
                            LB and UB are the lower and upper bounds, respectively. For 1-D plots,
                            this is the Y axis. For 2-D plots, this is the colorbar axis.
                            (Default: full range.)''')
    
    def add_args(self, parser):
        subparsers = parser.add_subparsers(help='plotting modes')
        
        iparser = subparsers.add_parser('instant', help='Plot histogram for a given iteration')
        aparser = subparsers.add_parser('average', help='Plot histogram averaged over a number of iterations')
        eparser = subparsers.add_parser('evolution', help='Plot evolution of histogram over iterations')
        
        iparser.set_defaults(opmode='instant')
        aparser.set_defaults(opmode='average')
        eparser.set_defaults(opmode='evolution')
        
        for subparser in (iparser, aparser, eparser):
            igroup = subparser.add_argument_group('input & calculation options')
            igroup.add_argument('input',
                                help='''HDF5 file containing histogram data''')

            # 2-D plots only permissible for non-evolution plots            
            igroup.add_argument('firstdim', nargs='?', metavar='DIMENSION',
                                help='''Plot for the given DIMENSION, specified as INT[:[LB,UB]:LABEL], where
                                INT is a zero-based integer identifying the dimension in the histogram, 
                                LB and UB are lower and upper bounds for plotting, and LABEL is the label for
                                the plot axis. (Default: dimension 0, full range.)''')
            if subparser is not eparser:
                igroup.add_argument('seconddim', nargs='?', metavar='ADDTLDIM',
                                    help='''For instantaneous/average plots, plot along the given additional
                                    dimension, producing a color map.''')
            
            # start/stop/step not useful for instantaneous plot
            if subparser is iparser:
                igroup.add_argument('--iter', metavar='N_ITER', dest='n_iter',
                                    help='''Plot distribution for iteration N_ITER
                                    (default: last completed iteration).''')
            elif subparser is aparser:
                igroup.add_argument('--first-iter', dest='first_iter', type=int, metavar='N_ITER', default=1,
                                    help='''Begin averaging at iteration N_ITER (default: %(default)d).''')
                igroup.add_argument('--last-iter', dest='last_iter', type=int, metavar='N_ITER',
                                    help='''Conclude averaging with N_ITER, inclusive (default: last completed iteration).''')                            
            else:
                assert subparser is eparser
                igroup.add_argument('--first-iter', dest='first_iter', type=int, metavar='N_ITER', default=1,
                                    help='''Begin analysis at iteration N_ITER (default: %(default)d).''')
                igroup.add_argument('--last-iter', dest='last_iter', type=int, metavar='N_ITER',
                                    help='''Conclude analysis with N_ITER, inclusive (default: last completed iteration).''')            
                igroup.add_argument('--step-iter', dest='step_iter', type=int, metavar='STEP',
                                    help='''Average in blocks of STEP iterations.''')
            
        
            # Add plot and output options
            self._add_plot_options(subparser)
            
            ogroup = subparser.add_argument_group('output options')
            ogroup.add_argument('-o', '--plot-output', default='hist.pdf',
                                help='''Store plot as PLOT_OUTPUT. This may be set to an empty string
                                (--plot-output='') to suppress plotting entirely (see --text-output
                                and --hdf5-output). The output format is determined by filename 
                                extension (and thus defaults to PDF). Default: %(default)s.''')
            ogroup.add_argument('--hdf5-output',
                                help='''Store plot data in HDF5 format at HDF5_OUTPUT (default: no
                                HDF5 output).''')
            if subparser is not eparser:
                ogroup.add_argument('--text-output',
                                    help='''Store plot data in a text format at TEXT_OUTPUT. This option is
                                    only valid for 1-D histograms. (Default: no text output.)''')
        
        parser.set_defaults(plotmode='energy')

    def parse_dimspec(self, dimspec):
        dimdata = {}
        match = re.match(r'([0-9]+)(?::(?:([^,]+),([^:,]+))?(?::(.*))?)?', dimspec)
        if not match:
            raise ValueError('invalid dimension specification {!r}'.format(dimspec))
        
        (idim_txt, lb_txt, ub_txt, label) = match.groups()
        try:
            dimdata['idim'] = int(idim_txt)
            if lb_txt:
                dimdata['lb'] = float(lb_txt)
            if ub_txt:
                dimdata['ub'] = float(ub_txt)
            if label:
                dimdata['label'] = label
            else:
                dimdata['label'] = 'dimension {}'.format(dimdata['idim'])
        except ValueError as e:
            raise ValueError('invalid dimension specification {!r}: {!r}'.format(dimspec, e))
        return dimdata
        
    def parse_range(self, rangespec):
        try:
            (lbt,ubt) = rangespec.split(',')
            return float(lbt), float(ubt)
        except (ValueError,TypeError) as e:
            raise ValueError('invalid range specification {!r}: {!r}'.format(rangespec, e))
        
    def process_args(self, args):
        self.opmode = args.opmode
        self.plotmode = args.plotmode        
        self.input_h5 = h5py.File(args.input, 'r')
        self.plot_output_filename = args.plot_output
        
        if self.opmode != 'evolution':
            self.text_output_filename = args.text_output
            
        self.hdf5_output_filename = args.hdf5_output
        
        if args.title:
            self.plottitle = args.title
        
        if args.range:
            self.plotrange = self.parse_range(args.range)
        
        if args.firstdim:
            self.dimensions.append(self.parse_dimspec(args.firstdim))
            
            if self.opmode in ('instant', 'average') and args.seconddim is not None:
                self.dimensions.append(self.parse_dimspec(args.seconddim))
        else:
            self.dimensions.append({'idim': 0, 'label':'dimension 0'})

        
        avail_iter_start, avail_iter_stop = h5io.get_iter_range(self.input_h5['histograms'])
        try:
            avail_iter_step = h5io.get_iter_step(self.input_h5['histograms']) 
        except KeyError:
            avail_iter_step = 1
        log.info('HDF5 file {!r} contains data for iterations {} -- {} with a step of {}'.format(args.input,
                                                                                                 avail_iter_start, avail_iter_stop,
                                                                                                 avail_iter_step))
        
        if self.opmode == 'evolution':    
            if args.first_iter:
                self.iter_start = max(args.first_iter, avail_iter_start)
            else:
                self.iter_start = avail_iter_start
                
            if args.last_iter:
                self.iter_stop = min(args.last_iter+1, avail_iter_stop)
            else:
                self.iter_stop = avail_iter_stop
                
            if args.step_iter:
                self.iter_step = max(args.step_iter, avail_iter_step)
            else:
                self.iter_step = avail_iter_step
            log.info('using data for iterations {} -- {} with a step of {}'.format(self.iter_start, self.iter_stop, self.iter_step))
        elif self.opmode == 'average':
            if args.first_iter:
                self.iter_start = max(args.first_iter, avail_iter_start)
            else:
                self.iter_start = avail_iter_start
                
            if args.last_iter:
                self.iter_stop = min(args.last_iter+1, avail_iter_stop)
            else:
                self.iter_stop = avail_iter_stop            
        else:
            if args.n_iter:
                self.n_iter = min(args.n_iter, avail_iter_stop-1)        
            else:
                self.n_iter = avail_iter_stop - 1        
    
    def go(self):
        if self.opmode == 'average':
            if len(self.dimensions) == 2:
                self.do_average_plot_2d()
            else:
                self.do_average_plot_1d()
        elif self.opmode == 'instant':
            if len(self.dimensions) == 2:
                self.do_instant_plot_2d()
            else:
                self.do_instant_plot_1d()
        else:
            assert self.opmode == 'evolution'
            self.do_evolution_plot()
            
    def _do_1d_output(self, hist, idim, midpoints):
        enehist = -numpy.log(hist)
        log10hist = numpy.log10(hist)
        
        if self.hdf5_output_filename:
            with h5py.File(self.hdf5_output_filename, 'w') as output_h5:
                h5io.stamp_creator_data(output_h5)
                output_h5.attrs['source_data'] = os.path.abspath(self.input_h5.filename)
                output_h5.attrs['source_dimension'] = idim
                output_h5['midpoints'] = midpoints
                output_h5['histogram'] = hist
                
        if self.text_output_filename:
            with textio.NumericTextOutputFormatter(self.text_output_filename) as output_file:
                output_file.write_header('source data: {} dimension {}'.format(os.path.abspath(self.input_h5.filename),idim))
                output_file.write_header('column 0: midpoint of bin')
                output_file.write_header('column 1: probability in bin')
                output_file.write_header('column 2: -ln P')
                output_file.write_header('column 3: log10 P')
                numpy.savetxt(output_file, numpy.column_stack([midpoints,hist, enehist, log10hist]))
        
        if self.plot_output_filename:
            if self.plotmode == 'energy':
                plothist = enehist
                label = r'$\Delta F(x)\,/\,kT$' +'\n' + r'$\left[-\ln\,P(x)\right]$'
            elif self.plotmode == 'log10':
                plothist = log10hist
                label = r'$\log_{10}\ P(x)$'
            else:
                plothist = hist
                label = r'$P(x)$'
            pyplot.figure()
            pyplot.plot(midpoints, plothist)
            pyplot.xlim(self.dimensions[0].get('lb'), self.dimensions[0].get('ub'))
            if self.plotrange:
                pyplot.ylim(*self.plotrange)
            pyplot.xlabel(self.dimensions[0]['label'])
            pyplot.ylabel(label)
            if self.plottitle:
                pyplot.title(self.plottitle)
            pyplot.savefig(self.plot_output_filename)

            
    def do_instant_plot_1d(self):
        '''Plot the histogram for iteration self.n_iter'''
        
        idim = self.dimensions[0]['idim']
        n_iters = self.input_h5['n_iter'][...]
        iiter = numpy.searchsorted(n_iters, self.n_iter)
        binbounds = self.input_h5['binbounds_{}'.format(idim)][...]
        midpoints = self.input_h5['midpoints_{}'.format(idim)][...]
        hist = self.input_h5['histograms'][iiter]
        
        # Average over other dimensions
        hist = sum_except_along(hist, idim)
        normhistnd(hist, [binbounds])
        self._do_1d_output(hist, idim, midpoints)
            
    def do_average_plot_1d(self):
        '''Plot the average histogram for iterations self.iter_start to self.iter_stop'''
        
        idim = self.dimensions[0]['idim']
        n_iters = self.input_h5['n_iter'][...]
        iiter_start = numpy.searchsorted(n_iters, self.iter_start)
        iiter_stop  = numpy.searchsorted(n_iters, self.iter_stop)
        binbounds = self.input_h5['binbounds_{}'.format(idim)][...]
        midpoints = self.input_h5['midpoints_{}'.format(idim)][...]
        hist = self.input_h5['histograms'][iiter_start:iiter_stop]
        
        # Average over time
        hist = numpy.add.reduce(hist, axis=0)
        
        # Average over other dimensions
        hist = sum_except_along(hist, idim)

        normhistnd(hist, [binbounds])
        self._do_1d_output(hist, idim, midpoints)
            
    def do_instant_plot_2d(self):
        '''Plot the histogram for iteration self.n_iter'''
        
        idim0 = self.dimensions[0]['idim']
        idim1 = self.dimensions[1]['idim']
        
        n_iters = self.input_h5['n_iter'][...]
        iiter = numpy.searchsorted(n_iters, self.n_iter)
        binbounds_0 = self.input_h5['binbounds_{}'.format(idim0)][...]
        midpoints_0 = self.input_h5['midpoints_{}'.format(idim0)][...]
        binbounds_1 = self.input_h5['binbounds_{}'.format(idim1)][...]
        midpoints_1 = self.input_h5['midpoints_{}'.format(idim1)][...]
        
        hist = self.input_h5['histograms'][iiter]
        
        # Average over other dimensions
        hist = sum_except_along(hist, [idim0,idim1])
        #for dim in xrange(hist.ndim):
        #    if dim not in (idim0,idim1):
        #        hist = numpy.add.reduce(hist, axis=dim)

        if idim0 > idim1:
            hist = hist.T
                
        normhistnd(hist, [binbounds_0,binbounds_1])
        self._do_2d_output(hist, [idim0,idim1], [midpoints_0,midpoints_1])

    def do_average_plot_2d(self):
        '''Plot the histogram for iteration self.n_iter'''
        
        idim0 = self.dimensions[0]['idim']
        idim1 = self.dimensions[1]['idim']
        
        n_iters = self.input_h5['n_iter'][...]
        iiter_start = numpy.searchsorted(n_iters, self.iter_start)
        iiter_stop  = numpy.searchsorted(n_iters, self.iter_stop)

        binbounds_0 = self.input_h5['binbounds_{}'.format(idim0)][...]
        midpoints_0 = self.input_h5['midpoints_{}'.format(idim0)][...]
        binbounds_1 = self.input_h5['binbounds_{}'.format(idim1)][...]
        midpoints_1 = self.input_h5['midpoints_{}'.format(idim1)][...]
        
        hist = self.input_h5['histograms'][iiter_start:iiter_stop]
        
        # Average over time
        hist = numpy.add.reduce(hist, axis=0)        
        
        # Average over other dimensions
        for dim in xrange(hist.ndim):
            if dim not in (idim0,idim1):
                hist = numpy.add.reduce(hist, axis=dim)

        if idim0 > idim1:
            hist = hist.T
                
        normhistnd(hist, [binbounds_0,binbounds_1])
        self._do_2d_output(hist, [idim0,idim1], [midpoints_0,midpoints_1])

        
    def _do_2d_output(self, hist, idims, midpoints):
        enehist = -numpy.log(hist)
        log10hist = numpy.log10(hist)
        
        if self.hdf5_output_filename:
            with h5py.File(self.hdf5_output_filename, 'w') as output_h5:
                h5io.stamp_creator_data(output_h5)
                output_h5.attrs['source_data'] = os.path.abspath(self.input_h5.filename)
                output_h5.attrs['source_dimensions'] = numpy.array(idims, numpy.min_scalar_type(max(idims)))
                output_h5.attrs['source_dimension_labels'] = numpy.array([dim['label'] for dim in self.dimensions])
                for idim in idims:
                    output_h5['midpoints_{}'.format(idim)] = midpoints
                output_h5['histogram'] = hist

                        
        if self.plot_output_filename:
            if self.plotmode == 'energy':
                plothist = enehist
                label = r'$\Delta F(x)\,/\,kT$' +'\n' + r'$\left[-\ln\,P(x)\right]$'
            elif self.plotmode == 'log10':
                plothist = log10hist
                label = r'$\log_{10}\ P(x)$'
            else:
                plothist = hist
                label = r'$P(x)$'
            
            try:
                vmin, vmax = self.plotrange
            except TypeError:
                vmin, vmax = None, None
                
            pyplot.figure()
            # Transpose input so that axis 0 is displayed as x and axis 1 is displayed as y
            pyplot.imshow(plothist.T, interpolation='nearest', aspect='auto',
                          extent=(midpoints[0][0], midpoints[0][-1], midpoints[1][0], midpoints[1][-1]),
                          origin='lower', vmin=vmin, vmax=vmax)
            cb = pyplot.colorbar()
            cb.set_label(label)
            pyplot.xlabel(self.dimensions[0]['label'])
            pyplot.xlim(self.dimensions[0].get('lb'), self.dimensions[0].get('ub'))
            pyplot.ylabel(self.dimensions[1]['label'])
            pyplot.ylim(self.dimensions[1].get('lb'), self.dimensions[1].get('ub'))
            if self.plottitle:
                pyplot.title(self.plottitle)
            pyplot.savefig(self.plot_output_filename)        
        
    def do_evolution_plot(self):
        '''Plot the evolution of the histogram for iterations self.iter_start to self.iter_stop'''
        
        idim = self.dimensions[0]['idim']
        n_iters = self.input_h5['n_iter'][...]
        iiter_start = numpy.searchsorted(n_iters, self.iter_start)
        iiter_stop  = numpy.searchsorted(n_iters, self.iter_stop)
        binbounds = self.input_h5['binbounds_{}'.format(idim)][...]
        midpoints = self.input_h5['midpoints_{}'.format(idim)][...]
        hists = self.input_h5['histograms'][...]
        
        itercount = self.iter_stop - self.iter_start
        
        # We always round down, so that we don't have a dangling partial block at the end
        nblocks = itercount // self.iter_step
            
        block_iters = numpy.empty((nblocks,2), dtype=n_iters.dtype)
        blocked_hists = numpy.zeros((nblocks,hists.shape[1]), dtype=hists.dtype) 
        
        for iblock, istart in enumerate(xrange(iiter_start, iiter_start+nblocks*self.iter_step, self.iter_step)):
            istop = min(istart+self.iter_step, iiter_stop)
            histslice = hists[istart:istop]
            
            # Sum over time
            histslice = numpy.add.reduce(histslice, axis=0)
            
            # Sum over other dimensions
            blocked_hists[iblock] = sum_except_along(histslice, idim)
            
            # Normalize
            normhistnd(blocked_hists[iblock], [binbounds])
            
            block_iters[iblock,0] = n_iters[istart]
            block_iters[iblock,1] = n_iters[istop-1]+1
                        
        enehists = -numpy.log(blocked_hists)
        log10hists = numpy.log10(blocked_hists)
        
        
        if self.hdf5_output_filename:
            with h5py.File(self.hdf5_output_filename, 'w') as output_h5:
                h5io.stamp_creator_data(output_h5)
                output_h5.attrs['source_data'] = os.path.abspath(self.input_h5.filename)
                output_h5.attrs['source_dimension'] = idim
                output_h5['midpoints'] = midpoints
                output_h5['histograms'] = blocked_hists
                output_h5['n_iter'] = block_iters
                        
        if self.plot_output_filename:
            if self.plotmode == 'energy':
                plothist = enehists
                label = r'$\Delta F(x)\,/\,kT$' +'\n' + r'$\left[-\ln\,P(x)\right]$'
            elif self.plotmode == 'log10':
                plothist = log10hists
                label = r'$\log_{10}\ P(x)$'
            else:
                plothist = blocked_hists
                label = r'$P(x)$'
            
            try:
                vmin, vmax = self.plotrange
            except TypeError:
                vmin, vmax = None, None
                
            pyplot.figure()
            pyplot.imshow(plothist, interpolation='nearest', aspect='auto',
                          extent=(midpoints[0], midpoints[-1], block_iters[0,0], block_iters[-1,1]),
                          origin='lower', vmin=vmin, vmax=vmax)
            cb = pyplot.colorbar()
            cb.set_label(label)
            pyplot.xlabel(self.dimensions[0]['label'])
            pyplot.xlim(self.dimensions[0].get('lb'), self.dimensions[0].get('ub'))
            pyplot.savefig(self.plot_output_filename)
        
    
#axins.imshow(loghist.T, interpolation='nearest', extent=(midpoints_dist[0], midpoints_dist[-1], midpoints_rmsd[0], midpoints_rmsd[-1]), origin='lower', vmin=0, vmax=10, aspect='auto')
if __name__ == '__main__':
    PlotHistTool().main()
    