import numpy as np
import astropy.io.fits as fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import sunpy.cm as cm
from path_utils import click_path, select_path, select_area_from_path, select_image
from scipy.optimize import curve_fit
import cv2
import pickle

plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['image.origin'] = 'lower'
plt.rcParams['image.cmap'] = 'viridis'

class analysis:
    """
    Class to make time ditance plots of a coronal loop, chosen by clicking a superimposed image of coronal loop data from iris spacecraft.
    """
    def __init__(self, filename, wavelength, date, thefilepath):
        self.filename = filename
        self.wavelength = wavelength
        self.irissjicolor = plt.get_cmap('irissji'+self.wavelength)
        self.date = date
        self.thefilepath = thefilepath

    def superimpose(self, start, stop, plot=True, vmin=0, vmax=1500, path=True):
        """
        Shows a superimposed image of data within a time interval, and though click_path makes a path along chosen coronal loop.
        ---------
        start:  time interval start
        stop:   time interval end
        """

        self.sji = fits.open(self.thefilepath+'/'+self.filename)

        length = stop-start
        self.timesteps = np.linspace(start, stop, length, dtype=int)
        self.data = self.sji[0].data[self.timesteps]
        self.sum_data = np.sum(self.data, axis=0)   #superimposed data

        if plot:
            plt.imshow(self.sum_data, vmin=vmin, vmax=vmax, cmap=self.irissjicolor)
            if path:
                pos = click_path(ecolor='Blue', thick=1, psym=8)
                self.xx = np.array(pos.xs)
                self.yy = np.array(pos.ys)

                coords = self.data, self.xx, self.yy, self.sum_data, self.timesteps    #pickle file of data created, used in use_previous
                filepath = self.thefilepath+'/prev_superimpose.pkl'
                with open(filepath,'wb') as fo:
                    pickle.dump(coords,fo)
                    fo.close()

    def make_cut(self, nr_transv=301, eps=15, vmin=0, vmax=30, plot=True):
        """
        Takes data made in superimpose and makes a 3D array of every time step on the coronal loop maked. Plots 9 time distance plots along the loop to find optimal spot.
        -------
        nr_transv:  width of path along loop
        """

        self.eps=eps
        x,y,s = select_path(self.xx,self.yy,spline='spline')
        nr_path = len(s)
        self.x_area,self.y_area = select_area_from_path(x,y,s,nr_transv=nr_transv)
        self.cut = np.zeros([self.data.shape[0], nr_path, nr_transv])
        for i in range(0, self.data.shape[0]):
            self.cut[i,:,:]=select_image(self.data[i,:,:], self.x_area, self.y_area)

        coords = self.cut, self.eps, self.x_area, self.y_area   #pickle file of data created, used in use_previous
        filepath = self.thefilepath+'/prev_make_cut.pkl'
        with open(filepath,'wb') as fo:
            pickle.dump(coords,fo)
            fo.close()

        if plot:
            h_ops = []
            step=(self.x_area.shape[0])/9
            for i in range(9):
                if i ==0:
                    h_ops.append(15)
                else:
                    h_ops.append(int((step)*i))

            fig,axs = plt.subplots(3,3,figsize=(20,7))
            fig.tight_layout()

            count = 0
            for i in range(3):
                for j in range(3):
                    axs[i][j].imshow(self.cut[:,h_ops[count]-self.eps:h_ops[count]+self.eps,:].mean(1).T, vmin=vmin, vmax=vmax, cmap=self.irissjicolor, aspect=0.4)
                    axs[i][j].set_title('h=%i'%h_ops[count])
                    count+=1

    def wiggles(self, h, path=True, vmin=0, vmax=30):
        """
        Makes wiggles through click_path at heigh of the loop created in make_cut.
        -------
        h:  height of cut in loop
        """

        self.h = h
        self.title_h = 'h='+str(self.h)

        if path:
            plt.imshow(self.cut[:,self.h-self.eps:self.h+self.eps,:].mean(1).T, vmin=vmin, vmax=vmax, cmap=self.irissjicolor, aspect=0.4)
            plt.axis('off')

            self.img_name = 'time_dist_'+self.wavelength+'_'+self.title_h+'.png'
            wiggle_nr = input('how manny wiggles? (0 to break) ')


            pos_x = []
            pos_y = []

            for i in range(int(wiggle_nr)):
                pos_c = click_path(ecolor='Blue', thick=1, psym=8)
                pos_x.append(np.array(pos_c.xs))
                pos_y.append(np.array(pos_c.ys))

            self.all_pos= np.array(list(zip(pos_x,pos_y)))

            coords = self.all_pos, self.h   #pickle file of data created, used in use_previous
            filepath = self.thefilepath+'/prev_wiggles.pkl'
            with open(filepath,'wb') as fo:
                pickle.dump(coords,fo)
            fo.close()

    def curve_fit_sin(self, T, detrend=False):
        """
        Does a curve fit of the oscillations from wiggles using scipy.optimize.curve_fit.
        --------
        T:  period of each oscillation, given as a list of all the values.
        """

        self.best_params = np.zeros((self.all_pos.shape[0], 3))
        self.funcs =[]
        for i in range(self.all_pos.shape[0]):
            print(i)
            xx = self.all_pos[i,0]
            yy = self.all_pos[i,1]
            a = (max(yy)-min(yy))/2
            b = np.pi/T[i]
            c = 0
            d = max(yy)-a

            if detrend:
                a_l = (yy[-1]-yy[0])/(xx[-1]-xx[0]) #Consider changing!!!
                b_l = d
                def f(x, a, b, c, d):
                    return a*np.sin(b*(x-c))+d
                def f_l(x, a, b):
                    return a*x+b
                self.popt, self.pcov = curve_fit(f, xx, yy, p0=[a,b,c,d], maxfev=5000)
                popt_l, pcov_l = curve_fit(f_l, xx, yy, p0=[a_l,b_l], maxfev=5000)
                self.func = (self.popt[0]*np.sin(self.popt[1]*(xx-self.popt[2]))+self.popt[3])-((popt_l[0]*xx+popt_l[1])-self.popt[3])
                self.funcs.append(self.func)
                self.best_params[i] = self.popt[:-1]
            else:
                def f(x, a, b, c, d):
                    return a*np.sin(b*(x-c))+d
                self.popt, self.pcov = curve_fit(f, xx, yy, p0=[a,b,c,d], maxfev=5000)
                self.func = self.popt[0]*np.sin(self.popt[1]*(xx-self.popt[2]))+self.popt[3]
                self.funcs.append(self.func)
                self.best_params[i] = self.popt[0:-1]

        avrg = np.average(abs(self.best_params),axis=0)
        self.avrg_amp = avrg[0]
        self.avrg_period = avrg[1]
        self.avrg_phase = avrg[2]

        coords = self.funcs, self.avrg_amp, self.avrg_period, self.avrg_phase, self.best_params #pickle file of data created, used in use_previous
        filepath = self.thefilepath+'/prev_curve_fit.pkl'
        with open(filepath,'wb') as fo:
            pickle.dump(coords,fo)
            fo.close()

    def loop_len(self,vmax=50000):
        plt.imshow(np.sum(self.data, axis=0), vmin=0, vmax=vmax, cmap=self.irissjicolor)
        plt.scatter(self.x_area, self.y_area, s=0.0001)
        plt.plot(self.x_area[self.h], self.y_area[self.h],'g')
        print('Choose origin of loop.')
        origin_pos = click_path(ecolor='Blue', thick=1, psym=8)
        self.origin = [np.array(origin_pos.xs),np.array(origin_pos.ys)]
        points = int(input('loop points: '))
        if points > 1:
            x = np.zeros(points)
            y = np.zeros(points)
            for i in range(points):
                point_pos = click_path(ecolor='Blue', thick=1, psym=8)
                x[i] = point_pos.xs[0]
                y[i] = point_pos.ys[0]
                print(i)
            self.x = np.average(x)
            self.y = np.average(y)
        elif points == 1:
            point_pos = click_path(ecolor='Blue', thick=1, psym=8)
            self.x = point_pos.xs[0]
            self.y = point_pos.ys[0]
        elif points == 0:
            raise ValueError('points must be 1 or larger')

        radius = np.sqrt((self.x-self.origin[0])**2+(self.y-self.origin[1])**2)
        self.loop_length = radius*np.pi

        coords = self.loop_length   #pickle file of data created
        filepath = self.thefilepath+'/prev_loop_len.pkl'
        with open(filepath,'wb') as fo:
            pickle.dump(coords,fo)
            fo.close()

    def convert_units(self):
        self.sji = fits.open(self.thefilepath+'/'+self.filename)
        dt = self.sji[0].header['CADEX_AV']
        dx = self.sji[0].header['CDELT1']
        dy = self.sji[0].header['CDELT2']
        AU = 149597871*0.001    #[Mm]
        rad = 0.0003*(np.pi/180)


        self.best_params_conv = np.zeros(self.best_params.shape)

        for i in range(self.best_params.shape[0]):
            self.best_params_conv[i] = self.best_params[i]*np.array([dx*rad*AU,dt*np.pi/(self.best_params[i,1]**2),dx*rad*AU])

        avrg = np.average(abs(self.best_params_conv),axis=0)
        self.avrg_amp_conv = avrg[0]
        self.avrg_period_conv = avrg[1]
        self.avrg_phase_conv = avrg[2]

        self.loop_length_conv = self.loop_length*dx*rad*AU

    def use_previous(self):
        """
        Option to use variables made the last time. Therefore able to skip the use of redoing superimpose, make_cut, wiggles, curve_fit ans loop_len. Define which previous variables to use by keywords: all, superimpose, cut, wiggles, curve fit, loop. Everything is before converion of units.
        """

        choice = input('which? (all, superimpose, cut, wiggles, curve fit, loop) ')

        if choice == 'all':
            self.data, self.xx,self.yy,self.sum_data, self.timesteps = pickle.load(open(self.thefilepath+'/prev_superimpose.pkl','rb'))
            self.cut, self.eps, self.x_area, self.y_area = pickle.load(open(self.thefilepath+'/prev_make_cut.pkl','rb'))
            self.all_pos, self.h = pickle.load(open(self.thefilepath+'/prev_wiggles.pkl','rb'))
            self.funcs, self.avrg_amp, self.avrg_period, self.avrg_phase, self.best_params = pickle.load(open(self.thefilepath+'/prev_curve_fit.pkl','rb'))
            self.loop_length = pickle.load(open(self.thefilepath+'/prev_loop_len.pkl','rb'))

        if 'superimpose' in choice:
            self.data, self.xx, self.yy, self.sum_data, self.timesteps = pickle.load(open(self.thefilepath+'/prev_superimpose.pkl','rb'))

        if 'cut' in choice:
            self.cut, self.eps, self.x_area, self.y_area = pickle.load(open(self.thefilepath+'/prev_make_cut.pkl','rb'))

        if 'wiggles' in choice:
            self.all_pos, self.h = pickle.load(open(self.thefilepath+'/prev_wiggles.pkl','rb'))

        if 'curve fit' in choice:
            self.funcs, self.avrg_amp, self.avrg_period, self.avrg_phase, self.best_params = pickle.load(open(self.thefilepath+'/prev_curve_fit.pkl','rb'))

        if 'loop' in choice:
            self.loop_length = pickle.load(open(self.thefilepath+'/prev_loop_len.pkl','rb'))

    def visualizer(self):
        """
        Option to visualize superimposed image with overlay of the path area and cut chosen, and time distance plot with best fit wiggles.
        """
        plt.subplot(121)    #superimposed image
        plt.imshow(self.sum_data, vmin=0, vmax=1500, cmap=self.irissjicolor)
        plt.scatter(self.x_area, self.y_area, s=0.0005)
        plt.plot(self.x_area[self.h], self.y_area[self.h],'g')

        plt.subplot(122)    #time ditance image
        plt.imshow(self.cut[:,self.h-self.eps:self.h+self.eps,:].mean(1).T, vmin=0, vmax=30, cmap=self.irissjicolor,aspect=0.4)
        for i in range(self.all_pos.shape[0]):
            plt.plot(self.all_pos[i,0], self.funcs[i])

    def the_final_pickle(self):
        """
        Option to save important arrays to analyze and recreate data. File saved as pickle file with the date data was observed by the iris spacecraft.
        -----
        Improvements: (22.07.19)
        added
         - self.data
         - self.sum_data
         - self.all_pos to dict
         - self.funcs
         - self.best_params
         larger dictionary
        Imporvements: (25.07.19)
         - converion added
         - loop length added
        """

        super_pos = {'data':self.data, 'sum_data':self.sum_data, 'x_pos':self.xx, 'y_pos':self.yy}
        time_dist = {'h':self.h, 'cut':self.cut, 'x_area':self.x_area, 'y_area':self.y_area}
        wigl_pos = {'all_pos':self.all_pos, 'funcs':self.funcs}
        best_params = self.best_params_conv
        avrg = {'amplitude':self.avrg_amp_conv, 'period':self.avrg_period_conv, 'phase':self.avrg_phase_conv}
        loop_len = self.loop_length_conv

        dict = {'superposition':super_pos, 'time-distance':time_dist, 'averages':avrg, 'wiggles':wigl_pos, 'best parameters':best_params, 'loop length':loop_len}
        filepath = self.thefilepath+'/'+self.date+'.pkl'
        with open(filepath,'wb') as fo:
            pickle.dump(dict,fo)
            fo.close()

def get_results(file, wav, best_params=True, plot=True):
    """
    Function to open picke file with analysis results for all files made after 22.07.19.
    -----
    file: file with saved values to represent
    wav: wavelength of observation
    """

    data = pickle.load(open(file,'rb'))
    irissjicolor = plt.get_cmap('irissji'+wav)

    super_pos = data['superposition']
    time_dist = data['time-distance']
    wigl_pos = data['wiggles']
    avg = data['averages']

    if plot:
        plt.subplot(121)
        plt.imshow(super_pos['sum_data'], vmin=0, vmax=50000, cmap=irissjicolor)
        plt.scatter(time_dist['x_area'], time_dist['y_area'], s=0.0005)
        plt.plot(time_dist['x_area'][time_dist['h']], time_dist['y_area'][time_dist['h']],'g')

        plt.subplot(122)
        plt.imshow(time_dist['cut'][:,time_dist['h']-15:time_dist['h']+15,:].mean(1).T, vmin=0, vmax=20, cmap=irissjicolor,aspect=0.4)
        for i in range(wigl_pos['all_pos'].shape[0]):
            plt.plot(wigl_pos['all_pos'][i,0], wigl_pos['funcs'][i])
        plt.show()

    print('AVERAGES\n--------')
    print('amplitude: %.1f Mm\nperiod:    %.1f s\nphase:     %.1f Mm' %(avg['amplitude'], avg['period'], avg['phase']))
    print('--------')
    print('loop length: %.1f Mm'%data['loop length'][0])

    if best_params:
        print('--------\n')
        print('BEST FIT PARAMETERS')
        for i in range(data['best parameters'].shape[0]):
            print('--------')
            print('amplitude: %.1f Mm\nperiod:    %.1f s\nphase:     %.1f Mm' %(data['best parameters'][i,0], data['best parameters'][i,1], data['best parameters'][i,2]))

"""
    def use_pickle(self):
        self.pkl_filename = self.date+'.pkl'
        file = pickle.load(open(self.thefilepath+'/'+self.pkl_filename,'rb'))
        super_pos = file['superposition']
        time_dist = file['time-distance']
        wigl_pos = file['wiggles']
        avg = file['averages']
        self.data, self.xx,self.yy,self.sum_data = super_pos['data'], super_pos['x_pos'], super_pos['y_pos'], super_pos['sum_data']
        self.cut, self.eps, self.x_area, self.y_area = time_dist['cut'], 15, time_dist['x_area'], time_dist['y_area']
        self.all_pos, self.h = wigl_pos['all_pos'], time_dist['h']
        self.funcs, self.avrg_amp_conv, self.avrg_period_conv, self.avrg_phase_conv = wigl_pos['funcs'], avg['amplitude'], avg['period'], avg['phase']
        self.best_params_conv = file['best parameters']
        self.loop_length_conv = file['loop length']

        pos_x = []
        pos_y = []

        for i in range(self.all_pos.shape[0]):
            pos_x.append(self.all_pos[i][0])
            pos_y.append(self.all_pos[i][1])

        plt.imshow(self.cut[:,self.h-self.eps:self.h+self.eps,:].mean(1).T, vmin=0, vmax=70, cmap=self.irissjicolor, aspect=0.4)
        pos_c = click_path(ecolor='Blue', thick=1, psym=8)
        pos_x.append(np.array(pos_c.xs))
        pos_y.append(np.array(pos_c.ys))

        self.all_pos= np.array(list(zip(pos_x,pos_y)))
"""
