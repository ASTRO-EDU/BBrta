import numpy as np
import matplotlib.pyplot as plt

class BBlocksUtils:
    def __init__(self):
        pass

    # Plot methods
    def plot_blocks(self, bb, color='black', label="", normalization=None, t_delta=True, y_err=True, data_points=False, edge_points=False, data_cells=False, mean_blocks=False, sum_blocks=False, rate_blocks=True):

        """
        Plot the results of the Bayesian Blocks analysis, including the light curve, 
        detected blocks, and optionally the mean value of each block.
        
        Parameters:
        -----------
        t_delta : bool, optional
            If True, includes error bars based on time resolution (dt) and Poisson uncertainties 
            in the plot. Default is True.
        edge_points : bool, optional
            If True, plots the positions of the edge points between blocks. Default is True.
        data_cells : bool, optional
            If True, plots the positions of data cells or segments identified by the algorithm. 
            Default is True.
        mean_blocks : bool, optional
            If True, plots the mean value within each identified block. Default is True.
        sum_blocks : bool, optional
            If True, plots the sum within each identified block. Default is False.
        
        Raises:
        -------
        Exception:
            If an unsupported value is provided for 'alg'.
        """
        
        # A small constant to avoid division by zero or other numerical issues
        eps = 1e+5
        self.data_in = bb.get_data_in()
        self.data_out = bb.get_data_out()
        self.bb = bb
        self.color = color

        # Determine error bars: Use dt for time and sqrt of counts for Poisson error if t_delta is True
        if t_delta:
            try:
                xerr = self.data_in['t_delta'] / 2.0
            except:
                xerr = np.full_like(self.data_in['x'], self.data_in['dt'] / 2.0)
            if y_err:
                #yerr = np.sqrt(self.data_in['x'])
                yerr = self.data_in['sigma']
            else:
                yerr = np.zeros_like(self.data_in['x'])
        else:
            xerr = np.zeros_like(self.data_in['x'])
            yerr = np.zeros_like(self.data_in['x'])
              

        
        # Create the figure for plotting
        #plt.figure(figsize=(10, 6))
        
        # Plot the edge points if specified
        if edge_points:
            plt.vlines(self.data_out['edge_points'], 0, max(self.data_in['x'] + yerr), 
                       label='Edge blocks', color=self.color, ls='--', 
                       linewidth=1.5)
        
        # Plot the data cells if specified
        if data_cells:
            plt.vlines(self.data_out['data_cells'], 0, max(self.data_in['x'] + yerr), 
                       label='Data cell', color='gray', ls='--', 
                       linewidth=0.5)
        
        # Plot the actual data with error bars
        if data_points:
            plt.errorbar(self.data_in['t'], self.data_in['x'], 
                     xerr=xerr, yerr=yerr, fmt="o", color='tab:blue', 
                     label='light curve')
        
        # Plot the mean values of the blocks if specified
        if mean_blocks:
            means = []
            i_edge = 0
            # Normalize if needed
            meanb = self.data_out['mean_blocks']
            edge_points = self.data_out['edge_points']
            data_cells = self.data_out['edge_points']
            if normalization == 'max':
                max_mean_block = max(meanb)
                meanb = meanb / max_mean_block
            elif normalization == 'integral':
                # Calculate the width of each block for integral normalization
                block_widths = np.diff(np.concatenate(([edge_points[0]], edge_points, [edge_points[-1]])))
                integral = np.sum(meanb * block_widths)
                meanb = meanb / integral
            # Calculate mean for each block
            for t in self.data_out['data_cells']:
                means.append(meanb[i_edge])
                if i_edge < len(self.data_out['edge_points']) and t >= self.data_out['edge_points'][i_edge]:
                    i_edge += 1
            plt.step(self.data_out['data_cells'], means, color=self.color, label='blocks mean ' + label)
            #print(self.data_out['data_cells'], means)
            plt.ylabel('Count')

        if rate_blocks:
            rates = []
            i_edge = 0
            # Normalize if needed
            meanb = self.data_out['blockrate2']
            edge_points = self.data_out['edge_points']
            data_cells = self.data_out['edge_points']
            if normalization == 'max':
                max_mean_block = max(meanb)
                meanb = meanb / max_mean_block
            elif normalization == 'integral':
                # Calculate the width of each block for integral normalization
                block_widths = np.diff(np.concatenate(([edge_points[0]], edge_points, [edge_points[-1]])))
                integral = np.sum(meanb * block_widths)
                meanb = meanb / integral
            # Calculate mean for each block
            for t in self.data_out['data_cells']:
                rates.append(meanb[i_edge])
                if i_edge < len(self.data_out['edge_points']) and t >= self.data_out['edge_points'][i_edge]:
                    i_edge += 1
            plt.step(self.data_out['data_cells'], rates, color=self.color, label='rates ' + label)
            plt.ylabel('Rate')


        # Plot the mean values of the blocks if specified
        if sum_blocks:
            sumb = []
            i_edge = 0
            # Calculate sum for each block
            for t in self.data_out['data_cells']:
                sumb.append(self.data_out['sum_blocks'][i_edge])
                if i_edge < len(self.data_out['edge_points']) and t >= self.data_out['edge_points'][i_edge]:
                    i_edge += 1
            plt.step(self.data_out['data_cells'], sumb, color='red', label='blocks sum ' + label)
            plt.ylabel('Count')
        
        # Label the axes
        plt.xlabel('Time')
        
        
        # Add a legend to the plot
        plt.legend()
        
        # Rotate the x-axis labels for better readability
        #plt.xticks(self.data_in['t'], self.data_in['t'], rotation=90)

        # Reduce the number of xticks
        #xticks = plt.gca().get_xticks()  # Get current xticks
        #plt.xticks(xticks[::2], rotation=45)  # Only show every 2nd label and rotate them to 45 degrees
        
        # use MaxNLocator for automatic tick reduction
        from matplotlib.ticker import MaxNLocator
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=5))


        # Set the title of the plot
        plt.title(f'Bayesian Blocks Analysis on Time Series ()')
        # Uncomment plt.show() if running outside of a script that automatically renders plots
        # plt.show()
