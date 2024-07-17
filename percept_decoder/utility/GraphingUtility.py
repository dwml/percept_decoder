# -*- coding: utf-8 -*-
"""
@author: Jackson Cagle, University of Florida
@email: jackson.cagle@neurology.ufl.edu
@date: Fri Oct 17 2020
"""

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy import stats

import plotly.graph_objects as go
import plotly.offline as po
from plotly.subplots import make_subplots
from matplotlib import dates

def stderr(data, axis=0):
    return np.std(data, axis=axis)/np.sqrt(data.shape[axis])

def shadedErrorBar(x, y, shadedY, lineprops=dict(), alpha=1.0, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    
    line = ax.plot(x,y)[0]
    shade = ax.fill_between(x, y-shadedY, y+shadedY, alpha=alpha)
    
    for key in lineprops.keys():
        if key == "color":
            line.set_color(lineprops[key])
            shade.set_color(lineprops[key])
        elif key == "alpha":
            shade.set_alpha(lineprops[key])
        elif key == "linewidth":
            line.set_linewidth(lineprops[key])
        elif key == "linestyle":
            line.set_linestyle(lineprops[key])
    
    return (line, shade)

def surfacePlot(X, Y, Z, cmap=plt.get_cmap("jet"), ax=None):
    if ax is None:
        ax = plt.gca()
        
    bound = (X[0], X[-1], Y[0], Y[-1])
    image = ax.imshow(Z, cmap=cmap, aspect="auto", origin="lower", extent=bound, interpolation="gaussian")
    ax.set_ylim(Y[0], Y[-1])
    
    return image

def addColorbar(ax, image, title, padding=5.5):
    colorAxes = inset_axes(ax, width="2%", height="100%", loc="right", bbox_to_anchor=(0.05, 0, 1, 1), bbox_transform=ax.transAxes)
    colorbar = plt.colorbar(image, cax=colorAxes)
    #colorAxes.set_title(title,rotation=-90, loc="right", verticalalignment="center")
    colorAxes.set_ylabel(title, rotation=-90, verticalalignment="bottom")
    return colorAxes
    
def addExternalLegend(ax, lines, legends, fontsize=12):
    legendAxes = inset_axes(ax, width="2%", height="100%", loc="right", bbox_to_anchor=(0.15, 0.05, 1, 1), bbox_transform=ax.transAxes)
    legendAxes.legend(lines, legends, frameon=False, fontsize=fontsize)
    legendAxes.axis("off")
    return legendAxes
    
def singleViolin(x, y, width=0.5, showmeans=False, showextrema=True, showmedians=False, vert=True, color=None, ax=None):
    if ax is None:
        ax = plt.gca()
        
    violin = ax.violinplot(y, positions=[x], widths=[width], showmeans=showmeans, showextrema=showextrema, showmedians=showmedians, vert=vert)
    
    if color != None:
        lineParts = list()
        if showextrema:
            lineParts.append("cbars")
            lineParts.append("cmins")
            lineParts.append("cmaxes")
        if showmedians:
            lineParts.append("cmedians")
        if showmeans:
            lineParts.append("cmeans")
        
        for pc in lineParts:
            vp = violin[pc]
            vp.set_edgecolor(color)
            vp.set_linewidth(2)
        
        for pc in violin["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(0.4)
            
    return violin

def standardErrorBox(x, y, width=0.5, color=None, sym="*", showfliers=True, flierprops=None, ax=None):
    if ax is None:
        ax = plt.gca()
    
    if not color == None:
        box = ax.boxplot(y, positions=[x], widths=[width], showfliers=showfliers, sym=sym, patch_artist=True)
        for patch in box['boxes']:
            patch.set(facecolor=color)
    else:
        box = ax.boxplot(y, positions=[x], widths=[width], showfliers=showfliers, sym=sym)
    
    return box
    

def addStatisticBar(data, ax=None):
    if ax is None:
        ax = plt.gca()
    
    totalGroups = len(data)
    totalTests = sum(range(totalGroups))

    SignificantStatistc = list()
    for index1 in range(totalGroups):
        for index2 in range(index1 + 1, totalGroups):
            pvalue = stats.ttest_ind(data[index1]["Y"],data[index2]["Y"],equal_var=False).pvalue
            if pvalue * totalTests < 0.01:
                SignificantStatistc.append({"pvalue": pvalue*totalTests, "Group1": [data[index1]["X"],max(data[index1]["Y"])], "Group2": [data[index2]["X"],max(data[index2]["Y"])]})

    existingBarHeight = np.zeros(len(SignificantStatistc))
    lineIndex = 0
    for stat in SignificantStatistc:
        proposedHeight = max([stat["Group1"][1],stat["Group2"][1]]) + 1
        while np.any(existingBarHeight == proposedHeight):
            proposedHeight += 1
        existingBarHeight[lineIndex] = proposedHeight
        lineIndex += 1
        ax.plot([stat["Group1"][0],stat["Group2"][0]],[proposedHeight,proposedHeight], "k", linewidth=2)
        
        sigStars = "*"
        if stat["pvalue"] < 0.001:
            sigStars += "*"
        if stat["pvalue"] < 0.0001:
            sigStars += "*"
        ax.text(np.mean([stat["Group1"][0],stat["Group2"][0]]), proposedHeight+0.1, sigStars, fontsize=12, horizontalalignment="center", verticalalignment="center")

    return SignificantStatistc

def largeFigure(n, resolution=(1600,900), dpi=100.0):
    if n == 0:
        figure = plt.figure(figsize=(resolution[0]/dpi,resolution[1]/dpi), dpi=dpi)
    else:
        figure = plt.figure(n, figsize=(resolution[0]/dpi,resolution[1]/dpi), dpi=dpi)
    
    figure.clf()
    return figure

def imagesc(x, y, z, clim=None, cmap=plt.get_cmap("jet"), interpolation="gaussian", ax=None):
    bound = (x[0], x[-1], y[0], y[-1])

    if ax:
        image = ax.imshow(z, cmap=cmap, aspect="auto", origin="lower", extent=bound, interpolation=interpolation)
    else:
        image = plt.imshow(z, cmap=cmap, aspect="auto", origin="lower", extent=bound, interpolation=interpolation)
    if clim:
        image.set_clim(clim[0],clim[1])
    return image

def addAxes(fig):
    return fig.add_axes([0.1,0.1,0.8,0.8])

def colorTextFromCmap(color):
    if type(color) == str:
        colorInfoString = color.split(",")
        colorInfoString = [string.replace("rgb(","").replace(")","") for string in colorInfoString]
        colorInfo = [int(i) for i in colorInfoString]
    else:
        colorInfo = np.array(color[:-1]) * 255
    colorText = f"#{hex(int(colorInfo[0])).replace('0x',''):0>2}{hex(int(colorInfo[1])).replace('0x',''):0>2}{hex(int(colorInfo[2])).replace('0x',''):0>2}"
    return colorText
class PlotlyFigure:
    def __init__(self, resolution=(1600,900), subplots=[1,1], vertical_spacing=0.1, subplot_titles=None, shared_xaxes=True, shared_yaxes=True, dpi=100.0):
        self.resolution = resolution 
        self.dpi = dpi
        self.layout = subplots
        self.sharex = shared_xaxes
        self.sharey = shared_yaxes

        self.fig = make_subplots(rows=subplots[0], cols=subplots[1], shared_xaxes=shared_xaxes, shared_yaxes=shared_yaxes,
            vertical_spacing=vertical_spacing, subplot_titles=subplot_titles)
        
        for i in range(subplots[0]*subplots[1]):
            self.set_ylayout(dict(type="linear", range=(0,100), tickfont_size=12,
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD"), ax=i)
            self.set_xlayout(dict(type="linear", range=(0,100), tickfont_size=12,
                     color="#000000", ticks="outside", showline=True, linecolor="#000000",
                     showgrid=True, gridcolor="#DDDDDD"), ax=i)
            
        self.fig.layout["paper_bgcolor"]="#FFFFFF"
        self.fig.layout["plot_bgcolor"]="#FFFFFF"
        self.fig.update_layout(hovermode="x")

    def plot(self, x, y, name="", color="#3BDEFF", linewidth=2, hovertemplate="<extra></extra>", legendgroup=None, ax=0):
        RGB = [int(color[i:i+2], 16) for i in (1, 3, 5)]
        
        self.fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines", name=name, 
                       line=dict(color="rgba({0},{1},{2},1)".format(RGB[0], RGB[1], RGB[2]), width=linewidth),
                       hovertemplate=hovertemplate, legendgroup=legendgroup, showlegend=True),
            row=self.getRow(ax), col=self.getCol(ax)
        )
    
    def box(self, x, y, name="", color="#3BDEFF", width=0.3, points="outliers", hovertemplate="<extra></extra>", legendgroup=None, ax=0):
        RGB = [int(color[i:i+2], 16) for i in (1, 3, 5)]
        self.fig.add_trace(
            go.Box(x=x, y=y, width=width, name=name,
                   marker_color="rgba({0},{1},{2},1)".format(RGB[0], RGB[1], RGB[2]), 
                   line_color="rgba({0},{1},{2},1)".format(RGB[0], RGB[1], RGB[2]),
                   hovertemplate=hovertemplate, legendgroup=legendgroup, showlegend=True),
            row=self.getRow(ax), col=self.getCol(ax)
        )

    def scatter(self, x, y, name="", color="#3BDEFF", size=2, hovertemplate="<extra></extra>", legendgroup=None, ax=0):
        RGB = [int(color[i:i+2], 16) for i in (1, 3, 5)]
        self.fig.add_trace(
            go.Scatter(x=x, y=y, mode="markers", name=name, 
                       marker=dict(color="rgba({0},{1},{2},1)".format(RGB[0], RGB[1], RGB[2]), size=size),
                       hovertemplate=hovertemplate, legendgroup=legendgroup, showlegend=True),
            row=self.getRow(ax), col=self.getCol(ax)
        )
    
    def getRow(self, ax):
        return int(ax / self.layout[1])+1
    
    def getCol(self, ax):
        return np.remainder(ax, self.layout[1])+1

    def addShadedErrorBar(self, x, y, stderr, color="#3BDEFF", alpha=0.5, legendgroup=None, ax=0):
        RGB = [int(color[i:i+2], 16) for i in (1, 3, 5)]

        self.fig.add_trace(
            go.Scatter(x=x,
                    y=y+stderr,
                    mode="lines",
                    line=dict(color=color, width=0.0),
                    fill=None,
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=legendgroup),
            row=self.getRow(ax), col=self.getCol(ax)
        )

        self.fig.add_trace(
            go.Scatter(x=x,
                    y=y-stderr,
                    mode="lines",
                    line=dict(color=color, width=0.0),
                    fill="tonexty", fillcolor="rgba({0},{1},{2},{3})".format(RGB[0], RGB[1], RGB[2], alpha),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=legendgroup),
            row=self.getRow(ax), col=self.getCol(ax)
        )

    def set_xlabel(self, label, fontsize=15, ax=0): 
        self.fig.update_xaxes(title_font_size=fontsize, title_text=label, 
            row=self.getRow(ax), col=self.getCol(ax))
        
    def set_xlim(self, limits, ax=0):
        if self.sharex:
            for i in range(self.layout[0]):
                self.fig.update_xaxes(range=limits, row=i+1, col=self.getCol(ax))
        else:
            self.fig.update_xaxes(range=limits, row=self.getRow(ax), col=self.getCol(ax))

    def set_xlayout(self, layoutprops, ax=0):
        self.fig.update_xaxes(layoutprops, row=self.getRow(ax), col=self.getCol(ax))

    def set_ylabel(self, label, fontsize=15, ax=0): 
        self.fig.update_yaxes(title_font_size=fontsize, title_text=label, 
            row=self.getRow(ax), col=self.getCol(ax))

    def set_ylim(self, limits, ax=0):
        if self.sharey:
            for i in range(self.layout[0]):
                self.fig.update_yaxes(range=limits, row=self.getRow(ax), col=i+1)
        else:
            self.fig.update_yaxes(range=limits, row=self.getRow(ax), col=self.getCol(ax))
    
    def set_ylayout(self, layoutprops, ax=0):
        self.fig.update_yaxes(layoutprops, row=self.getRow(ax), col=self.getCol(ax))

    def set_title(self, title, font="Arial", color="#000000", fontsize=15):
        self.fig.update_layout(title=title,  title_font_family=font, title_font_color=color, title_font_size=fontsize)
    
    def show(self, filename=None):
        if filename:
            return po.plot(self.fig, filename=filename, auto_open=False) 
        else:
            return po.plot(self.fig)
