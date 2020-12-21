import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import os
import logging
import xlwt
from src.mp_common import *

class mp_analyse:

    def __init__(self, config = None):
        
        # initialize logger
        self.logger = logging.getLogger('metapath')
        
        if config == None:
            return

        self.cfg = config

        
    def run(self, model):
        
        # check object class
        if not model.__class__.__name__ == 'mp_model':
            self.logger.error("could not run analysis: 'model' has to be mp_model instance!")
            return False
        
        for output in self.cfg['output']:
            parts = output.split(':')
            
            if parts[0] == 'report':
                self.logger.info(" * create report")
                self.report(model)
            elif parts[0] == 'plot':
                if len(parts) < 2:
                    continue
                
                self.logger.info(" * plot %s" % (parts[1]))
                self.plot(model, plot = parts[1])

        return True
        
    #
    # create plots
    #
        
    # plot wrapper
    def plot(self, model, plot = '', file = None):
        
        # check object class
        if not model.__class__.__name__ == 'mp_model':
            self.logger.error("could not plot model: 'model' has to be mp_model instance!")
            return False
        
        if plot == 'graph':
            return self.plot_graph(model, file = file)
        if plot == 'network':
            return self.plot_network(model, file = file)
        if plot == 'knockout':
            return self.plot_knockout(model, file = file)
        if plot == 'correlation':
            return self.plot_correlation(model, file = file)
        
        return False
        
    # plot model graph
    def plot_graph(self, model, output = 'file', file = None):
        
        # check object class
        if not model.__class__.__name__ == 'mp_model':
            self.logger.error("could not plot model: 'model' has to be mp_model instance!")
            return False
        
        # set default settings
        settings = {
            'dpi': 300,
            'file_ext': 'png',
            'edge_threshold': 0.0,
            'edge_weight': 'link_energy',
            'node_caption': 'rel_approx',
            'edge_zoom': 1}
                
        # set configured settings
        for key, value in self.cfg['plot']['model']['graph'].items():
            if key in settings:
                settings[key] = value
        
        # configure output
        if output == 'file':
        
            # get file name
            if not file:
                file = self.cfg['path'] + 'model-' + str(model.id) + \
                    '/graph-%s-%s.%s' % (settings['edge_weight'], settings['node_caption'], settings['file_ext'])
                
            # get empty file
            file = get_empty_file(file)
            
        # get title
        title = \
            r'$\mathbf{Network:}\,\mathrm{%s}' % (model.network.cfg['name']) \
            + r',\,\mathbf{Dataset:}\,\mathrm{%s}$' % (model.dataset.cfg['name'])

        # get node captions
        model_caption, node_caption = model.get_approx(type = settings['node_caption'])
        
        # get edge weights
        edge_weight = model.get_weights(type = settings['edge_weight'])
            
        # get model caption
        caption = \
            r'$\mathbf{Approximation:}\,\mathrm{%.1f' % (100 * model_caption) + '\%}$'

        # labels
        lbl = {}
        lbl['e'] = model.network.node_labels(type = 'e')
        lbl['tf'] = model.network.node_labels(type = 'tf')
        lbl['s'] = model.network.node_labels(type = 's')
        
        # graph
        G = model.network.graph

        # calculate sizes
        zoom = 1
        scale = min(250.0 / max(len(lbl['e']), len(lbl['tf']), len(lbl['s'])), 30.0)
        graph_node_size = scale ** 2
        graph_font_size = 0.4 * scale
        graph_caption_factor = 0.5 + 0.003 * scale
        graph_line_width = 0.5

        # calculate node positions for 'stack layout'
        pos = {}
        pos_caption = {}
        for node, attr in G.nodes(data = True):
            i = 1.0 / len(lbl[attr['params']['type']])
            x_node = (attr['params']['type_node_id'] + 0.5) * i
            y_node = attr['params']['type_id'] * 0.5
            y_caption = (attr['params']['type_id'] - 1) * graph_caption_factor + 0.5
            
            pos[node] = (x_node, y_node)
            pos_caption[node] = (x_node, y_caption)

        # create figure object
        fig = plt.figure()

        # draw labeled nodes
        for node, attr in G.nodes(data = True):
            type = attr['params']['type']
            label = attr['label']
            
            weight_sum = 0
            for (n1, n2, edge_attr) in G.edges(nbunch = [node], data = True):
                weight_sum += np.abs(edge_weight[(n1, n2)])
                
            weight_sum = min(0.01 + 0.3 * weight_sum, 1)
            c = 1 - weight_sum
            color = {
                's': (1, c, c, 1),
                'tf': (c, 1, c,  1),
                'e': (1, 1, c, 1)
            }[type]
            
            # draw node
            nx.draw_networkx_nodes(
                G, pos,
                node_size = graph_node_size,
                linewidths = graph_line_width,
                nodelist = [node],
                node_shape = 'o',
                node_color = color)
            
            # draw node label
            node_font_size = \
                1.5 * graph_font_size / np.sqrt(max(len(node) - 1, 1))
            nx.draw_networkx_labels(
                G, pos,
                font_size = node_font_size,
                labels = {node: label},
                font_weight = 'normal')
            
            # draw node caption
            if not type == 'tf':
                nx.draw_networkx_labels(
                    G, pos_caption,
                    font_size = 0.65 * graph_font_size,
                    labels = {node: ' $' + '%d' % (100 * node_caption[node]) + '\%$'},
                    font_weight = 'normal')
        
        # draw labeled edges
        for (v, h) in G.edges():
            
            if edge_weight[(v, h)] < 0:
                color = 'red'
            else:
                color = 'green'
                    
            if np.abs(edge_weight[(v, h)]) > settings['edge_threshold']:
                nx.draw_networkx_edges(
                    G, pos,
                    width = np.abs(edge_weight[(v, h)]) * graph_line_width * settings['edge_zoom'],
                    edgelist = [(v, h)],
                    edge_color = color,
                    alpha = 1)
                    
                size = graph_font_size / 1.5
                label = ' $' + ('%.2g' % (np.abs(edge_weight[(v, h)]))) + '$'
                nx.draw_networkx_edge_labels(
                    G, pos,
                    edge_labels = {(v, h): label},
                    font_color = color,
                    clip_on = False,
                    font_size = size, font_weight = 'normal')
            
        # draw title
        plt.figtext(.5, .92, title, fontsize = 10, ha = 'center')

        # draw caption
        if caption == None:
            if edges == 'weights':
                label_text = r'$\mathbf{Network:}\,\mathrm{%s}$' % (self.graph)
            elif edges == 'adjacency':
                label_text = r'$\mathbf{Network:}\,\mathrm{%s}$' % (self.graph)
            plt.figtext(.5, .06, label_text, fontsize = 10, ha = 'center')
        else:
            plt.figtext(.5, .06, caption, fontsize = 10, ha = 'center')

        plt.axis('off')
        
        # output
        if output == 'file':
            plt.savefig(file, dpi = settings['dpi'])
        elif output == 'screen':
            plt.show()
        
        # clear current figure object and release memory
        plt.clf()
        plt.close(fig)
    
    # plot model knockout test as heatmap
    def plot_knockout(self, model, output = 'file', file = None):

        # check object class
        if not model.__class__.__name__ == 'mp_model':
            self.logger.error("could not plot model knockout test: 'model' has to be mp_model instance!")
            return False
        
        # set default settings
        settings = {
            'dpi': 300,
            'file_ext': 'png',
            'interpolation': 'nearest'}
                
        # set configured settings
        for key, value in self.cfg['plot']['model']['knockout'].items():
            if key in settings:
                settings[key] = value

        # configure output
        if output == 'file':
        
            # get file name
            if not file:
                file = self.cfg['path'] + 'model-' + str(model.id) + '/knockout.' + settings['file_ext']
                
            # get empty file
            file = get_empty_file(file)

        # calculate knockout matrix
        data = model.get_knockout_matrix()
        title = 'Simulated Gene Knockout Test'
        label = model.dataset.cfg['label']
        
        # create figure object
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True)
        num = len(label)
        cax = ax.imshow(data,
            cmap = matplotlib.cm.hot_r,
            interpolation = settings['interpolation'],
            extent = (0, num, 0, num))
        
        # set ticks and labels
        plt.xticks(
            np.arange(num) + 0.5,
            tuple(label),
            fontsize = 9,
            rotation = 70)
        plt.yticks(
            num - np.arange(num) - 0.5,
            tuple(label),
            fontsize = 9)
        
        # add colorbar
        cbar = fig.colorbar(cax)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontsize(9)
        
        # draw title
        plt.title(title, fontsize = 11)
        
        # output
        if output == 'file':
            plt.savefig(file, dpi = settings['dpi'])
        elif output == 'screen':
            plt.show()
        
        # clear current figure object and release memory
        plt.clf()
        plt.close(fig)
        
    # plot network as graph
    def plot_network(self, model, output = 'file', file = None):
        
        # check object class
        if not model.__class__.__name__ == 'mp_model':
            self.logger.error("could not plot model knockout test: 'model' has to be mp_model instance!")
            return False
        
        # get network
        network = model.network
        
        # set default settings
        settings = {
            'dpi': 300,
            'file_ext': 'png'}
                
        # set configured settings
        for key, value in self.cfg['plot']['network']['graph'].items():
            if key in settings:
                settings[key] = value
                
        # configure output
        if output == 'file':
        
            # get file name
            if not file:
                file = '%smodel-%s/network-%s.%s' % (self.cfg['path'], model.id, network.cfg['id'], settings['file_ext'])
                
            # get empty file
            file = get_empty_file(file)

        # plot parameters
        dpi = 300
        title = r'$\mathbf{Network:}\,\mathrm{%s}$' % (network.cfg['name'])

        # labels
        lbl = {}
        lbl['e'] = network.node_labels(type = 'e')
        lbl['tf'] = network.node_labels(type = 'tf')
        lbl['s'] = network.node_labels(type = 's')
        
        # graph
        G = network.graph

        # calculate sizes
        zoom = 1
        scale = min(250.0 / max(len(lbl['e']), len(lbl['tf']), len(lbl['s'])), 30.0)
        graph_node_size = scale ** 2
        graph_font_size = 0.4 * scale
        graph_caption_factor = 0.5 + 0.003 * scale
        graph_line_width = 0.5

        # calculate node positions for 'stack layout'
        pos = {}
        pos_caption = {}
        for node, attr in G.nodes(data = True):
            i = 1.0 / len(lbl[attr['params']['type']])
            x_node = (attr['params']['type_node_id'] + 0.5) * i
            y_node = attr['params']['type_id'] * 0.5
            y_caption = (attr['params']['type_id'] - 1) * graph_caption_factor + 0.5
            
            pos[node] = (x_node, y_node)
            pos_caption[node] = (x_node, y_caption)

        # create figure object
        fig = plt.figure()

        # draw labeled nodes
        for node, attr in G.nodes(data = True):
            type = attr['params']['type']
            label = attr['label']
            
            color = {
                's': (1, 0, 0, 1),
                'tf': (0, 1, 0,  1),
                'e': (1, 1, 0, 1)
            }[type]
            
            # draw node
            nx.draw_networkx_nodes(
                G, pos,
                node_size = graph_node_size,
                linewidths = graph_line_width,
                nodelist = [node],
                node_shape = 'o',
                node_color = color)
            
            # draw node label
            node_font_size = \
                1.5 * graph_font_size / np.sqrt(max(len(node) - 1, 1))
            nx.draw_networkx_labels(
                G, pos,
                font_size = node_font_size,
                labels = {node: label},
                font_weight = 'normal')
        
        # draw unlabeled edges
        for (v, h) in G.edges():
                    
            nx.draw_networkx_edges(
                G, pos,
                width = graph_line_width,
                edgelist = [(v, h)],
                edge_color = 'black',
                alpha = 1)
            
        # draw title
        plt.figtext(.5, .92, title, fontsize = 10, ha = 'center')
        plt.axis('off')
        
        # output
        if output == 'file':
            plt.savefig(file, dpi = settings['dpi'])
        elif output == 'screen':
            plt.show()
        
        # clear current figure object and release memory
        plt.clf()
        plt.close(fig)
    
    # plot dataset correlation as heatmap
    def plot_correlation(self, model, output = 'file', file = None):
        
        # check model object class
        if not model.__class__.__name__ == 'mp_model':
            self.logger.error("could not plot dataset correlation: 'model' has to be mp_model instance!")
            return False
        
        # get dataset
        dataset = model.dataset
        
        # set default settings
        settings = {
            'dpi': 300,
            'file_ext': 'png',
            'interpolation': 'nearest'}
                
        # set configured settings
        for key, value in self.cfg['plot']['dataset']['correlation'].items():
            if key in settings:
                settings[key] = value
                
        # configure output
        if output == 'file':
        
            # get file name
            if not file:
                file = self.cfg['path'] + 'model-%s/dataset-%s-correlation.%s' % (model.id, dataset.cfg['id'], settings['file_ext'])
                
            # get empty file
            file = get_empty_file(file)

        # create correlation matrix
        data = np.corrcoef(dataset.data.T)
        title = dataset.cfg['name']
        
        # create figure object
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.grid(True)
        num = len(dataset.cfg['label'])
        cax = ax.imshow(data,
            cmap = matplotlib.cm.hot_r,
            interpolation = settings['interpolation'],
            extent = (0, num, 0, num))
        
        # set ticks and labels
        plt.xticks(
            np.arange(num) + 0.5,
            tuple(dataset.cfg['label']),
            fontsize = 9,
            rotation = 70)
        plt.yticks(
            num - np.arange(num) - 0.5,
            tuple(dataset.cfg['label']),
            fontsize = 9)
        
        # add colorbar
        cbar = fig.colorbar(cax)
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontsize(9)
        
        # draw title
        plt.title(title, fontsize = 11)
        
        # output
        if output == 'file':
            plt.savefig(file, dpi = settings['dpi'])
        elif output == 'screen':
            plt.show()
        
        # clear current figure object and release memory
        plt.clf()
        plt.close(fig)

    #
    # create tables
    #

    def report(self, model, file = None):
        
        # check object class
        if not model.__class__.__name__ == 'mp_model':
            self.logger.error("could not create table: 'model' has to be mp_model instance!")
            return False
        
        # set default settings
        settings = {
            'columns': ['knockout_approx']}
        
        # get file name
        if not file:
            file = self.cfg['path'] + 'model-%s/report.xls' % (model.id)
            
        # get empty file
        file = get_empty_file(file)

        # start document
        book = xlwt.Workbook(encoding="utf-8")

        # define common styles
        style_sheet_head = xlwt.Style.easyxf(
            'font: height 300;')
        style_section_head = xlwt.Style.easyxf('font: bold True;')
        style_head = xlwt.Style.easyxf(
            'pattern: pattern solid, fore_colour gray_ega;'
            'borders: bottom thin;'
            'font: colour white;')
        style_str = xlwt.Style.easyxf('', '')
        style_num = xlwt.Style.easyxf('alignment: horizontal left;', '#,###0.000')
        style_num_1 = xlwt.Style.easyxf('alignment: horizontal left;', '#,###0.0')
        style_num_2 = xlwt.Style.easyxf('alignment: horizontal left;', '#,###0.00')
        style_num_3 = xlwt.Style.easyxf('alignment: horizontal left;', '#,###0.000')
        style_border_left = xlwt.Style.easyxf('borders: left thin;')
        style_border_bottom = xlwt.Style.easyxf('borders: bottom thin;')

        sheet = {}
        
        #
        # EXCEL SHEET 'UNITS'
        #
        
        sheet['units'] = book.add_sheet("Units")
        row = 0
        
        # write sheet headline
        sheet['units'].row(row).height = 390
        sheet['units'].row(row).write(0, 'Units', style_sheet_head)
        row +=2
        
        # write section headline
        sheet['units'].row(row).write(0, 'Unit', style_section_head)
        sheet['units'].row(row).write(3, 'Data', style_section_head)
        sheet['units'].row(row).write(5, 'Parameter', style_section_head)
        sheet['units'].row(row).write(7, 'Effect', style_section_head)
        row += 1
        
        # write column headline
        columns = [ {
                'label': 'id', 'col': 0,
                'info': 'node_id', 'type': 'string', 'style': style_str
            }, {
                'label': 'class', 'col': 1,
                'info': 'node_type', 'type': 'string', 'style': style_str
            }, {
                'label': 'label', 'col': 2,
                'info': 'node_label', 'type': 'string', 'style': style_str
            }, {
                'label': 'mean', 'col': 3,
                'info': 'data_mean', 'type': 'number', 'style': style_num
            }, {
                'label': 'sdev', 'col': 4,
                'info': 'data_sdev', 'type': 'number', 'style': style_num
            }, {
                'label': 'bias', 'col': 5,
                'info': 'model_bias', 'type': 'number', 'style': style_num
            }, {
                'label': 'sdev', 'col': 6,
                'info': 'model_sdev', 'type': 'number', 'style': style_num
            }, {
                'label': 'rel approx [%]', 'col': 7,
                'info': 'model_rel_approx', 'type': 'number', 'style': style_num_1
            }, {
                'label': 'abs approx [%]', 'col': 8,
                'info': 'model_abs_approx', 'type': 'number', 'style': style_num_1
            }]
            
        if 'knockout_approx' in settings['columns']:
            columns.append({
                'label': 'knockout', 'col': 9,
                'info': 'model_knockout_approx', 'type': 'number', 'style': style_num_1
            })

        for cell in columns:
            sheet['units'].row(row).write(cell['col'], cell['label'], style_head)
        
        row += 1
        
        # write unit information
        nodes = model.network.nodes()
        
        # get simulation info
        model_rel_approx, node_rel_approx = model.get_approx(type = 'rel_approx')
        model_abs_approx, node_abs_approx = model.get_approx(type = 'abs_approx')
            
        if 'knockout_approx' in settings['columns']:
            node_knockout_approx = model.get_knockout_approx()

        for node in nodes:
            # create dict with info
            info = {}
            
            # get node information
            network_info = model.network.node(node)
            info['node_id'] = node
            info['node_type'] = network_info['params']['type']
            info['node_label'] = network_info['label']

            # get data and model information
            machine_info = model.machine.unit(node)
            if machine_info['type'] == 'visible':
                info['data_mean'] = machine_info['data']['mean']
                info['data_sdev'] = machine_info['data']['sdev']
                info['model_bias'] = machine_info['params']['bias']
                info['model_sdev'] = machine_info['params']['sdev']
                info['model_rel_approx'] = node_rel_approx[node] * 100
                info['model_abs_approx'] = node_abs_approx[node] * 100
            else:
                info['model_bias'] = machine_info['params']['bias']
                
            if 'knockout_approx' in settings['columns']:
                info['model_knockout_approx'] = node_knockout_approx[node] * 100
                
            # write cell content
            for cell in columns:
                if not cell['info'] in info:
                    continue
                
                if cell['type'] == 'string':
                    sheet['units'].row(row).write(
                        cell['col'], info[cell['info']], cell['style'])
                elif cell['type'] == 'number':
                    sheet['units'].row(row).set_cell_number(
                        cell['col'], info[cell['info']], cell['style'])
                        
            row += 1

        #
        # EXCEL SHEET 'LINKS'
        #
        
        sheet['links'] = book.add_sheet("Links")
        row = 0
        
        # write sheet headline
        sheet['links'].row(row).height = 390
        sheet['links'].row(row).write(0, 'Links', style_sheet_head)
        row +=2
        
        # write section headline
        sheet['links'].row(row).write(0, 'Source', style_section_head)
        sheet['links'].row(row).write(3, 'Target', style_section_head)
        sheet['links'].row(row).write(6, 'Parameter', style_section_head)
        sheet['links'].row(row).write(7, 'Effect', style_section_head)
        row += 1
            
        # write column headline
        columns = [ {
                'label': 'id', 'col': 0,
                'info': 'src_node_id', 'type': 'string', 'style': style_str
            }, {
                'label': 'class', 'col': 1,
                'info': 'src_node_type', 'type': 'string', 'style': style_str
            }, {
                'label': 'label', 'col': 2,
                'info': 'src_node_label', 'type': 'string', 'style': style_str
            },{
                'label': 'id', 'col': 3,
                'info': 'tgt_node_id', 'type': 'string', 'style': style_str
            }, {
                'label': 'class', 'col': 4,
                'info': 'tgt_node_type', 'type': 'string', 'style': style_str
            }, {
                'label': 'label', 'col': 5,
                'info': 'tgt_node_label', 'type': 'string', 'style': style_str
            }, {
                'label': 'weight', 'col': 6,
                'info': 'weight', 'type': 'number', 'style': style_num
            }, {
                'label': 'energy', 'col': 7,
                'info': 'energy', 'type': 'number', 'style': style_num
            } ]
            
        for cell in columns:
            sheet['links'].row(row).write(cell['col'], cell['label'], style_head)
        
        row += 1
        
        # write link information
        edges = model.network.edges()
        
        # link knockout simulation
        edge_energy = model.get_weights(type = 'link_energy')
        
        for (src_node, tgt_node) in edges:
            
            # create dict with info
            info = {}
            
            # get source node information
            network_info_src = model.network.node(src_node)
            info['src_node_id'] = src_node
            info['src_node_type'] = network_info_src['params']['type']
            info['src_node_label'] = network_info_src['label']
            
            # get target node information
            network_info_tgt = model.network.node(tgt_node)
            info['tgt_node_id'] = tgt_node
            info['tgt_node_type'] = network_info_tgt['params']['type']
            info['tgt_node_label'] = network_info_tgt['label']
            
            # simulation
            info['energy'] =  edge_energy[(src_node, tgt_node)]
            
            # get data and model information
            machine_info = model.machine.link((src_node, tgt_node))
            
            if not machine_info == {}:
                info['weight'] = machine_info['params']['weight']
            
            # write cell content
            for cell in columns:
                if not cell['info'] in info:
                    continue
                
                if cell['type'] == 'string':
                    sheet['links'].row(row).write(
                        cell['col'], info[cell['info']], cell['style'])
                elif cell['type'] == 'number':
                    sheet['links'].row(row).set_cell_number(
                        cell['col'], info[cell['info']], cell['style'])
                        
            row += 1
            
        book.save(file)