"""Library for parsing metadata files from Bruker scope. Modified from Deisseroth lab code"""
import os
import json
import logging
import pathlib
import pprint

from xml.etree import ElementTree
logger = logging.getLogger(__name__)


class MetadataError(Exception):
    """Error while extracting metadata."""
    
class MetadataWarning(Warning):
    """Warning while extracting metadata."""

def read_main_xml(fname):
    '''
    
    '''
    
    data = {
        'layout': {
            'sequences': None,
            'frames_per_sequence': None,
            'samples_per_pixel': None
        },
        'size': {
            'frames': None,
            'channels': None,
            'z_planes': None,
            'y_px': None,
            'x_px': None,
        },
        'pixel_size': None,
        'laser_power': {},
        'frame_period': None,
        'line_period': None,
        'optical_zoom': None,
        'pmts': {},
        'preamp_filter': None,
        'preamp_offsets': {},
        'scan_mode': None,
        'bit_depth': None,
        }
    
    if fname is None:
        return data
    
    mdata_root = ElementTree.parse(fname).getroot()

    def state_value(key, type_fn=str):
        element = mdata_root.find(f'.//PVStateValue[@key="{key}"]')
        value = element.attrib['value']
        return type_fn(value)

    def indexed_value(key, index, type_fn=None, required=True, description=False):
        element = mdata_root.find(f'.//PVStateValue[@key="{key}"]/IndexedValue[@index="{index}"]')
        if element is None:
            if required:
                raise MetadataError('Could not find required key:index of %s:%s' % (key, index))
            return None
        
        value = element.attrib['value']
        
        if description:
            return type_fn(value), element.attrib['description']
        else:
            return type_fn(value)
        
    data['scan_mode'] = state_value('activeMode',str)
    data['bit_depth'] = state_value('bitDepth', int)

    sequences = mdata_root.findall('Sequence')
    
    data['layout']['sequences'] = len(sequences)

    # Frames/sequence should be constant, except for perhaps the last frame.
    data['layout']['frames_per_sequence'] = len(sequences[0].findall('Frame'))

    if data['layout']['sequences'] == 1:
        data['size']['frames'] = data['layout']['frames_per_sequence']
        data['size']['z_planes'] = 1
    else:
        # If the last sequence has a different number of frames, ignore it.
        num_frames_last_sequence = len(sequences[-1].findall('Frame'))
        if data['layout']['frames_per_sequence'] != num_frames_last_sequence:
            logging.warning('Skipping final stack because it was found with fewer z-planes (%d, expected: %d).',
                            num_frames_last_sequence, data['layout']['frames_per_sequence'])
            data['layout']['sequences'] -= 1
        data['size']['frames'] = data['layout']['sequences']
        data['size']['z_planes'] = data['layout']['frames_per_sequence']

    data['size']['channels'] = len(mdata_root.find('Sequence/Frame').findall('File'))
    data['size']['y_px'] = state_value('linesPerFrame', int)
    data['size']['x_pix'] = state_value('pixelsPerLine', int)
    
    data['layout']['samples_per_pixel'] = state_value('samplesPerPixel', int)
    
    data['pixel_size'] = {
             'x': indexed_value('micronsPerPixel', 'XAxis', float),
             'y': indexed_value('micronsPerPixel', 'YAxis', float),
             'z': indexed_value('micronsPerPixel', 'ZAxis', float),
             }
             
    for i in range(3):
        _elem = indexed_value('laserPower', i, float, required=True, description=True)
        data['laser_power'][_elem[1]]=_elem[0]

    data['frame_period'] = state_value('framePeriod', float)
    data['line_period'] = state_value('scanLinePeriod', float)
    
    data['optical_zoom'] = state_value('opticalZoom', float)
    
    data['pmts'] = {0: indexed_value('pmtGain', 0, float),
           1: indexed_value('pmtGain', 1, float)}
    
    data['preamp_filter'] = state_value('preampFilter')
    preamp_subindex = mdata_root.find('.//PVStateValue[@key="preampOffset"]/SubindexedValues[@index="0"]')
    data['preamp_offsets'] = {0: preamp_subindex.find('.//SubindexedValue[@subindex="0"]').attrib['value'],
                              1: preamp_subindex.find('.//SubindexedValue[@subindex="1"]').attrib['value']}
    
    return data
    
def read_vr_xml(fname):
    """_summary_

    Args:
        fname (_type_): _description_
    """
    data = {
        'channels': {},
        'acquisition_rate': None,
        'acquisition_time': None,
        'num_samples_acquired': None,
        'trigger': None,
        'trigger_count': None
    }
    
    if fname is None:
        return data
   
    voltage_root = ElementTree.parse(fname).getroot()

    for signal in voltage_root.findall('Experiment/SignalList/VRecSignal'):
        channel_num = int(signal.find('Channel').text)
        channel_name = signal.find('Name').text
        enabled = signal.find('Enabled').text == 'true'
        data['channels'][channel_num] = {'name': channel_name, 'enabled': enabled}
        
    data['acquisition_rate'] = int(voltage_root.find('Experiment/Rate').text) # Hz
    data['acquisition_time'] = int(voltage_root.find('Experiment/AcquisitionTime').text) # ms
    data['num_samples_acquired'] = int(voltage_root.find('SamplesAcquired').text)
    data['trigger'] = voltage_root.find('Experiment/Trigger').text
    data['trigger_count'] = int(voltage_root.find('Experiment/TriggerCount').text)
    
    return data

def read_mp_xml(fname):
    """

    Args:
        fname (_type_): _description_
    """
    data = {
        'iterations': None,
        'iteration_delay': None,
        'repetitions': None,
        'uncaging_laser': None,
        'trigger': None,
        'trigger_freq': None,
        'trigger_count': None,
        'uncaging_power': None,
        'custom_laser_power': None,
        'initial_delay': None,
        'inter_point_delay': None,
        'duration': None,
        'spiral_revolutions': None,
        'indices': None,
        'points_list': None,
        
    }
    
    if fname is None:
        return data
    
    markpoints_root = ElementTree.parse(fname).getroot()
    
    data['iterations'] = int(markpoints_root.attrib['Iterations'])
    data['iteration_delay'] = float(markpoints_root.attrib['IterationDelay'])
        
    mpe = markpoints_root.find('PVMarkPointElement').attrib
    data['repetitions'] = int(mpe['Repetitions'])
    data['uncaging_laser'] = mpe['UncagingLaser']
    data['trigger'] = mpe['TriggerSelection']
    data['trigger_freq'] = mpe['TriggerFrequency']
    data['trigger_count'] = int(mpe['TriggerCount'])
    data['uncaging_power'] = int(mpe['UncagingLaserPower'])
    data['custom_laser_power'] = [float(p)/16384.*1000 for p in mpe['CustomLaserPower'].split(',')]
    
    gpe = markpoints_root.find('PVMarkPointElement/PVGalvoPointElement').attrib
    data['initial_delay'] = float(gpe['InitialDelay'])
    data['inter_point_delay'] = float(gpe['InterPointDelay'])
    data['duration'] = float(gpe['Duration'])
    data['spiral_revolutions'] = int(gpe['SpiralRevolutions'])
    data['indices'] = gpe['Indices']
    
    data['points_list'] = [p.attrib for p in markpoints_root.findall('PVMarkPointElement/PVGalvoPointElement/Point')]
    
    for d in data['points_list']:
        for key, val in d.items():
            if key == 'IsSpiral':
                d[key] = bool(val)
            else:
                d[key] = float(val) 
                
    return data
    
    
    
def read(basename_input, dirname_output):
    """Read in metdata from XML files."""
    fname_xml = basename_input.with_suffix('.xml')
    fname_vr_xml = pathlib.Path(str(basename_input) + '_Cycle00001_VoltageRecording_001').with_suffix('.xml')
    fname_mp_xml = pathlib.Path(str(basename_input) + '_Cycle00001_MarkPoints').with_suffix('.xml')
    
    if not os.path.exists(fname_xml):
        raise MetadataError("metadata file %s does not exist"  % str(fname_xml))
    
    fname_metadata = dirname_output / 'metadata.json'

    logger.info('Extracting metadata from xml files:\n%s', fname_xml)
    scan_data = read_main_xml(fname_xml)
    
    if not os.path.exists(fname_vr_xml):
        vr_data = read_vr_xml(None)
        raise MetadataWarning("voltage recording file %s does not exist, skipping" % fname_vr_xml)
    else:
        logger.info('\n%s' % fname_vr_xml)
        vr_data = read_vr_xml(fname_vr_xml)
        
    if not os.path.exists(fname_mp_xml):
        mp_data = read_mp_xml(None)
        raise MetadataWarning("mark points file %s does not exist, skipping" % fname_mp_xml)
    else:
        mp_data = read_mp_xml(fname_mp_xml)
        
    
    metadata = {
        **scan_data,
        'voltage_recording': vr_data,
        'mark_points': mp_data,
        }
        
    with open(fname_metadata, 'w') as fout:
        json.dump(metadata, fout, indent=4, sort_keys=True)
        
    logger.info('Metadata written to: %s\n', fname_metadata)
    
    return metadata