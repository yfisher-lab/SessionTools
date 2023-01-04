"""Library for parsing metadata files from Bruker scope. Modified from Deisseroth lab code"""

import json
import logging
import pathlib
import pprint
from xml.etree import ElementTree

logger = logging.getLogger(__file__)


class MetadataError(Exception):
    """Error while extracting metadata."""


def read(basename_input, dirname_output):
    """Read in metdata from XML files."""
    fname_xml = basename_input.with_suffix('.xml')
    fname_vr_xml = pathlib.Path(str(basename_input) + '_Cycle00001_VoltageRecording_001').with_suffix('.xml')
    fname_metadata = dirname_output / 'metadata.json'

    logger.info('Extracting metadata from xml files:\n%s\n%s', fname_xml, fname_vr_xml)

    mdata_root = ElementTree.parse(fname_xml).getroot()

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
        
        

    sequences = mdata_root.findall('Sequence')
    num_sequences = len(sequences)

    # Frames/sequence should be constant, except for perhaps the last frame.
    num_frames_per_sequence = len(sequences[0].findall('Frame'))

    if num_sequences == 1:
        num_frames = num_frames_per_sequence
        num_z_planes = 1
    else:
        # If the last sequence has a different number of frames, ignore it.
        num_frames_last_sequence = len(sequences[-1].findall('Frame'))
        if num_frames_per_sequence != num_frames_last_sequence:
            logging.warning('Skipping final stack because it was found with fewer z-planes (%d, expected: %d).',
                            num_frames_last_sequence, num_frames_per_sequence)
            num_sequences -= 1
        num_frames = num_sequences
        num_z_planes = num_frames_per_sequence

    num_channels = len(mdata_root.find('Sequence/Frame').findall('File'))
    num_y_px = state_value('linesPerFrame', int)
    num_x_px = state_value('pixelsPerLine', int)
    
    px_sz = {
             'x': indexed_value('micronsPerPixel', 'XAxis', float),
             'y': indexed_value('micronsPerPixel', 'YAxis', float),
             'z': indexed_value('micronsPerPixel', 'ZAxis', float),
             }
             
    lasers = {}
    for i in range(3):
        _elem = indexed_value('laserPower', i, float, required=True, description=True)
        lasers[_elem[1]]=_elem[0]

    frame_period = state_value('framePeriod', float)
    optical_zoom = state_value('opticalZoom', float)
    
    pmt = {0: indexed_value('pmtGain', 0, float),
           1: indexed_value('pmtGain', 1, float)}
    
    preamp_filter = state_value('preAmpFilter')
    preamp_subindex = mdata_root.find('.//PVStateValue[@key="preampOffset"]/SubindexedValues[@index="0"]')
    preampOffsets = {0: preamp_subindex.find('.//SubindexedValue[@subindex=0]').attrib['value'],
                     1: preamp_subindex.find('.//SubindexedValue[@subindex=1]').attrib['value']}
    
    
    ####

    voltage_root = ElementTree.parse(fname_vr_xml).getroot()

    channels = {}
    for signal in voltage_root.findall('Experiment/SignalList/VRecSignal'):
        channel_num = int(signal.find('Channel').text)
        channel_name = signal.find('Name').text
        enabled = signal.find('Enabled').text == 'true'
        channels[channel_num] = {'name': channel_name, 'enabled': enabled}
        
    vr_rate = int(voltage_root.find('Experiment/Rate').text) # Hz
    vr_acq_time = int(voltage_root.find('Experiment/AcquisitionTime').text) # ms
    num_samp_acqd = int(voltage_root.find('SamplesAcquired').text)

    
    ####
    markpoints_root = ElementTree.parse(fname_mp_xml).getroot()
    
    markpoints_dict = {}
    for k,v in markpoints_root.attrib:
        markpoints_dict[k] = v
        
    mpe = markpoints_root.find('PVMarkPointElement').attrib
    markpoints_dict['repetitions'] = int(mpe['Repetitions'])
    markpoints_dict['uncaging_laser'] = mpe['UncagingLaser']
    markpoints_dict['trigger'] = mpe['TriggerSelection']
    markpoints_dict['trigger_freq'] = mpe['TriggerFrequency']
    markpoints_dict['trigger_count'] = int(mpe['TriggerCount'])
    markpoints_dict['uncaging_power'] = int(mpe['UncagingLaserPower'])
    markpoints_dict['custom_laser_power'] = [float(p)/16384.*1000 for p in mpe['CustomLaserPower'].split(',')]
    markpoints_dict['points_list'] = [p.attrib for p in markpoints_root.findall('PVMarkPointElement/PVGalvoPointElement/Point')]
    
    
    
    metadata = {
        'layout': {
            'sequences': num_sequences,
            'frames_per_sequence': num_frames_per_sequence,
        },
        'size': {
            'frames': num_frames,
            'channels': num_channels,
            'z_planes': num_z_planes,
            'y_px': num_y_px,
            'x_px': num_x_px
        },
        'pixel_dims': px_sz,
        'laser_power': lasers,
        'period': frame_period,
        'optical_zoom': optical_zoom,
        'channels': channels,
    }

    with open(fname_metadata, 'w') as fout:
        json.dump(metadata, fout, indent=4, sort_keys=True)

    logger.info('The following metadata is written to: %s\n%s', fname_metadata, pprint.pformat(metadata))
    return metadata