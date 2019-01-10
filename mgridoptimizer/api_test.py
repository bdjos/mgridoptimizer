from mgridoptimizer.modules.mgrid_model import Demand, Battery, Converter, Controller, Solar, Grid, System_Model
import os
import json

def api_sim(system_input, demand_file):
    # Project Specs
    project_years = 20
    data = json.loads(system_input)

    # Map component type to model component object
    component_mapping = {
                     'demand':  Demand.fifteenMinute,
                     'battery': Battery,
                     'solar': Solar.run_api,
                     'controller': Controller,
                     'converter': Converter,
                     # 'generator': Generator,
                     'grid': Grid
    }


    # Define system model
    system = System_Model()

    # Create components from JSON data and add to data dict
    data_objects = {}
    for component in data['components']:
         data_objects[component] = component_mapping[data['components'][component]['comp_type']](**data['components'][component]['input'])
         system.add_component(data_objects[component], component, str(data['components'][component]['zone']))

    # Add controller components in zone 1 to controller
    for component in data['components']:
        if data['components'][component]['configure']:
             data_objects['cnt1'].config_component(
                 data_objects[component],
                 data['components'][component]['comp_num'],
                 data['components'][component]['configure']['mode']
            )

    system.simulate()

    for component in system.system_components:
         data['components'][component]['output'] = system.system_components[component].output()

    print(json.dumps(data))
    return data

def test_api():
    demand_file = os.path.join(os.path.dirname(__file__), 'data/test_data.csv')
    system_input = {
        'components': {
                    'dmn1': {
                        'id': 1,
                        'system_name_id': 1,
                        'comp_type': 'demand',
                        'comp_num': 1,
                        'zone': 0,
                        'input': {
                            'file': demand_file
                        },
                        'configure': {}
                    },
                    'slr1': {
                        'id': 2,
                        'system_name_id': 1,
                        'comp_type': 'solar',
                        'comp_num': 2,
                        'zone': 0,
                        'input': {
                            'system_capacity': 500,
                            'base_cost': 10000,
                            'perw_cost': 1.5
                        },
                        'configure': {}
                    },
                    'btt1': {
                        'id': 3,
                        'system_name_id': 1,
                        'comp_type': 'battery',
                        'comp_num': 3,
                        'zone': 1,
                        'input': {
                             'energy_capacity': 500,
                             'soc_min': 20,
                             'soc_max': 80,
                             'efficiency': 0.98,
                             'base_cost': 10000,
                             'energy_cost': 1.5
                        },
                        'configure': {
                            'configured': True,
                            'mode': 'ss'
                        }
                    },
                    'btt2': {
                        'id': 4,
                        'system_name_id': 1,
                        'comp_type': 'battery',
                        'comp_num': 4,
                        'zone': 1,
                        'input': {
                            'energy_capacity': 800,
                            'soc_min': 20,
                            'soc_max': 80,
                            'efficiency': 0.98,
                            'base_cost': 15000,
                            'energy_cost': 1.7
                        },
                        'configure': {
                            'configured': True,
                            'mode': 'ss'
                        }
                    },
                    'cnv1': {
                        'id': 5,
                        'system_name_id': 1,
                        'comp_type': 'converter',
                        'comp_num': 5,
                        'zone': 1,
                        'input': {
                            'power': 800,
                            'base_cost': 10000,
                            'power_cost': 1.5
                        },
                        'configure': {
                            'configured': True,
                            'mode': 'ss'
                        }
                    },
                    'cnt1': {
                        'id': 6,
                        'system_name_id': 1,
                        'comp_type': 'controller',
                        'comp_num': 6,
                        'zone': 1,
                        'input': {'active': True},
                        'configure': {}
                    },
                    'grd1': {
                        'id': 7,
                        'system_name_id': 1,
                        'comp_type': 'grid',
                        'comp_num': 7,
                        'zone': 2,
                        'input': {
                            'energy_cost': 0.29,
                            'nm': False
                        },
                        'configure': {}
                    }
        }
    }

    system_input = json.dumps(system_input)

    system_output = api_sim(system_input, demand_file)

    return system_output

if __name__ == "__main__":
    test_api()