from modules.mgrid_model import Demand, Battery, Converter, Controller, Solar, Grid, System_Model
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
    for component in data['components']:
        component['object'] = component_mapping[component.comp_type](**component.input)
        system.addcomponent(component['object'])

    # Add controller components in zone 1 to controller
    for component in data['components']:
        if component['configure']['configured']:
            data['components']['cnt1']['object'].config_component(
                component['object'],
                **component['objects'][]
            )

    system.simulate()

    print(system)

if __name__ == "main":
    demand_file = 'data/test_data.csv'
    system_input = {
        'components': {
                    'dmn1': {
                        'id': 1,
                        'system_name_id': 1,
                        'comp_type': 'demand',
                        'comp_num': 1,
                        'zone': 0,
                        'input': {
                            'file': 'test_demand.csv'
                        },
                        'configure': {}
                    },
                    'slr1': {
                        'id': 1,
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
                        'id': 1,
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
                        'id': 1,
                        'system_name_id': 1,
                        'comp_type': 'battery',
                        'comp_num': 4,
                        'zone': 0,
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
                        'id': 1,
                        'system_name_id': 1,
                        'comp_type': 'converter',
                        'comp_num': 5,
                        'zone': 0,
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
                        'id': 1,
                        'system_name_id': 1,
                        'comp_type': 'controller',
                        'comp_num': 6,
                        'zone': 0,
                        'input': {},
                        'configure': {}
                    },
                    'grd1': {
                        'id': 1,
                        'system_name_id': 1,
                        'comp_type': 'grid',
                        'comp_num': 7,
                        'zone': 0,
                        'input': {
                            'energy_cost': 0.29,
                            'nm': False
                        },
                        'configure': {}
                    }
        }
    }

    api_sim(system_input, demand_file)