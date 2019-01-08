from modules.mgrid_model import Demand, Battery, Converter, Controller, Solar, Generator, Grid, System_Model

def api_sim(input):
    # Project Specs
    project_years = 20
    data = json.loads(input)

    component_mapping = {'demand':  Demand,
                         'battery': Battery,
                         'solar': Solar,
                         'controller': Controller,
                         'converter': Converter,
                         'generator': Generator,
                         'grid': Grid
                         }

    # Create components from JSON data
    component_objects = data['components']
    for component in data['components']:
        component_objects.append(component_mapping[component.comp_type](component))

    system = System_Model()

if __name__ == "main":
    input = {
        'components': {
                    'bat1': {
                         'energy_capacity': 500,
                         'soc_min': 20,
                         'soc_max': 80,
                         'efficiency': 0.98,
                         'base_cost': 10000,
                         'energy_cost': 1.5
                    }
                    'bat2': {
                        'energy_capacity': 500,
                        'soc_min': 20,
                        'soc_max': 80,
                        'efficiency': 0.98,
                        'base_cost': 10000,
                        'energy_cost': 1.5
                    }
        }
    }

    api_sim()