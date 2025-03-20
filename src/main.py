from simulation import Simulation

def main():
    simulation_type = "1"  # Or "2" for client interaction
    simulation = Simulation.pick_simulation(simulation_type)
    simulation.run()

if __name__ == "__main__":
    main()
