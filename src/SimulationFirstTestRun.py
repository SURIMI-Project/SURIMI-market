from Market import Market, Species, Tick
# simulation_run.py
if __name__ == "__main__":
    market = Market("Global Seafood Market", "EUR", "kg")

    market.add_port("Barcelona", "ESBCN")  # Barcelona, Spain
    market.add_port("Valencia", "ESVLC")  # Valencia, Spain

    market.add_species("Tuna", "TUN")
    market.add_species("Salmon", "SAL")

    for _ in range(10):
        Tick.increase_tick()
        messages = market.send_update_prices_message()

        print(f"Day {Tick.get_tick()} - {Tick.get_timestamp()}")

        for msg in messages:
            print(msg)

        print("-" * 50)

    # Plot fish prices over time
    market.action_log.plot_data()