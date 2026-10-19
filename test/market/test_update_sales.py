from datetime import datetime, UTC
from google.protobuf.json_format import ParseDict
import pytest
from surimi.v1.update_sales_pb2 import UpdateSalesRequest

@pytest.fixture()
def update_request_01() -> UpdateSalesRequest:
    data = {
        "simulation_id": "aaaa",
        "start_date_time": "2010-01-01T00:00:00Z",
        "end_date_time": "2100-01-01T00:00:00Z",
        "sales_summary": {
            "market_sales": [
                {
                    "market_code": "PORTA",
                    "currency": "EUR",
                    "sales": [
                        {
                            "species": { "species_code": "SPECIES_A" },
                            "quantity": 1,
                            "category_code": "PriceCategory_A"
                        },
                        {
                            "species": { "species_code": "SPECIES_A" },
                            "quantity": 1,
                            "category_code": "PriceCategory_A"
                        }
                    ]
                }
            ]
        }
    }

    return ParseDict(data, UpdateSalesRequest())

@pytest.fixture()
def update_request_02() -> UpdateSalesRequest:
    """
    This also tests whether floating point numbers are correctly registered.
    """
    data = {
        "simulation_id": "aaaa",
        "start_date_time": "2010-01-01T00:00:00Z",
        "end_date_time": "2100-01-01T00:00:00Z",
        "sales_summary": {
            "market_sales": [
                {
                    "market_code": "PORTA",
                    "currency": "EUR",
                    "sales": [
                        {
                            "species": { "species_code": "SPECIES_A" },
                            "quantity": 2.3,
                            "category_code": "PriceCategory_A"
                        },
                        {
                            "species": { "species_code": "SPECIES_B" },
                            "quantity": 1,
                            "category_code": "PriceCategory_A"
                        }
                    ]
                }
            ]
        }
    }

    return ParseDict(data, UpdateSalesRequest())

@pytest.fixture()
def update_request_03() -> UpdateSalesRequest:
    """
    Update multiple ports in a single update request
    """
    data = {
        "simulation_id": "aaaa",
        "start_date_time": "2010-01-01T00:00:00Z",
        "end_date_time": "2100-01-01T00:00:00Z",
        "sales_summary": {
            "market_sales": [
                {
                    "market_code": "PORTA",
                    "currency": "EUR",
                    "sales": [
                        {
                            "species": { "species_code": "SPECIES_A" },
                            "quantity": 1,
                            "category_code": "PriceCategory_A"
                        }
                    ]
                },
                {
                    "market_code": "PORTB",
                    "currency": "EUR",
                    "sales": [
                        {
                            "species": { "species_code": "SPECIES_A" },
                            "quantity": 1,
                            "category_code": "PriceCategory_A"
                        }
                    ]
                }
            ]
        }
    }

    return ParseDict(data, UpdateSalesRequest())

@pytest.fixture()
def update_future_landings_01() -> UpdateSalesRequest:
    data = {
        "simulation_id": "aaaa",
        "start_date_time": "2011-01-01T00:00:00Z",
        "end_date_time": "2100-01-01T00:00:00Z",
        "sales_summary": {
            "market_sales": [
                {
                    "market_code": "PORTA",
                    "currency": "EUR",
                    "sales": [
                        {
                            "species": { "species_code": "SPECIES_A" },
                            "quantity": 1,
                            "category_code": "PriceCategory_A"
                        },
                        {
                            "species": { "species_code": "SPECIES_A" },
                            "quantity": 2,
                            "category_code": "PriceCategory_A"
                        },
                    ]
                },
                {
                    "market_code": "PORTB",
                    "currency": "EUR",
                    "sales": [
                        {
                            "species": { "species_code": "SPECIES_A" },
                            "quantity": 1,
                            "category_code": "PriceCategory_A"
                        },
                    ]
                }
            ]
        }
    }

    return ParseDict(data, UpdateSalesRequest())

@pytest.fixture()
def update_future_landings_02() -> UpdateSalesRequest:
    """
    Updating adding to an already updated landings.
    """

    data = {
        "simulation_id": "aaaa",
        "start_date_time": "2011-01-01T00:00:00Z",
        "end_date_time": "2100-01-01T00:00:00Z",
        "sales_summary": {
            "market_sales": [
                {
                    "market_code": "PORTA",
                    "currency": "EUR",
                    "sales": [
                        {
                            "species": { "species_code": "SPECIES_A" },
                            "quantity": 1,
                            "category_code": "PriceCategory_A"
                        }
                    ]
                },
                {
                    "market_code": "PORTB",
                    "currency": "EUR",
                    "sales": [
                        {
                            "species": { "species_code": "SPECIES_A" },
                            "quantity": 1,
                            "category_code": "PriceCategory_A"
                        }
                    ]
                },
                {
                    "market_code": "PORTC",
                    "currency": "EUR",
                    "sales": [
                        {
                            "species": { "species_code": "SPECIES_A" },
                            "quantity": 1,
                            "category_code": "PriceCategory_A"
                        }
                    ]
                }
            ]
        }
    }

    return ParseDict(data, UpdateSalesRequest())

@pytest.fixture()
def update_future_landings_03() -> UpdateSalesRequest:
    """
    Add another month with pending changes
    """

    data = {
        "simulation_id": "aaaa",
        "start_date_time": "2011-02-01T00:00:00Z",
        "end_date_time": "2100-01-01T00:00:00Z",
        "sales_summary": {
            "market_sales": [
                {
                    "market_code": "PORTA",
                    "currency": "EUR",
                    "sales": [
                        {
                            "species": { "species_code": "SPECIES_A" },
                            "quantity": 1,
                            "category_code": "PriceCategory_A"
                        }
                    ]
                },
                {
                    "market_code": "PORTB",
                    "currency": "EUR",
                    "sales": [
                        {
                            "species": { "species_code": "SPECIES_A" },
                            "quantity": 1,
                            "category_code": "PriceCategory_A"
                        }
                    ]
                },
                {
                    "market_code": "PORTC",
                    "currency": "EUR",
                    "sales": [
                        {
                            "species": { "species_code": "SPECIES_A" },
                            "quantity": 1,
                            "category_code": "PriceCategory_A"
                        }
                    ]
                }
            ]
        }
    }

    return ParseDict(data, UpdateSalesRequest())

def test_update_non_existent_simulation(contract, grpc_context):
    # market_service
    pass

def test_accumulate_landings_for_the_current_month(contract,
                                                   market_service,
                                                   update_request_01,
                                                   update_request_02,
                                                   update_request_03):
    market_system = market_service \
                    .simulation_dictionary \
                    .get(contract.simulation_id) \
                    .market_system

    market_system.update_sales(update_request_01.sales_summary.market_sales,
                               update_request_01.start_date_time.seconds)

    # There is a flaw in the model where the quantity is not persistent across
    # the timeline. As a result, the catch is reset on each stepå

    snapshot_01 = market_system.local_markets["PORTA"].get_monthly_catch_snapshot()
    landings_01 = snapshot_01[("SPECIES_A", "PriceCategory_A")]
    assert landings_01 == 2, \
        "Local market with updated landings for the current timestep should" \
        "store this number of landings (test: 2kg)."
    
    # -- EDGE CASE --
    non_registered_landings = snapshot_01.get(("SPECIES_A", "PriceCategory_B"))
    assert not non_registered_landings, \
        "Port should not update landings for non-updated price identifier"
    
    # Send another update request
    market_system.update_sales(update_request_02.sales_summary.market_sales,
                               update_request_02.start_date_time.seconds)
    snapshot_02 = market_system.local_markets["PORTA"].get_monthly_catch_snapshot()
    landings_02a = snapshot_02[("SPECIES_A", "PriceCategory_A")]
    assert landings_02a == 4.3, \
        "Sending another update request to an already update landings should" \
        "add the new landings"
    
    landings_02b = snapshot_02[("SPECIES_B", "PriceCategory_A")]
    assert landings_02b == 1, \
        "Sending another update request for yet an unregistered price" \
        "identifier should result in the new landings only (test: 1 kg)."
    
    # Send another request, but update two different ports this time.
    market_system.update_sales(update_request_03.sales_summary.market_sales,
                               update_request_03.start_date_time.seconds)
    landings_03a = market_system.local_markets["PORTA"] \
                      .get_monthly_catch_snapshot() \
                      .get(("SPECIES_A", "PriceCategory_A"))
    assert landings_03a == 5.3, \
        "Updating an existing landings for the current month should add the" \
        "new landings"
    
    landings_03b = market_system.local_markets["PORTB"] \
                      .get_monthly_catch_snapshot() \
                      .get(("SPECIES_A", "PriceCategory_A"))
    assert landings_03b and landings_03b == 1.0, \
        "Updating a non-updated port for this step should set the new" \
        "landings as the current quantity"
    
def test_update_future_landings(contract,
                                market_service,
                                update_future_landings_01,
                                update_future_landings_02,
                                update_future_landings_03):

    market_system = market_service \
                .simulation_dictionary \
                .get(contract.simulation_id) \
                .market_system

    market_system.update_sales(
        update_future_landings_01.sales_summary.market_sales,
        update_future_landings_01.start_date_time.seconds
    )

    current_landings = market_system \
                       .local_markets.get("PORTA") \
                       .get_monthly_catch_snapshot() \
                       .get(("SPECIES_A", "PriceCategory_A"))
    assert not current_landings, \
        "Updating future landings should not affect the current landings."


    pending_sales_months = market_system.pending_sales_summary
    assert len(pending_sales_months) == 1, \
        "Updating changes for one future month should result in one month of " \
        "pending changes."

    month = list(pending_sales_months.keys())[0]
    num_ports = len(*pending_sales_months[month])
    sales_port_a = list(*pending_sales_months[month])[0].sales
    sales_port_b = list(*pending_sales_months[month])[1].sales

    assert num_ports == 2, \
        "Updating 2 different ports in the future, but registered ports " \
        "is not equivalent to 2."
    
    assert len(sales_port_a) == 2, \
        "Updating future PORTA with 2 different species identifiers, but " \
        "exactly 2 were not registered."

    assert len(sales_port_b) == 1, \
        "Updating future PORTB with 1 species identifier, but " \
        "exactly 1 was not registered."
    
    # --------------------------------------------------------------------------

    # Add to an already updated landings. This should add a new pending update
    # to the existing date.
    market_system.update_sales(
        update_future_landings_02.sales_summary.market_sales,
        update_future_landings_02.start_date_time.seconds
    )
    pending_sales_months = market_system.pending_sales_summary
    assert len(pending_sales_months) == 1, \
        "Adding another pending change to the same month should keep the same" \
        "number of pending change months"
    
    month = list(pending_sales_months.keys())[0]
    num_ports = len(pending_sales_months[month][-1])
    assert num_ports == 3, \
        "Adding 3 different ports with pending changes should register these " \
        "3 ports"
    
    # --------------------------------------------------------------------------

    # Add another month with pending changes
    market_system.update_sales(
        update_future_landings_03.sales_summary.market_sales,
        update_future_landings_03.start_date_time.seconds
    )
    pending_sales_months = market_system.pending_sales_summary
    months = list(pending_sales_months.keys())
    assert len(months) == 2, \
        "Adding another pending change to the same month should keep the same" \
        "number of pending change months"
    