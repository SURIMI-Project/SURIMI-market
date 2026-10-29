import pytest
from src.models.elsa.ElsaHandler import find_missing_ports, \
                                  get_valid_ports_only, \
                                  find_missing_species, \
                                  get_valid_species_only

@pytest.fixture
def contract_ports():
    return {"AAA", "BBB", "CCC"}

@pytest.fixture
def market_ports():
    return {"AAA", "CCC", "DDD"}

def test_find_missing_ports(contract_ports, market_ports):
    missing_ports = find_missing_ports(contract_ports, market_ports)

    assert len(missing_ports) > 0, \
           f"Expected {len(contract_ports)} missing ports, but no missing ports were found"
    assert missing_ports == {"BBB"}, \
           "Expected missing port 'BBB', but it was not found"
    assert missing_ports != {"BBB", "CCC"}, \
           f"Expected missing ports 'BBB', but 'CCC' was also found"

def test_get_valid_ports(contract_ports, market_ports):
    valid_ports = get_valid_ports_only(market_ports, contract_ports)

    assert len(valid_ports) > 0, f"Expected 1 port, but None were found"
    assert valid_ports == {"AAA", "CCC"}, \
           f"Expected missing port 'AAA', but it was not found"