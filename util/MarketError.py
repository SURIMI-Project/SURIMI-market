from grpc import StatusCode

class MarketError:
    """
    The MarketError class is used as a helper to perform explicit error
    handling. Its intended use case is to let higher level layers backpropagate
    errors to the routing layer. E.g., the SimulationHandler can use this to
    send errors to client via the MarketService (routing layer).
    
    One can consider this type of error handling a pattern. In
    essence, it's about explicitly checking defined cases for errors. E.g.:

    ```
    def add(a: float, b: float) -> float:
        return a + b
    ```

    Let's suppose that it is crucial for the program to validate that a and b
    indeed are floats, as the type hint does not enforce the interpreter to
    perform this check. One approach is to use exceptions:

    ```
    def add(a: float, b: float) -> float:
        if (type(a) != float) and (type(b) != float):
            raise TypeError(f"Arguments {a} and/or {b} is not a float.")
        else:
            return a + b
    
    try:
        val = add(3.14, 4)
    except TypeError as e:
        print(e)
    except Exception as e:
        print(e)
    ```

    The problem is we don't necessarily know what exception add will throw, and
    explicitly catching every exception may lead to us missing some errors. The
    last except in the code snippet above catches this, but it is not ideal if
    that ever happens. Instead, the `MarketError` class promotes the following:

    ```
    def add(a: float, b: float) -> tuple[float, MarketError]:
        if (type(a) != float) and (type(b) != float):
            raise None, MarketError(f"Arguments {a} and/or {b} is not a float.")
        else:
            return a + b, None
    
    val, err = add(3.14, 4)
    if err != None:
        print(err)
    else:
        print(val)
    ```

    This pattern is inspired by Go and is typically also used there. It promotes
    readability, and makes debugging easier as it is obvious what the program
    flow is.
    """
    def __init__(self, status_code: StatusCode, message: str):
        self.status_code = status_code
        self.message = message
