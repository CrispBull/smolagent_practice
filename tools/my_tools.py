from smolagents import tool

@tool
def get_current_time_in_timezone(timeZone: str) -> str:
    """
    Returns the current local time for a given timezone.

    Args:
        timeZone: A valid timezone string (e.g., 'America/New_York')
    """
    import pytz, datetime
    try:
        tz = pytz.timezone(timeZone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current time in {timeZone} is {local_time}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def my_custom_tool(arg1: str, arg2: int) -> str:
    """
    A dummy tool for demonstration.

    Args:
        arg1: First argument
        arg2: Second argument
    """
    return f"Custom tool called with arg1={arg1} and arg2={arg2}"

