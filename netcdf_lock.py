"""
One lock for every NetCDF read in the server.

The netCDF-C library under xarray is not thread-safe, and FastAPI runs each
request in a thread of its own. When a page asks for several fields at once,
two threads can open files together and the library aborts the whole process
("NClist failure"). Every read therefore takes this lock: reads are short and
their results are cached, so serialising them costs little. Re-entrant, so a
reader that calls another reader does not deadlock.
"""

import threading

NETCDF_LOCK = threading.RLock()
