class sdssQuery:
    """
    NAME: sdssQuery
 
    PURPOSE: class that can be initialized using Python's urllib tools to
    send an SQL command to SDSS web services
 
    CALLING SEQUENCE: from the UNIX command line:
      
      python sdssDR9query.py ra dec

    INPUTS: ra and dec shoud be sent from the command line:

      ra - Right Ascension of position to query around in SDSS DR9
      dec - declination of position to query around in SDSS DR9

    OUTPUTS: the result of the SQL command called "query" in the
    code, below, is executed by the SDSS DR9 SQL API and
    printed at the command line.
    
    COMMENTS: This is adapted from an example provide by the SDSS 
    in an early tutorial on web services.

    Note that the SQL command passed as "query" can be changed to
    any valid SDSS string.
 
    EXAMPLES: At the Unix command line:

      python sdssDR9query.py 145.285854 34.741254
      
    should return:

      145.28585385,34.74125418,21.132605,20.059248,19.613457,19.379345,19.423323,7.7489487E-4
    """
    # ADM this is the URL of the SDSS "web services" API.
    url='http://skyserver.sdss3.org/dr9/en/tools/search/x_sql.asp'
    # ADM always return the output in .csv format
    format = 'csv'

    # ADM initialize the class with a null query.
    def __init__(self):
        self.query = ''
        self.cleanQuery = ''

    # ADM use Python's urllib module to initialize a query string.
    def executeQuery(self):
        from urllib.parse import urlencode
        from urllib.request import urlopen
        self.filterQuery()
        params = urlencode({'cmd': self.cleanQuery, 'format':self.format})
        return urlopen(self.url + '?%s' % params)

    # ADM this cleans up the syntax in the query string.
    def filterQuery(self):
        from os import linesep
        self.cleanQuery = ''
        tempQuery = self.query.lstrip()
        for line in tempQuery.split('\n'):
            self.cleanQuery += line.split('--')[0] + ' ' + linesep;


if __name__ == '__main__':
    from time import sleep
    from argparse import ArgumentParser

    # ADM set up the inputs, with useful help information.
    ap = ArgumentParser(description='Query the SDSS database for objects \
    within 1.2" of an RA/Dec location')
    ap.add_argument("ra", help='Right Ascension (degrees)')
    ap.add_argument("dec", help='Declination (degrees)')

    # ADM store the input RA/Dec in ns.ra/ns.dec.
    ns = ap.parse_args()

    # ADM initialize the query.
    qry = sdssQuery()

    # ADM the query to be executed. You can substitute any query, here!
    query = """SELECT top 1 ra,dec,u,g,r,i,z,GNOE.distance*60 FROM PhotoObj as PT
    JOIN dbo.fGetNearbyObjEq(""" + ns.ra + """,""" + ns.dec + """,0.02) as GNOE
    on PT.objID = GNOE.objID ORDER BY GNOE.distance"""

    # ADM execute the query.
    qry.query = query
    for line in qry.executeQuery():
        result = line.strip()

    # ADM NEVER remove this line! It won't speed up your code, it will
    # ADM merely overwhelm the SDSS server (a denial-of-service attack)!
    sleep(1)

    # ADM the server returns a byte-type string. Convert it to a string.
    print(result.decode())
