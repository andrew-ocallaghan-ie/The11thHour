'''
Created on 11 Jun 2017

@author: Andy
'''

def hello_team ( name ):
    try:
        return "Hello " + name
    except:
        return "argument wasn't a string!"

    
if __name__ == '__main__':
    print ( hello_team( "11th Hour" ) )