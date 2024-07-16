import os
import termcolor
import sys

def toRed(content):
    return termcolor.colored(content,"red",attrs=["bold"])

def toGreen(content):
    return termcolor.colored(content,"green",attrs=["bold"])

def toBlue(content):
    return termcolor.colored(content,"blue",attrs=["bold"])

def toCyan(content):
    return termcolor.colored(content,"cyan",attrs=["bold"])

def toYellow(content):
    return termcolor.colored(content,"yellow",attrs=["bold"])

def toMagenta(content):
    return termcolor.colored(content,"magenta",attrs=["bold"])

def toGrey(content):
    return termcolor.colored(content,"grey",attrs=["bold"])

def toWhite(content):
    return termcolor.colored(content,"white",attrs=["bold"])

def print_alert(state, string): 
    CURSOR_UP_ONE = '\x1b[1A' 
    ERASE_LINE = '\x1b[2K'
    sys.stdout.write('{}[{}] {}\n'.format(
      CURSOR_UP_ONE + ERASE_LINE,
      toWhite(state),
      toYellow(string)
    ))

def print_logs(state, epoch, max_epoch, iter, max_iter, time, loss, name, eval=None, evalname=None, eval2=None, evalname2=None):
    CURSOR_UP_ONE = '\x1b[1A' 
    ERASE_LINE = '\x1b[2K'
    sys.stdout.write('{}[{}]{} {} {} {} {} {} {} {} {} {}\n'.format(
      CURSOR_UP_ONE + ERASE_LINE,
      toWhite(state),
      toWhite(' {} '.format('EP')) + toCyan('{}/{}'.format(epoch, max_epoch)),
      toWhite('ITER ') + toCyan('{}/{}'.format(iter, max_iter)),
      toGreen('{:5.2f}s'.format(time)),
      toGreen('(eta {:5.2f}s)'.format(time*(max_iter-iter))),
      toWhite(name),
      toYellow('{:.3f}'.format(loss)),
      toWhite(evalname) if evalname is not None else "",
      toYellow('{:.3f}'.format(eval) if eval is not None else ""),
      toWhite(evalname2) if evalname2 is not None else "",
      toYellow('{:.3f}'.format(eval2) if eval2 is not None else "")
      )
    )