from data import preprocess
from model import model
def main(data):

  '''
  The Main function to automate the processess

  '''

  M = model(*preprocess(data))
  M.logit()
  M.XGB()
  M.MLP()

  if __name__ == 'main':
    main()