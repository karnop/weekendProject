import json

def validateAndTrimBrackets(text : str) -> tuple[bool, str] : 
    text = text.replace('\n', '')

    firstBracketIndex = text.find('{')
    lastBracketIndex = text.rfind('}')

    if firstBracketIndex == -1 or lastBracketIndex == -1 or firstBracketIndex >lastBracketIndex: 
        return False, {}
    
    # trimming the string to only include content between the brace
    trimmedString = text[firstBracketIndex:lastBracketIndex+1]

    # validate the bracket balance using a cntr
    balance = 0
    for char in trimmedString:
        if char == '{' :
            balance+=1
        elif char == '}' :
            balance -=1

        if balance < 0:
            return False, {}
        
    isValid = balance == 0
    return isValid, json.loads(trimmedString)


        