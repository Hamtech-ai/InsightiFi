def zigzag(s, c):

    zigzagList = []
    zz = []
    zigzagLabel = {}
    signal = 0
    inflection = s[0]
    inflection_index = 0
    
    for i in range(1, len(s)):
        # Find first trend
        if signal == 0:
            if s[i] <= (inflection - c):
                signal = -1
            elif s[i] >= (inflection + c):
                signal = 1

        # Downtrend, inflection keeps track of the lowest point in the downtrend
        if signal == -1:
            # New Minimum, change inflection
            if s[i] < inflection:
                inflection = s[i]
                inflection_index = i
                
            # Trend Reversal
            elif s[i] >= (inflection + c):
                signal = 1
                zz.append(inflection)  # Append the lowest point of the downtrend to zz
                zigzagLabel[inflection_index] = -1
                inflection = s[i]
                inflection_index = i      # Current point becomes the highest point of the new uptrend

        # Uptrend, inflection keeps track of the highest point in the uptrend
        elif signal == 1:
            # New Maximum, change inflection
            if s[i] > inflection:
                inflection = s[i]
                inflection_index = i
                
            # Trend Reversal
            elif s[i] <= (inflection - c):
                signal = -1
                zz.append(inflection)  # Append the highest point of the uptrend to zz
                zigzagLabel[inflection_index] = 1
                inflection = s[i]
                inflection_index = i      # Current point becomes the lowest point of the new trend
        
    for i in range(len(s)):
        if i not in zigzagLabel:
            zigzagList.append(0)
        else:
            zigzagList.append(zigzagLabel[i])

    return zigzagList