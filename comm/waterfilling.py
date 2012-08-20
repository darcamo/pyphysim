#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def doWF(vtChannels, dPt, noiseVar=1.0, Es=1.0):
    """Performs the Waterfilling algorithm and returns the optimum power and water level.

    Arguments:
    - `vtChannels`: Numpy array with the channel POWER gains (power of the
                    parallel AWGN channels).
    - `dPt`: Total available power.
    - `noiseVar`: Noise variance (power in linear scale)
    - `Es`: Symbol energy (in linear scale)
    """
    ## Sort Channels (descending order)
    vtChannelsSortIndexes = np.argsort(vtChannels)[::-1]
    vtChannelsSorted = vtChannels[vtChannelsSortIndexes]

    #####################################################################
    # Calcula o waterlevel que toca o pior canal (o mais alto) e       #
    # portanto que transmite potencia 0 no pior canal.  Depois disso    #
    # calcula a potencia em cada canal (o vetor Ps) para esse           #
    # waterlevel.  Se a soma dessas potencias for menor do que a       #
    # portencia total entao e so dividir a potencia restante igualmente #
    # entre todos os canais (aumentar o waterlevel). Caso contrario,   #
    # removo o pior canal e repito o processo.                          #
    #####################################################################
    # Calculates minimum waterlevel $\mu$ required to use all channels
    dNChannels = vtChannels.size
    dRemoveChannels = 0

    minMu = float(noiseVar) / (
        Es * vtChannelsSorted[dNChannels - dRemoveChannels - 1])
    Ps = (minMu - float(noiseVar) / (
            Es * vtChannelsSorted[np.arange(0, dNChannels - dRemoveChannels)]))

    while (sum(Ps) > dPt) and (dRemoveChannels < dNChannels):
        dRemoveChannels = dRemoveChannels + 1
        minMu = float(noiseVar) / (
            Es * vtChannelsSorted[dNChannels - dRemoveChannels - 1])
        Ps = (minMu - float(noiseVar) / (
                Es * vtChannelsSorted[np.arange(0, dNChannels - dRemoveChannels)]))

    # Distribui a potencia restante entre todos os canais remanescentes
    dPdiff = dPt - Ps.sum()
    vtOptPaux = dPdiff / (dNChannels - dRemoveChannels) + Ps

    # Put optimum power in the original channel order
    vtOptP = np.zeros([vtChannels.size, ])
    vtOptP[vtChannelsSortIndexes[np.arange(0, dNChannels - dRemoveChannels)]] = vtOptPaux
    mu = vtOptPaux[0] + float(noiseVar) / vtChannelsSorted[0]

    return (vtOptP, mu)


def genLatexCode(vtChannels, waterLevel, noiseVar=1.0, channelLength=0.8):
    """Generates latex code to draw (using Tikz) the waterfilling representation.

    Arguments:
    - `vtChannels`: Numpy array with the channel gains of the parallel AWGN channels.
    - `waterLevel`: Water level
    - `channelLength`: Length (in cm) of the representation of each channel in the X axis.
    """

    # Draw Parameters
    maxYcoord = 3  # Maximum coordinate for the Y axis. Note that the axis
                   # itself will expand two millimeters

    # ----- Auxiliary function definition ---------------------------------
    def pointsToString(points):
        """
        Auxiliary function
        """
        pointsString = ""
        for index in range(0, numChannels):
            pointsString += ("{num}/{value},".format(num=index, value=points[index]))

        # The last element must not have a comma at the end
        # index = numChannels-1
        # pointsString += ("{num}/{value}".format(num=index,value=vtInvChannels[index]))

                          # Remove the last ',' character
        return pointsString.rstrip(',')
    # ----- End of auxiliary function -------------------------------------

    ## Code
    vtInvChannels = float(noiseVar) / np.squeeze(vtChannels)
    numChannels = vtChannels.size
    xMax = numChannels * channelLength + 0.2
    yMax = (np.concatenate((vtInvChannels, np.array([waterLevel])))).max()

    texCode = """
    \\documentclass[a4]{{report}}

    \\usepackage[english]{{babel}}
    \\usepackage[utf8]{{inputenc}} % Use this if the file is encoded with utf-8
    \\usepackage{{times}}
    \\usepackage[T1]{{fontenc}}
    \\usepackage{{amsmath,amssymb}} % Part of AMS-LaTeX
    % One of the good things of the amsmath package is the math enviroments matrix, pmatrix, bmatrix, Bmatrix, vmatrix and Vmatrix
    \\usepackage{{graphicx}}
    \\usepackage{{tikz}} % Create graphics in Latex
    \\usepackage{{cancel}} % teste $\\cancel{{x}}$ e voce vera o que ele faz. Outro melhor ainda e $\\cancelto{{x}}{{0}}$.

    \\everymath{{\\displaystyle}}
    \\begin{{document}}

    \\pgfdeclarelayer{{background}}
    \\pgfdeclarelayer{{foreground}}
    \\pgfsetlayers{{background,main,foreground}}


    \\begin{{tikzpicture}}[every node/.style={{scale=0.8}}]
      %% Desenha eixos
      \\coordinate (origem) at (0,0);
      \\def\\YMax{{ {YMax} }}
      \\def\\XMax{{ {XMax} }}

      \\draw[-latex,shorten <=-3mm] (origem) -- (0,\\YMax) node[left]{{$\\frac{{N_0}}{{|H_n|^2}}$}};
      \\draw[-latex,shorten <=-3mm,shorten >=-1mm] (origem) -- (\\XMax,0) node[below] {{Channel}};

      %% Desenha nivel de agual
      \\def\\waterLevelCoord{{ {WaterLevelCoord} }}
      \\def\\waterLevelLabel{{ {WaterLevelLabel:.4f} }}
      \\begin{{pgfonlayer}}{{background}}
        \\fill[gray!30!white] (origem) rectangle (\\XMax,\\waterLevelCoord);
      \\end{{pgfonlayer}}
      \\begin{{pgfonlayer}}{{foreground}}
        \\draw[dashed] (0,\\waterLevelCoord) node[left] {{ \\waterLevelLabel }} -- ++(\\XMax,0);
      \\end{{pgfonlayer}}

      %% Desenha os canais
      \\def\\channelLength{{8mm}}
      \\draw[fill=white] (0,0)
      \\foreach \\ind/\\value in {{ {Points} }}
      {{
                                              % Store coordinates P0,P1,...
        -| (\\ind*\\channelLength,\\value) coordinate (P\\ind)
      }}
       -- ++(\\channelLength,0) -- ++(0,-{LastPoint});

       %% Draw the Power arrows
       % \\draw[latex-] (P0) ++(\\channelLength/2,1mm) -- ++(30:0.6cm) node[right] {{$P_1^*$}};
       % \\draw[latex-latex] (P1) ++(\\channelLength/2,0) -- node[right] {{$P_2^*$}} ++(0,\\waterLevelCoord-0.6*\\YMax);
       % \\draw[latex-latex] (P2) ++(\\channelLength/2,0) -- node[right] {{$P_3^*$}} ++(0,\\waterLevelCoord-0.2*\\YMax);

    \\end{{tikzpicture}}

    \\end{{document}}
    """

    pointsString = pointsToString(maxYcoord * (vtInvChannels / yMax))
    newTexCode = texCode.format(XMax=xMax,
                                YMax=maxYcoord + 0.2,  # yMax,
                                WaterLevelCoord=maxYcoord * (waterLevel / yMax),
                                WaterLevelLabel=waterLevel,
                                Points=pointsString,
                                LastPoint=(maxYcoord * (vtInvChannels[-1] / yMax)))

    return newTexCode


def drawWF(vtChannels, waterLevel, noiseVar=1.0, channelLength=0.8):
    """
    Creates a file with the latex code to draw (using Tikz) the
    waterfilling representation.
    """
    texCode = genLatexCode(vtChannels, waterLevel, noiseVar, channelLength)
    fId = file("texCode.tex", 'w')
    fId.write(texCode)
    fId.close()


def test_drawwf():
    print "Inicio"
    vtChannels = np.array([9.32904521e-13, 2.63321084e-13, 5.06505202e-14])
    noiseVar = 2.5119e-14
    Pt = 0.2512
    (vtOptP, mu) = doWF(vtChannels, Pt, noiseVar)
    #print (vtOptP, mu)

    drawWF(vtChannels, mu, noiseVar)


if __name__ == '__main__':
    #vtChannels = np.abs(randn_c(4,1))
    vtChannels = np.array([ 0.49702888,
                            0.59012981,
                            0.43485267,
                            0.6692608 ])

    Power = 4
    noise_var = 0.1
    Es = 1

    (vtOptP, mu) = doWF(vtChannels, Power, noise_var)

    print "vtOptP"
    print vtOptP
    print "mu"
    print mu

    print sum(vtOptP)

    #drawWF(vtChannels, mu, noise_var)


if __name__ == '__main__1':
    import doctest
    doctest.testmod()
    print "{0} executed".format(__file__)
