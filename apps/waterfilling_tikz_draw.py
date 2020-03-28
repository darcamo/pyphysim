#!/usr/bin/env python
"""
Draw the waterfilling solution using TiKz.

The genLatexCode returns latex code that can draw the provided solution by
the waterfilling algorithm, while the drawWF creates a file with this
code.
"""

import numpy as np
from pyphysim.comm import waterfilling


def genLatexCode(vtChannels, waterLevel, noiseVar=1.0, channelLength=0.8):
    """Generates latex code to draw (using Tikz) the waterfilling
    representation.

    Parameters
    ----------
    vtChannels : np.ndarray
        Channel gains of the parallel AWGN channels. This is a 1D numpy array.
    waterLevel : float
        Water level.
    noiseVar : float
        The noise variance.
    channelLength : float
        Length (in cm) of the representation of each channel in the X axis.

    Returns
    -------
    newTexCode : str
        LaTex code to draw the waterfilling solution.
    """

    # Draw Parameters
    maxYcoord = 3  # Maximum coordinate for the Y axis. Note that the axis

    # itself will expand two millimeters

    # ----- Auxiliary function definition ---------------------------------
    def pointsToString(points):
        """
        Auxiliary function

        Parameters
        ----------
        points : np.ndarray

        Returns
        -------
        str
            String representation of the points.
        """
        pointsString = ""
        for index in range(0, numChannels):
            pointsString += ("{num}/{value},".format(num=index,
                                                     value=points[index]))

        # The last element must not have a comma at the end
        # index = numChannels-1
        # pointsString += (
        #     "{num}/{value}".format(num=index,value=vtInvChannels[index]))

        # Remove the last ',' character
        return pointsString.rstrip(',')

    # ----- End of auxiliary function -------------------------------------

    # Code
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
    % One of the good things of the amsmath package is the math environments matrix, pmatrix, bmatrix, Bmatrix, vmatrix and Vmatrix
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
    newTexCode = texCode.format(
        XMax=xMax,
        YMax=maxYcoord + 0.2,  # yMax,
        WaterLevelCoord=maxYcoord * (waterLevel / yMax),
        WaterLevelLabel=waterLevel,
        Points=pointsString,
        LastPoint=(maxYcoord * (vtInvChannels[-1] / yMax)))

    return newTexCode


def drawWF(vtChannels, waterLevel, noiseVar=1.0, channelLength=0.8):
    """Creates a file with the latex code to draw (using Tikz) the
    waterfilling representation.

    Parameters
    ----------
    vtChannels : np.ndarray
        Channel gains of the parallel AWGN channels. This is a 1D numpy array.
    waterLevel : float
        Water level.
    noiseVar : float
        Noise variance.
    channelLength : float
        Length (in cm) of the representation of each channel in the X axis.
    """
    texCode = genLatexCode(vtChannels, waterLevel, noiseVar, channelLength)
    fId = open("texCode.tex", 'w')
    fId.write(texCode)
    fId.close()


# This method is only for testing purposes and is does not need to be
# exposed to the outside of this module.
def _test_drawwf():
    import os

    print("Inicio")

    vtChannels = np.array([9.32904521e-13, 2.63321084e-13, 5.06505202e-14])
    noiseVar = 2.5119e-14
    Pt = 0.2512
    (vtOptP, mu) = waterfilling.doWF(vtChannels, Pt, noiseVar)
    # print (vtOptP, mu)

    drawWF(vtChannels, mu, noiseVar)
    os.system("pdflatex texCode.tex > /dev/null")
    os.system("rm texCode.log texCode.aux texCode.tex")


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
if __name__ == '__main__':  # pragma: no cover
    _test_drawwf()

if __name__ == '__main__1':  # pragma: no cover
    # vtChannels = np.abs(randn_c(4,1))
    vtChannels = np.array([0.49702888, 0.59012981, 0.43485267, 0.6692608])
    Power = 4
    noise_var = 0.1
    Es = 1

    (vtOptP, mu) = waterfilling.doWF(vtChannels, Power, noise_var)

    print("vtOptP")
    print(vtOptP)
    print("mu")
    print(mu)

    print(sum(vtOptP))

    # drawWF(vtChannels, mu, noise_var)
