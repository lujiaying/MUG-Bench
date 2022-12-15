import pandas as pd
import matplotlib.pyplot as plt
import json


if __name__ == '__main__':
    # plot raw image mean rgb
    with open('mean_rgb.json') as fopen:
        mean_rgb = json.load(fopen)

    n = 400
    poke = mean_rgb["Pokemon"][0:n]
    league = mean_rgb["LeagueOfLegends"][0:n]
    csgo = mean_rgb["CSGO"][0:n]
    hearthstone = mean_rgb["Hearthstone"][0:n]

    fig = plt.figure(figsize = (12, 12))
    ax = fig.add_subplot(projection='3d')
    x_pokemon = [i[0] for i in poke]
    y_pokemon = [i[1] for i in poke]
    z_pokemon = [i[2] for i in poke]

    x_hearthstone = [i[0] for i in hearthstone]
    y_hearthstone = [i[1] for i in hearthstone]
    z_hearthstone = [i[2] for i in hearthstone]


    x_league = [i[0] for i in league]
    y_league = [i[1] for i in league]
    z_league = [i[2] for i in league]


    x_csgo = [i[0] for i in csgo]
    y_csgo = [i[1] for i in csgo]
    z_csgo = [i[2] for i in csgo]
    marker_size=80
    alpha=0.50
    ax.scatter(x_pokemon, y_pokemon, z_pokemon, c="darkorange", s=marker_size, alpha=alpha)
    ax.scatter(x_league, y_league, z_league, c="firebrick", s=marker_size, alpha=alpha)
    ax.scatter(x_csgo, y_csgo, z_csgo, c="forestgreen", s=marker_size, alpha=alpha)
    ax.scatter(x_hearthstone, y_hearthstone, z_hearthstone, c="royalblue", s=marker_size, alpha=alpha)

    fontsize = 26
    ax.set_xlabel("Red Channel", fontsize=fontsize, labelpad=9)
    ax.set_ylabel("Green Channel", fontsize=fontsize, labelpad=9)
    ax.set_zlabel("Blue Channel", fontsize=fontsize, labelpad=9)

    ax.tick_params(axis='both', which='major', labelsize=16)
    legend = ax.legend(["pkm", "lol", "csg", "hs"], fontsize=38, loc='upper center', ncol=2,
                       bbox_to_anchor=(0.5, 1.0), shadow=False)
    ax.figure.savefig('MeanRGBScatter3D.pdf', bbox_inches='tight')
