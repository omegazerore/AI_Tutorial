"""
4500
"""

number_of_scrolls = {'delight': 110,
                     'regal': 115,
                     'basic': 6}

n_of_heros = {5: 6,
              4: 9,
              3: 8,
              2: 5}

score_of_shard = {5: 30000,
                  4: 10000,
                  3: 3000,
                  2: 750}

score_of_scrolls = {'delight': 2500,
                   'regal': 0,
                   'basic': 0}


pmf = {'delight': {5: 0.0466 + 0.0034 * 10,
                   4: 0.0286 + 0.0047 * 10,
                   3: 0.0299 + 0.0076 * 10},
       'regal': {4: 0.0139 + 0.0002 * 10,
                 3: 0.0099 + 0.0099 * 10,
                 2: 0.0872},
       'basic': {3: 0.0121 + 0.0041 * 10,
                 2: 0.0752 + 0.0203 * 10}}

for card, score_of_scroll in score_of_scrolls.items():

    expectation_value = 0

    expectation_value += score_of_scroll
    for star, probability in pmf[card].items():
        expectation_value += probability * n_of_heros[star] * score_of_shard[star]

    print(f"The expectation of the {card}: {expectation_value}")
    print(f"The expectation value from {number_of_scrolls[card]} {card} = {number_of_scrolls[card] * expectation_value}")


