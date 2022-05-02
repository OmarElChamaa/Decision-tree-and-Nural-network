#!/usr/bin/env python3
# -*- coding: utf-8 -*_

import pandas as pd
import numpy as np




def test_info_gain():
    data = {
        'Longueur' : [5, 6, 7, 20, 30],
        'Largeur' : [8, 9,  15, 30, 40],
        'Prix' : [100, 250, 600, 200, 500],
        'Class' : ['Mobile','Mobile', 'Mobile', 'Tele', 'Tele']
    } 
    df = pd.Dataframe(data)
    print(df)
