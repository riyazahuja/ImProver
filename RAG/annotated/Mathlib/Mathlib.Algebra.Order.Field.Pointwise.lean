theorem smul_Ioo : r • Ioo a b = Ioo (r • a) (r • b) := (OrderIso.mulLeft₀ r hr).image_Ioo a b

theorem smul_Icc : r • Icc a b = Icc (r • a) (r • b) := (OrderIso.mulLeft₀ r hr).image_Icc a b

theorem smul_Ico : r • Ico a b = Ico (r • a) (r • b) := (OrderIso.mulLeft₀ r hr).image_Ico a b

theorem smul_Ioc : r • Ioc a b = Ioc (r • a) (r • b) := (OrderIso.mulLeft₀ r hr).image_Ioc a b

theorem smul_Ioi : r • Ioi a = Ioi (r • a) := (OrderIso.mulLeft₀ r hr).image_Ioi a

theorem smul_Iio : r • Iio a = Iio (r • a) := (OrderIso.mulLeft₀ r hr).image_Iio a

theorem smul_Ici : r • Ici a = Ici (r • a) := (OrderIso.mulLeft₀ r hr).image_Ici a

theorem smul_Iic : r • Iic a = Iic (r • a) := (OrderIso.mulLeft₀ r hr).image_Iic a


