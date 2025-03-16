instance : Preadditive (SingleObj α) where
  add_comp _ _ _ f f' g := mul_add g f f'
  comp_add _ _ _ f g g' := add_mul g g' f

-- TODO define `PreAddCat` (with additive functors as morphisms), and `Ring ⥤ PreAddCat`.

