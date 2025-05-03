/-- The data of a localized category with a given universe
for the morphisms. -/
class HasLocalization where
  /-- the objects of the localized category. -/
  {D : Type u}
  /-- the category structure. -/
  [hD : Category.{w} D]
  /-- the localization functor. -/
  L : C ⥤ D
  [hL : L.IsLocalization W]


/-- The localized category for `W : MorphismProperty C`
that is fixed by the `[HasLocalization W]` instance. -/
def Localization' := HasLocalization.D W


instance : Category W.Localization' := HasLocalization.hD


/-- The localization functor `C ⥤ W.Localization'`
that is fixed by the `[HasLocalization W]` instance. -/
def Q' : C ⥤ W.Localization' := HasLocalization.L


instance : W.Q'.IsLocalization W := HasLocalization.hL


/-- The constructed localized category. -/
def HasLocalization.standard : HasLocalization.{max u v} W where
  L := W.Q


