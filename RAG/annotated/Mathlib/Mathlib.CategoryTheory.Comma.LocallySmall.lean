instance Comma.locallySmall
    (L : A ⥤ T) (R : B ⥤ T) [LocallySmall.{w} A] [LocallySmall.{w} B] :
    LocallySmall.{w} (Comma L R) where
  hom_small X Y := small_of_injective.{w}
      (f := fun g ↦ (⟨g.left, g.right⟩ : _ × _))
                        /-
                          A : Type u₁
                          B : Type u₂
                          T : Type u₃
                          inst✝⁴ : CategoryTheory.Category.{v₁, u₁} A
                          inst✝³ : CategoryTheory.Category.{v₂, u₂} B
                          inst✝² : CategoryTheory.Category.{v₃, u₃} T
                          L : CategoryTheory.Functor A T
                          R : CategoryTheory.Functor B T
                          inst✝¹ : CategoryTheory.LocallySmall.{w, v₁, u₁} A
                          inst✝ : CategoryTheory.LocallySmall.{w, v₂, u₂} B
                          X Y : CategoryTheory.Comma L R
                          x✝² x✝¹ : Quiver.Hom X Y
                          x✝ : Eq ((fun g => { fst := g.left, snd := g.right }) x✝²) ((fun g => { fst := …
                          ⊢ Eq x✝² x✝¹
                        -/
        (fun _ _ _ ↦ by aesop)
                        /-
                          🎉 no goals
                        -/


instance StructuredArrow.locallySmall (S : T) (T : B ⥤ T)
    [LocallySmall.{w} B] :
    LocallySmall.{w} (StructuredArrow S T) :=
  Comma.locallySmall _ _


instance CostructuredArrow.locallySmall (S : A ⥤ T) (X : T)
    [LocallySmall.{w} A] :
    LocallySmall.{w} (CostructuredArrow S X) :=
  Comma.locallySmall _ _


instance Over.locallySmall (X : T) [LocallySmall.{w} T] :
    LocallySmall.{w} (Over X) :=
  CostructuredArrow.locallySmall _ _


instance Under.locallySmall (X : T) [LocallySmall.{w} T] :
    LocallySmall.{w} (Under X) :=
  StructuredArrow.locallySmall _ _


