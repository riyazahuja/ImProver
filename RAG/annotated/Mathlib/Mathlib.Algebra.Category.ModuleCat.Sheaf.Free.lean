/-- The free sheaf of modules on a certain type `I`. -/
noncomputable def free (I : Type u) : SheafOfModules.{u} R := ∐ (fun (_ : I) ↦ unit R)


/-- The data of a morphism `free I ⟶ M` from a free sheaf of modules is
equivalent to the data of a family `I → M.sections` of sections of `M`. -/
noncomputable def freeHomEquiv (M : SheafOfModules.{u} R) {I : Type u} :
    (free I ⟶ M) ≃ (I → M.sections) where
  toFun f i := M.unitHomEquiv (Sigma.ι (fun (_ : I) ↦ unit R) i ≫ f)
  invFun s := Sigma.desc (fun i ↦ M.unitHomEquiv.symm (s i))
                                      /-
                                        C : Type u'
                                        inst✝³ : CategoryTheory.Category.{v', u'} C
                                        J : CategoryTheory.GrothendieckTopology C
                                        R : CategoryTheory.Sheaf J RingCat
                                        inst✝² : CategoryTheory.HasWeakSheafify J AddCommGrp
                                        inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
                                        inst✝ : J.HasSheafCompose (CategoryTheory.forget₂ RingCat AddCommGrp)
                                        M : SheafOfModules R
                                        I : Type u
                                        s : Quiver.Hom (SheafOfModules.free I) M
                                        ⊢ ∀ (b : I), Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Sig …
                                      -/
  left_inv s := Sigma.hom_ext _ _ (by simp)
                                      /-
                                        🎉 no goals
                                      -/
                    /-
                      C : Type u'
                      inst✝³ : CategoryTheory.Category.{v', u'} C
                      J : CategoryTheory.GrothendieckTopology C
                      R : CategoryTheory.Sheaf J RingCat
                      inst✝² : CategoryTheory.HasWeakSheafify J AddCommGrp
                      inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
                      inst✝ : J.HasSheafCompose (CategoryTheory.forget₂ RingCat AddCommGrp)
                      M : SheafOfModules R
                      I : Type u
                      f : I → M.sections
                      ⊢ Eq ((fun f i => M.unitHomEquiv (CategoryTheory.CategoryStruct.comp (Category …
                    -/
  right_inv f := by ext1 i; simp
                            /-
                              🎉 no goals
                            -/


lemma freeHomEquiv_comp_apply {M N : SheafOfModules.{u} R} {I : Type u}
    (f : free I ⟶ M) (p : M ⟶ N) (i : I) :
    N.freeHomEquiv (f ≫ p) i = sectionsMap p (M.freeHomEquiv f i) := rfl


lemma freeHomEquiv_symm_comp {M N : SheafOfModules.{u} R} {I : Type u} (s : I → M.sections)
    (p : M ⟶ N) :
    M.freeHomEquiv.symm s ≫ p = N.freeHomEquiv.symm (fun i ↦ sectionsMap p (s i)) :=
                               /-
                                 C : Type u'
                                 inst✝³ : CategoryTheory.Category.{v', u'} C
                                 J : CategoryTheory.GrothendieckTopology C
                                 R : CategoryTheory.Sheaf J RingCat
                                 inst✝² : CategoryTheory.HasWeakSheafify J AddCommGrp
                                 inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
                                 inst✝ : J.HasSheafCompose (CategoryTheory.forget₂ RingCat AddCommGrp)
                                 M N : SheafOfModules R
                                 I : Type u
                                 s : I → M.sections
                                 p : Quiver.Hom M N
                                 ⊢ Eq (N.freeHomEquiv (CategoryTheory.CategoryStruct.comp (M.freeHomEquiv.symm  …
                               -/
  N.freeHomEquiv.injective (by ext; simp [freeHomEquiv_comp_apply])
                                    /-
                                      🎉 no goals
                                    -/


/-- The tautological section of `free I : SheafOfModules R` corresponding to `i : I`. -/
noncomputable abbrev freeSection {I : Type u} (i : I) : (free (R := R) I).sections :=
  (free (R := R) I).freeHomEquiv (𝟙 (free I)) i


/-- The morphism of presheaves of `R`-modules `free I ⟶ free J` induced by
a map `f : I → J`. -/
noncomputable def freeMap : free (R := R) I ⟶ free J :=
  (freeHomEquiv _).symm (fun i ↦ freeSection (f i))


@[simp]
lemma freeHomEquiv_freeMap :
    (freeHomEquiv _ (freeMap (R := R) f)) = freeSection.comp f :=
                                      /-
                                        C : Type u'
                                        inst✝³ : CategoryTheory.Category.{v', u'} C
                                        J✝ : CategoryTheory.GrothendieckTopology C
                                        R : CategoryTheory.Sheaf J✝ RingCat
                                        inst✝² : CategoryTheory.HasWeakSheafify J✝ AddCommGrp
                                        inst✝¹ : J✝.WEqualsLocallyBijective AddCommGrp
                                        inst✝ : J✝.HasSheafCompose (CategoryTheory.forget₂ RingCat AddCommGrp)
                                        I J : Type u
                                        f : I → J
                                        ⊢ Eq ((SheafOfModules.free J).freeHomEquiv.symm ((SheafOfModules.free J).freeH …
                                      -/
  (freeHomEquiv _).symm.injective (by simp; rfl)
                                            /-
                                              🎉 no goals
                                            -/


@[simp]
lemma sectionMap_freeMap_freeSection (i : I) :
    sectionsMap (freeMap (R := R) f) (freeSection i) = freeSection (f i) := by
  /-
    C : Type u'
    inst✝³ : CategoryTheory.Category.{v', u'} C
    J✝ : CategoryTheory.GrothendieckTopology C
    R : CategoryTheory.Sheaf J✝ RingCat
    inst✝² : CategoryTheory.HasWeakSheafify J✝ AddCommGrp
    inst✝¹ : J✝.WEqualsLocallyBijective AddCommGrp
    inst✝ : J✝.HasSheafCompose (CategoryTheory.forget₂ RingCat AddCommGrp)
    I J : Type u
    f : I → J
    i : I
    ⊢ Eq (SheafOfModules.sectionsMap (SheafOfModules.freeMap f) (SheafOfModules.fr …
  -/
  simp [← freeHomEquiv_comp_apply]
  /-
    🎉 no goals
  -/


/-- The functor `Type u ⥤ SheafOfModules.{u} R` which sends a type `I` to
`free I` which is a coproduct indexed by `I` of copies of `R` (thought as a
presheaf of modules over itself). --/
noncomputable def freeFunctor : Type u ⥤ SheafOfModules.{u} R where
  obj := free
  map f := freeMap f
                                             /-
                                               C : Type u'
                                               inst✝³ : CategoryTheory.Category.{v', u'} C
                                               J : CategoryTheory.GrothendieckTopology C
                                               R : CategoryTheory.Sheaf J RingCat
                                               inst✝² : CategoryTheory.HasWeakSheafify J AddCommGrp
                                               inst✝¹ : J.WEqualsLocallyBijective AddCommGrp
                                               inst✝ : J.HasSheafCompose (CategoryTheory.forget₂ RingCat AddCommGrp)
                                               X : Type u
                                               ⊢ Eq (({ obj := SheafOfModules.free, map := fun {X Y} f => SheafOfModules.free …
                                             -/
  map_id X := (freeHomEquiv _).injective (by ext1 i; simp)
                                                     /-
                                                       🎉 no goals
                                                     -/
                                                         /-
                                                           C : Type u'
                                                           inst✝³ : CategoryTheory.Category.{v', u'} C
                                                           J✝ : CategoryTheory.GrothendieckTopology C
                                                           R : CategoryTheory.Sheaf J✝ RingCat
                                                           inst✝² : CategoryTheory.HasWeakSheafify J✝ AddCommGrp
                                                           inst✝¹ : J✝.WEqualsLocallyBijective AddCommGrp
                                                           inst✝ : J✝.HasSheafCompose (CategoryTheory.forget₂ RingCat AddCommGrp)
                                                           I J K : Type u
                                                           f : Quiver.Hom I J
                                                           g : Quiver.Hom J K
                                                           ⊢ Eq (({ obj := SheafOfModules.free, map := fun {X Y} f => SheafOfModules.free …
                                                         -/
  map_comp {I J K} f g := (freeHomEquiv _).injective (by ext1; simp [freeHomEquiv_comp_apply])
                                                               /-
                                                                 🎉 no goals
                                                               -/


