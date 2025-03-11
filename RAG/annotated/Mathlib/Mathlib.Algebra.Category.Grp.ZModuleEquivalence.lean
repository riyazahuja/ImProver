/-- The forgetful functor from `ℤ` modules to `AddCommGrp` is full. -/
instance forget₂_addCommGroup_full : (forget₂ (ModuleCat ℤ) AddCommGrp.{u}).Full where
  map_surjective {A B}
    -- `AddMonoidHom.toIntLinearMap` doesn't work here because `A` and `B` are not
    -- definitionally equal to the canonical `AddCommGroup.toIntModule` module
    -- instances it expects.
    f := ⟨@ModuleCat.ofHom _ _ _ _ _ A.isModule _ B.isModule <|
            @LinearMap.mk _ _ _ _ _ _ _ _ _ A.isModule B.isModule
            { toFun := f,
              map_add' := AddMonoidHom.map_add (show A.carrier →+ B.carrier from f) }
            (fun n x => by
              /-
                A B : ModuleCat Int
                f : Quiver.Hom ((CategoryTheory.forget₂ (ModuleCat Int) AddCommGrp).obj A) ((C …
                n : Int
                x : ↑A
                ⊢ Eq ({ toFun := ⇑f, map_add' := ⋯ }.toFun (HSMul.hSMul n x)) (HSMul.hSMul ((R …
              -/
              convert AddMonoidHom.map_zsmul (show A.carrier →+ B.carrier from f) x n <;>
                /-
                  case h.e'_2.h.e'_1.h.e'_4.h.e'_3
                  A B : ModuleCat Int
                  f : Quiver.Hom ((CategoryTheory.forget₂ (ModuleCat Int) AddCommGrp).obj A) ((C …
                  n : Int
                  x : ↑A
                  ⊢ Eq SMulZeroClass.toSMul SubNegMonoid.toZSMul
                -/
                        /-
                          🎉 no goals
                        -/
                ext <;> apply int_smul_eq_zsmul), rfl⟩
                        /-
                          🎉 no goals
                        -/


/-- The forgetful functor from `ℤ` modules to `AddCommGrp` is essentially surjective. -/
instance forget₂_addCommGrp_essSurj : (forget₂ (ModuleCat ℤ) AddCommGrp.{u}).EssSurj where
  mem_essImage A :=
    ⟨ModuleCat.of ℤ A,
      ⟨{  hom := 𝟙 A
          inv := 𝟙 A }⟩⟩


noncomputable instance forget₂AddCommGroupIsEquivalence :
    (forget₂ (ModuleCat ℤ) AddCommGrp.{u}).IsEquivalence where


