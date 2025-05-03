/-- The functor which sends a triangle `X ⟶ Y ⟶ Z ⟶ X⟦1⟧` in `C` to the triangle
`op Z ⟶ op Y ⟶ op X ⟶ (op Z)⟦1⟧` in `Cᵒᵖ` (without introducing signs). -/
@[simps]
noncomputable def functor : (Triangle C)ᵒᵖ ⥤ Triangle Cᵒᵖ where
  obj T := Triangle.mk T.unop.mor₂.op T.unop.mor₁.op
      ((opShiftFunctorEquivalence C 1).counitIso.inv.app (Opposite.op T.unop.obj₁) ≫
        T.unop.mor₃.op⟦(1 : ℤ)⟧')
  map {T₁ T₂} φ :=
    { hom₁ := φ.unop.hom₃.op
      hom₂ := φ.unop.hom₂.op
      hom₃ := φ.unop.hom₁.op
      comm₁ := Quiver.Hom.unop_inj φ.unop.comm₂.symm
      comm₂ := Quiver.Hom.unop_inj φ.unop.comm₁.symm
      comm₃ := by
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.38, u_1} C
          inst✝ : CategoryTheory.HasShift C Int
          T₁ T₂ : Opposite (CategoryTheory.Pretriangulated.Triangle C)
          φ : Quiver.Hom T₁ T₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun T => CategoryTheory.Pretriangul …
        -/
        dsimp
        rw [assoc, ← Functor.map_comp, ← op_comp, ← φ.unop.comm₃, op_comp, Functor.map_comp,
          opShiftFunctorEquivalence_counitIso_inv_naturality_assoc]
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.38, u_1} C
          inst✝ : CategoryTheory.HasShift C Int
          T₁ T₂ : Opposite (CategoryTheory.Pretriangulated.Triangle C)
          φ : Quiver.Hom T₁ T₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Pretriangulated.opSh …
        -/
        rfl }
        /-
          🎉 no goals
        -/


/-- The functor which sends a triangle `X ⟶ Y ⟶ Z ⟶ X⟦1⟧` in `Cᵒᵖ` to the triangle
`Z.unop ⟶ Y.unop ⟶ X.unop ⟶ Z.unop⟦1⟧` in `C` (without introducing signs). -/
@[simps]
noncomputable def inverse : Triangle Cᵒᵖ ⥤ (Triangle C)ᵒᵖ where
  obj T := Opposite.op (Triangle.mk T.mor₂.unop T.mor₁.unop
      (((opShiftFunctorEquivalence C 1).unitIso.inv.app T.obj₁).unop ≫ T.mor₃.unop⟦(1 : ℤ)⟧'))
  map {T₁ T₂} φ := Quiver.Hom.op
    { hom₁ := φ.hom₃.unop
      hom₂ := φ.hom₂.unop
      hom₃ := φ.hom₁.unop
      comm₁ := Quiver.Hom.op_inj φ.comm₂.symm
      comm₂ := Quiver.Hom.op_inj φ.comm₁.symm
      comm₃ := Quiver.Hom.op_inj (by
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.12488, u_1} C
          inst✝ : CategoryTheory.HasShift C Int
          T₁ T₂ : CategoryTheory.Pretriangulated.Triangle (Opposite C)
          φ : Quiver.Hom T₁ T₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.Trian …
        -/
        dsimp
        rw [assoc, ← opShiftFunctorEquivalence_unitIso_inv_naturality,
          ← op_comp_assoc, ← Functor.map_comp, ← unop_comp, ← φ.comm₃,
          unop_comp, Functor.map_comp, op_comp, assoc]) }


/-- The unit isomorphism of the
equivalence `triangleOpEquivalence C : (Triangle C)ᵒᵖ ≌ Triangle Cᵒᵖ` . -/
@[simps!]
noncomputable def unitIso : 𝟭 _ ≅ functor C ⋙ inverse C :=
  NatIso.ofComponents (fun T => Iso.op
                                                                   /-
                                                                     C : Type u_1
                                                                     inst✝¹ : CategoryTheory.Category.{?u.36977, u_1} C
                                                                     inst✝ : CategoryTheory.HasShift C Int
                                                                     T : Opposite (CategoryTheory.Pretriangulated.Triangle C)
                                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.Trian …
                                                                   -/
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
    (Triangle.isoMk _ _ (Iso.refl _) (Iso.refl _) (Iso.refl _) (by aesop_cat) (by aesop_cat)
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/
      (Quiver.Hom.op_inj
            /-
              C : Type u_1
              inst✝¹ : CategoryTheory.Category.{?u.36977, u_1} C
              inst✝ : CategoryTheory.HasShift C Int
              T : Opposite (CategoryTheory.Pretriangulated.Triangle C)
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Pretriangulated.Trian …
            -/
        (by simp [shift_unop_opShiftFunctorEquivalence_counitIso_inv_app]))))
            /-
              🎉 no goals
            -/
                                              /-
                                                C : Type u_1
                                                inst✝¹ : CategoryTheory.Category.{?u.36977, u_1} C
                                                inst✝ : CategoryTheory.HasShift C Int
                                                T₁ T₂ : Opposite (CategoryTheory.Pretriangulated.Triangle C)
                                                f : Quiver.Hom T₁ T₂
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Functor.id (Opposite …
                                              -/
    (fun {T₁ T₂} f => Quiver.Hom.unop_inj (by aesop_cat))
                                              /-
                                                🎉 no goals
                                              -/


/-- The counit isomorphism of the
equivalence `triangleOpEquivalence C : (Triangle C)ᵒᵖ ≌ Triangle Cᵒᵖ` . -/
@[simps!]
noncomputable def counitIso : inverse C ⋙ functor C ≅ 𝟭 _ :=
  NatIso.ofComponents (fun T => by
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.81172, u_1} C
      inst✝ : CategoryTheory.HasShift C Int
      T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
      ⊢ CategoryTheory.Iso (((CategoryTheory.Pretriangulated.TriangleOpEquivalence.i …
    -/
    refine Triangle.isoMk _ _ (Iso.refl _) (Iso.refl _) (Iso.refl _) ?_ ?_ ?_
      /-
        case refine_1
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.81172, u_1} C
        inst✝ : CategoryTheory.HasShift C Int
        T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Pretriangulated.Tri …
      -/
    · aesop_cat
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.81172, u_1} C
        inst✝ : CategoryTheory.HasShift C Int
        T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Pretriangulated.Tri …
      -/
    · aesop_cat
      /-
        🎉 no goals
      -/
      /-
        case refine_3
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.81172, u_1} C
        inst✝ : CategoryTheory.HasShift C Int
        T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Pretriangulated.Tri …
      -/
    · dsimp
      rw [Functor.map_id, comp_id, id_comp, Functor.map_comp,
        ← opShiftFunctorEquivalence_counitIso_inv_naturality_assoc,
        opShiftFunctorEquivalence_counitIso_inv_app_shift, ← Functor.map_comp,
        Iso.hom_inv_id_app, Functor.map_id]
      /-
        case refine_3
        C : Type u_1
        inst✝¹ : CategoryTheory.Category.{?u.81172, u_1} C
        inst✝ : CategoryTheory.HasShift C Int
        T : CategoryTheory.Pretriangulated.Triangle (Opposite C)
        ⊢ Eq (CategoryTheory.CategoryStruct.comp T.mor₃ (CategoryTheory.CategoryStruct …
      -/
      simp only [Functor.id_obj, comp_id])
      /-
        🎉 no goals
      -/
        /-
          C : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.81172, u_1} C
          inst✝ : CategoryTheory.HasShift C Int
          ⊢ ∀ {X Y : CategoryTheory.Pretriangulated.Triangle (Opposite C)} (f : Quiver.H …
        -/
    (by aesop_cat)
        /-
          🎉 no goals
        -/


/-- An anti-equivalence between the categories of triangles in `C` and in `Cᵒᵖ`.
A triangle in `Cᵒᵖ` shall be distinguished iff it correspond to a distinguished
triangle in `C` via this equivalence. -/
@[simps]
noncomputable def triangleOpEquivalence :
    (Triangle C)ᵒᵖ ≌ Triangle Cᵒᵖ where
  functor := TriangleOpEquivalence.functor C
  inverse := TriangleOpEquivalence.inverse C
  unitIso := TriangleOpEquivalence.unitIso C
  counitIso := TriangleOpEquivalence.counitIso C


