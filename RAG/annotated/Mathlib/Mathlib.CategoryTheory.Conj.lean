/-- An isomorphism between two objects defines a monoid isomorphism between their
monoid of endomorphisms. -/
def conj : End X ≃* End Y :=
  { homCongr α α with map_mul' := fun f g => homCongr_comp α α α g f }


theorem conj_apply (f : End X) : α.conj f = α.inv ≫ f ≫ α.hom :=
  rfl


@[simp]
theorem conj_comp (f g : End X) : α.conj (f ≫ g) = α.conj f ≫ α.conj g :=
  map_mul α.conj g f


@[simp]
theorem conj_id : α.conj (𝟙 X) = 𝟙 Y :=
  map_one α.conj


@[simp]
theorem refl_conj (f : End X) : (Iso.refl X).conj f = f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X : C
    f : CategoryTheory.End X
    ⊢ Eq ((CategoryTheory.Iso.refl X).conj f) f
  -/
  rw [conj_apply, Iso.refl_inv, Iso.refl_hom, Category.id_comp, Category.comp_id]
  /-
    🎉 no goals
  -/


@[simp]
theorem trans_conj {Z : C} (β : Y ≅ Z) (f : End X) : (α ≪≫ β).conj f = β.conj (α.conj f) :=
  homCongr_trans α α β β f


@[simp]
theorem symm_self_conj (f : End X) : α.symm.conj (α.conj f) = f := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    α : CategoryTheory.Iso X Y
    f : CategoryTheory.End X
    ⊢ Eq (α.symm.conj (α.conj f)) f
  -/
  rw [← trans_conj, α.self_symm_id, refl_conj]
  /-
    🎉 no goals
  -/


@[simp]
theorem self_symm_conj (f : End Y) : α.conj (α.symm.conj f) = f :=
  α.symm.symm_self_conj f


@[simp]
theorem conj_pow (f : End X) (n : ℕ) : α.conj (f ^ n) = α.conj f ^ n :=
  α.conj.toMonoidHom.map_pow f n

-- Porting note (https://github.com/leanprover-community/mathlib4/issues/11215): TODO: change definition so that `conjAut_apply` becomes a `rfl`?

/-- `conj` defines a group isomorphisms between groups of automorphisms -/
def conjAut : Aut X ≃* Aut Y :=
  (Aut.unitsEndEquivAut X).symm.trans <| (Units.mapEquiv α.conj).trans <| Aut.unitsEndEquivAut Y


                                                                         /-
                                                                           C : Type u
                                                                           inst✝ : CategoryTheory.Category.{v, u} C
                                                                           X Y : C
                                                                           α : CategoryTheory.Iso X Y
                                                                           f : CategoryTheory.Aut X
                                                                           ⊢ Eq (α.conjAut f) (α.symm.trans (CategoryTheory.Iso.trans f α))
                                                                         -/
theorem conjAut_apply (f : Aut X) : α.conjAut f = α.symm ≪≫ f ≪≫ α := by aesop_cat
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


@[simp]
theorem conjAut_hom (f : Aut X) : (α.conjAut f).hom = α.conj f.hom :=
  rfl


@[simp]
theorem trans_conjAut {Z : C} (β : Y ≅ Z) (f : Aut X) :
    (α ≪≫ β).conjAut f = β.conjAut (α.conjAut f) := by
  /-
    C : Type u
    inst✝ : CategoryTheory.Category.{v, u} C
    X Y : C
    α : CategoryTheory.Iso X Y
    Z : C
    β : CategoryTheory.Iso Y Z
    f : CategoryTheory.Aut X
    ⊢ Eq ((α.trans β).conjAut f) (β.conjAut (α.conjAut f))
  -/
  simp only [conjAut_apply, Iso.trans_symm, Iso.trans_assoc]
  /-
    🎉 no goals
  -/


@[simp]
theorem conjAut_mul (f g : Aut X) : α.conjAut (f * g) = α.conjAut f * α.conjAut g :=
  map_mul α.conjAut f g


@[simp]
theorem conjAut_trans (f g : Aut X) : α.conjAut (f ≪≫ g) = α.conjAut f ≪≫ α.conjAut g :=
  conjAut_mul α g f


@[simp]
theorem conjAut_pow (f : Aut X) (n : ℕ) : α.conjAut (f ^ n) = α.conjAut f ^ n :=
  map_pow α.conjAut f n


@[simp]
theorem conjAut_zpow (f : Aut X) (n : ℤ) : α.conjAut (f ^ n) = α.conjAut f ^ n :=
  map_zpow α.conjAut f n


theorem map_conj {X Y : C} (α : X ≅ Y) (f : End X) :
    F.map (α.conj f) = (F.mapIso α).conj (F.map f) :=
  map_homCongr F α α f


theorem map_conjAut (F : C ⥤ D) {X Y : C} (α : X ≅ Y) (f : Aut X) :
    F.mapIso (α.conjAut f) = (F.mapIso α).conjAut (F.mapIso f) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    D : Type u₁
    inst✝ : CategoryTheory.Category.{v₁, u₁} D
    F : CategoryTheory.Functor C D
    X Y : C
    α : CategoryTheory.Iso X Y
    f : CategoryTheory.Aut X
    ⊢ Eq (F.mapIso (α.conjAut f)) ((F.mapIso α).conjAut (F.mapIso f))
  -/
  ext; simp only [mapIso_hom, Iso.conjAut_hom, F.map_conj]
       /-
         🎉 no goals
       -/

-- alternative proof: by simp only [Iso.conjAut_apply, F.mapIso_trans, F.mapIso_symm]

