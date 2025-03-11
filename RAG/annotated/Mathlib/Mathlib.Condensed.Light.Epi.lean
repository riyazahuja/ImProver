lemma isLocallySurjective_iff_locallySurjective_on_lightProfinite : IsLocallySurjective f ↔
    ∀ (S : LightProfinite) (y : Y.val.obj ⟨S⟩),
      (∃ (S' : LightProfinite) (φ : S' ⟶ S) (_ : Function.Surjective φ) (x : X.val.obj ⟨S'⟩),
        f.val.app ⟨S'⟩ x = Y.val.map ⟨φ⟩ y) := by
  rw [coherentTopology.isLocallySurjective_iff,
    regularTopology.isLocallySurjective_iff]
  /-
    A : Type u'
    inst✝² : CategoryTheory.Category.{v', u'} A
    inst✝¹ : CategoryTheory.ConcreteCategory A
    inst✝ : CategoryTheory.Limits.PreservesFiniteProducts (CategoryTheory.forget A)
    X Y : LightCondensed A
    f : Quiver.Hom X Y
    ⊢ Iff (∀ (X_1 : LightProfinite) (y : (CategoryTheory.forget A).obj (Y.val.obj  …
  -/
  simp_rw [LightProfinite.effectiveEpi_iff_surjective]
  /-
    🎉 no goals
  -/


lemma epi_iff_locallySurjective_on_lightProfinite : Epi f ↔
    ∀ (S : LightProfinite) (y : Y.val.obj ⟨S⟩),
      (∃ (S' : LightProfinite) (φ : S' ⟶ S) (_ : Function.Surjective φ) (x : X.val.obj ⟨S'⟩),
        f.val.app ⟨S'⟩ x = Y.val.map ⟨φ⟩ y) := by
  /-
    X Y : LightCondSet
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Epi f) (∀ (S : LightProfinite) (y : Y.val.obj { unop :=  …
  -/
  rw [← isLocallySurjective_iff_epi']
  /-
    X Y : LightCondSet
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Sheaf.IsLocallySurjective f) (∀ (S : LightProfinite) (y  …
  -/
  exact LightCondensed.isLocallySurjective_iff_locallySurjective_on_lightProfinite _ f
  /-
    🎉 no goals
  -/


lemma epi_iff_locallySurjective_on_lightProfinite : Epi f ↔
    ∀ (S : LightProfinite) (y : Y.val.obj ⟨S⟩),
      (∃ (S' : LightProfinite) (φ : S' ⟶ S) (_ : Function.Surjective φ) (x : X.val.obj ⟨S'⟩),
        f.val.app ⟨S'⟩ x = Y.val.map ⟨φ⟩ y) := by
  /-
    R : Type u
    inst✝ : Ring R
    X Y : LightCondMod R
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Epi f) (∀ (S : LightProfinite) (y : ↑(Y.val.obj { unop : …
  -/
  rw [← isLocallySurjective_iff_epi']
  /-
    R : Type u
    inst✝ : Ring R
    X Y : LightCondMod R
    f : Quiver.Hom X Y
    ⊢ Iff (CategoryTheory.Sheaf.IsLocallySurjective f) (∀ (S : LightProfinite) (y  …
  -/
  exact LightCondensed.isLocallySurjective_iff_locallySurjective_on_lightProfinite _ f
  /-
    🎉 no goals
  -/


instance : (LightCondensed.forget R).ReflectsEpimorphisms where
  reflects f hf := by
    /-
      R : Type u
      inst✝ : Ring R
      X Y : LightCondMod R
      f✝ : Quiver.Hom X Y
      X✝ Y✝ : LightCondMod R
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.Epi ((LightCondensed.forget R).map f)
      ⊢ CategoryTheory.Epi f
    -/
    rw [← Sheaf.isLocallySurjective_iff_epi'] at hf ⊢
    /-
      R : Type u
      inst✝ : Ring R
      X Y : LightCondMod R
      f✝ : Quiver.Hom X Y
      X✝ Y✝ : LightCondMod R
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.Sheaf.IsLocallySurjective ((LightCondensed.forget R).map f)
      ⊢ CategoryTheory.Sheaf.IsLocallySurjective f
    -/
    exact (Presheaf.isLocallySurjective_iff_whisker_forget _ f.val).mpr hf
    /-
      🎉 no goals
    -/


instance : (LightCondensed.forget R).PreservesEpimorphisms where
  preserves f hf := by
    /-
      R : Type u
      inst✝ : Ring R
      X Y : LightCondMod R
      f✝ : Quiver.Hom X Y
      X✝ Y✝ : LightCondMod R
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.Epi f
      ⊢ CategoryTheory.Epi ((LightCondensed.forget R).map f)
    -/
    rw [← Sheaf.isLocallySurjective_iff_epi'] at hf ⊢
    /-
      R : Type u
      inst✝ : Ring R
      X Y : LightCondMod R
      f✝ : Quiver.Hom X Y
      X✝ Y✝ : LightCondMod R
      f : Quiver.Hom X✝ Y✝
      hf : CategoryTheory.Sheaf.IsLocallySurjective f
      ⊢ CategoryTheory.Sheaf.IsLocallySurjective ((LightCondensed.forget R).map f)
    -/
    exact (Presheaf.isLocallySurjective_iff_whisker_forget _ f.val).mp hf
    /-
      🎉 no goals
    -/


include hc hF in
lemma epi_π_app_zero_of_epi : Epi (c.π.app ⟨0⟩) := by
  /-
    R : Type u_1
    inst✝ : Ring R
    F : CategoryTheory.Functor (Opposite Nat) (LightCondMod R)
    c : CategoryTheory.Limits.Cone F
    hc : CategoryTheory.Limits.IsLimit c
    hF : ∀ (n : Nat), CategoryTheory.Epi (F.map (CategoryTheory.homOfLE ⋯).op)
    ⊢ CategoryTheory.Epi (c.π.app { unop := 0 })
  -/
  apply Functor.epi_of_epi_map (forget R)
  /-
    R : Type u_1
    inst✝ : Ring R
    F : CategoryTheory.Functor (Opposite Nat) (LightCondMod R)
    c : CategoryTheory.Limits.Cone F
    hc : CategoryTheory.Limits.IsLimit c
    hF : ∀ (n : Nat), CategoryTheory.Epi (F.map (CategoryTheory.homOfLE ⋯).op)
    ⊢ CategoryTheory.Epi ((LightCondensed.forget R).map (c.π.app { unop := 0 }))
  -/
  change Epi (((forget R).mapCone c).π.app ⟨0⟩)
  /-
    R : Type u_1
    inst✝ : Ring R
    F : CategoryTheory.Functor (Opposite Nat) (LightCondMod R)
    c : CategoryTheory.Limits.Cone F
    hc : CategoryTheory.Limits.IsLimit c
    hF : ∀ (n : Nat), CategoryTheory.Epi (F.map (CategoryTheory.homOfLE ⋯).op)
    ⊢ CategoryTheory.Epi (((LightCondensed.forget R).mapCone c).π.app { unop := 0 })
  -/
  apply coherentTopology.epi_π_app_zero_of_epi
    /-
      case h
      R : Type u_1
      inst✝ : Ring R
      F : CategoryTheory.Functor (Opposite Nat) (LightCondMod R)
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      hF : ∀ (n : Nat), CategoryTheory.Epi (F.map (CategoryTheory.homOfLE ⋯).op)
      ⊢ ∀ (G : CategoryTheory.Functor (Opposite Nat) LightProfinite), (∀ (n : Nat),  …
    -/
  · simp only [LightProfinite.effectiveEpi_iff_surjective]
    /-
      case h
      R : Type u_1
      inst✝ : Ring R
      F : CategoryTheory.Functor (Opposite Nat) (LightCondMod R)
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      hF : ∀ (n : Nat), CategoryTheory.Epi (F.map (CategoryTheory.homOfLE ⋯).op)
      ⊢ ∀ (G : CategoryTheory.Functor (Opposite Nat) LightProfinite), (∀ (n : Nat),  …
    -/
    exact fun _ h ↦ Concrete.surjective_π_app_zero_of_surjective_map (limit.isLimit _) h
    /-
      🎉 no goals
    -/
    /-
      case hc
      R : Type u_1
      inst✝ : Ring R
      F : CategoryTheory.Functor (Opposite Nat) (LightCondMod R)
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      hF : ∀ (n : Nat), CategoryTheory.Epi (F.map (CategoryTheory.homOfLE ⋯).op)
      ⊢ CategoryTheory.Limits.IsLimit ((LightCondensed.forget R).mapCone c)
    -/
  · have := (freeForgetAdjunction R).isRightAdjoint
    /-
      case hc
      R : Type u_1
      inst✝ : Ring R
      F : CategoryTheory.Functor (Opposite Nat) (LightCondMod R)
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      hF : ∀ (n : Nat), CategoryTheory.Epi (F.map (CategoryTheory.homOfLE ⋯).op)
      this : (LightCondensed.forget R).IsRightAdjoint
      ⊢ CategoryTheory.Limits.IsLimit ((LightCondensed.forget R).mapCone c)
    -/
    exact isLimitOfPreserves _ hc
    /-
      🎉 no goals
    -/
    /-
      case hF
      R : Type u_1
      inst✝ : Ring R
      F : CategoryTheory.Functor (Opposite Nat) (LightCondMod R)
      c : CategoryTheory.Limits.Cone F
      hc : CategoryTheory.Limits.IsLimit c
      hF : ∀ (n : Nat), CategoryTheory.Epi (F.map (CategoryTheory.homOfLE ⋯).op)
      ⊢ ∀ (n : Nat), CategoryTheory.Epi ((F.comp (LightCondensed.forget R)).map (Cat …
    -/
  · exact fun _ ↦ (forget R).map_epi _
    /-
      🎉 no goals
    -/


instance : Epi (Limits.Pi.map f) := by
  /-
    n : Nat
    R : Type u
    inst✝¹ : Ring R
    M N : Nat → LightCondMod R
    f : (n : Nat) → Quiver.Hom (M n) (N n)
    inst✝ : ∀ (n : Nat), CategoryTheory.Epi (f n)
    ⊢ CategoryTheory.Epi (CategoryTheory.Limits.Pi.map f)
  -/
  have : Limits.Pi.map f = (cone f).π.app ⟨0⟩ := rfl
  /-
    n : Nat
    R : Type u
    inst✝¹ : Ring R
    M N : Nat → LightCondMod R
    f : (n : Nat) → Quiver.Hom (M n) (N n)
    inst✝ : ∀ (n : Nat), CategoryTheory.Epi (f n)
    this : Eq (CategoryTheory.Limits.Pi.map f) ((CategoryTheory.Limits.SequentialP …
    ⊢ CategoryTheory.Epi (CategoryTheory.Limits.Pi.map f)
  -/
  rw [this]
  /-
    n : Nat
    R : Type u
    inst✝¹ : Ring R
    M N : Nat → LightCondMod R
    f : (n : Nat) → Quiver.Hom (M n) (N n)
    inst✝ : ∀ (n : Nat), CategoryTheory.Epi (f n)
    this : Eq (CategoryTheory.Limits.Pi.map f) ((CategoryTheory.Limits.SequentialP …
    ⊢ CategoryTheory.Epi ((CategoryTheory.Limits.SequentialProduct.cone f).π.app { …
  -/
  exact epi_π_app_zero_of_epi R (isLimit f) (fun n ↦ by simp; infer_instance)
  /-
    🎉 no goals
  -/


instance : (lim (J := Discrete ℕ) (C := LightCondMod R)).PreservesEpimorphisms where
  preserves f _ := by
    have : lim.map f = (Pi.isoLimit _).inv ≫ Limits.Pi.map (f.app ⟨·⟩) ≫ (Pi.isoLimit _).hom := by
      apply limit.hom_ext
      intro ⟨n⟩
      simp only [lim_obj, lim_map, limMap, IsLimit.map, limit.isLimit_lift, limit.lift_π,
        Cones.postcompose_obj_pt, limit.cone_x, Cones.postcompose_obj_π, NatTrans.comp_app,
        Functor.const_obj_obj, limit.cone_π, Pi.isoLimit, Limits.Pi.map, Category.assoc,
        limit.conePointUniqueUpToIso_hom_comp, Pi.cone_pt, Pi.cone_π, Discrete.natTrans_app,
        Discrete.functor_obj_eq_as]
      erw [IsLimit.conePointUniqueUpToIso_inv_comp_assoc]
      rfl
    /-
      n : Nat
      R : Type u
      inst✝¹ : Ring R
      M N : Nat → LightCondMod R
      f✝ : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝ : ∀ (n : Nat), CategoryTheory.Epi (f✝ n)
      X✝ Y✝ : CategoryTheory.Functor (CategoryTheory.Discrete Nat) (LightCondMod R)
      f : Quiver.Hom X✝ Y✝
      x✝ : CategoryTheory.Epi f
      this : Eq (CategoryTheory.Limits.lim.map f) (CategoryTheory.CategoryStruct.com …
      ⊢ CategoryTheory.Epi (CategoryTheory.Limits.lim.map f)
    -/
    rw [this]
    /-
      n : Nat
      R : Type u
      inst✝¹ : Ring R
      M N : Nat → LightCondMod R
      f✝ : (n : Nat) → Quiver.Hom (M n) (N n)
      inst✝ : ∀ (n : Nat), CategoryTheory.Epi (f✝ n)
      X✝ Y✝ : CategoryTheory.Functor (CategoryTheory.Discrete Nat) (LightCondMod R)
      f : Quiver.Hom X✝ Y✝
      x✝ : CategoryTheory.Epi f
      this : Eq (CategoryTheory.Limits.lim.map f) (CategoryTheory.CategoryStruct.com …
      ⊢ CategoryTheory.Epi (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limit …
    -/
    infer_instance
    /-
      🎉 no goals
    -/


