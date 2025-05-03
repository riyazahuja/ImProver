/-- The diagonal object of a morphism `f : X ⟶ Y` is `Δ_{X/Y} := pullback f f`. -/
abbrev diagonalObj : C :=
  pullback f f


/-- The diagonal morphism `X ⟶ Δ_{X/Y}` for a morphism `f : X ⟶ Y`. -/
def diagonal : X ⟶ diagonalObj f :=
  pullback.lift (𝟙 _) (𝟙 _) rfl


@[reassoc (attr := simp)]
theorem diagonal_fst : diagonal f ≫ pullback.fst _ _ = 𝟙 _ :=
  pullback.lift_fst _ _ _


@[reassoc (attr := simp)]
theorem diagonal_snd : diagonal f ≫ pullback.snd _ _ = 𝟙 _ :=
  pullback.lift_snd _ _ _


instance : IsSplitMono (diagonal f) :=
  ⟨⟨⟨pullback.fst _ _, diagonal_fst f⟩⟩⟩


instance : IsSplitEpi (pullback.fst f f) :=
  ⟨⟨⟨diagonal f, diagonal_fst f⟩⟩⟩


instance : IsSplitEpi (pullback.snd f f) :=
  ⟨⟨⟨diagonal f, diagonal_snd f⟩⟩⟩


instance [Mono f] : IsIso (diagonal f) := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    X Y Z : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasPullback f f
    inst✝ : CategoryTheory.Mono f
    ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pullback.diagonal f)
  -/
  rw [(IsIso.inv_eq_of_inv_hom_id (diagonal_fst f)).symm]
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    X Y Z : C
    f : Quiver.Hom X Y
    inst✝¹ : CategoryTheory.Limits.HasPullback f f
    inst✝ : CategoryTheory.Mono f
    ⊢ CategoryTheory.IsIso (CategoryTheory.inv (CategoryTheory.Limits.pullback.fst …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma isIso_diagonal_iff : IsIso (diagonal f) ↔ Mono f :=
  ⟨fun H ↦ ⟨fun _ _ e ↦ by rw [← lift_fst _ _ e, (cancel_epi (g := fst f f) (h := snd f f)
    (diagonal f)).mp (by simp), lift_snd]⟩, fun _ ↦ inferInstance⟩


/-- The two projections `Δ_{X/Y} ⟶ X` form a kernel pair for `f : X ⟶ Y`. -/
theorem diagonal_isKernelPair : IsKernelPair f (pullback.fst f f) (pullback.snd f f) :=
  IsPullback.of_hasPullback f f


@[reassoc (attr := simp)]
theorem pullback_diagonal_map_snd_fst_fst :
    (pullback.snd (diagonal f)
      (map (i₁ ≫ snd f i) (i₂ ≫ snd f i) f f (i₁ ≫ fst f i) (i₂ ≫ fst f i) i
            /-
              C : Type u_1
              inst✝¹ : CategoryTheory.Category.{?u.11291, u_1} C
              X Y Z : C
              inst✝ : CategoryTheory.Limits.HasPullbacks C
              U V₁ V₂ : C
              f : Quiver.Hom X Y
              i : Quiver.Hom U Y
              i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
              i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp i …
            -/
            /-
              🎉 no goals
            -/
        (by simp [condition]) (by simp [condition]))) ≫
                                  /-
                                    🎉 no goals
                                  -/
      fst _ _ ≫ i₁ ≫ fst _ _ =
      pullback.fst _ _ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    U V₁ V₂ : C
    f : Quiver.Hom X Y
    i : Quiver.Hom U Y
    i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
    i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd ( …
  -/
  conv_rhs => rw [← Category.comp_id (pullback.fst _ _)]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    U V₁ V₂ : C
    f : Quiver.Hom X Y
    i : Quiver.Hom U Y
    i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
    i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd ( …
  -/
  rw [← diagonal_fst f, pullback.condition_assoc, pullback.lift_fst]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem pullback_diagonal_map_snd_snd_fst :
    (pullback.snd (diagonal f)
      (map (i₁ ≫ snd f i) (i₂ ≫ snd f i) f f (i₁ ≫ fst f i) (i₂ ≫ fst f i) i
            /-
              C : Type u_1
              inst✝¹ : CategoryTheory.Category.{?u.337725, u_1} C
              X Y Z : C
              inst✝ : CategoryTheory.Limits.HasPullbacks C
              U V₁ V₂ : C
              f : Quiver.Hom X Y
              i : Quiver.Hom U Y
              i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
              i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp i …
            -/
            /-
              🎉 no goals
            -/
        (by simp [condition]) (by simp [condition]))) ≫
                                  /-
                                    🎉 no goals
                                  -/
      snd _ _ ≫ i₂ ≫ fst _ _ =
      pullback.fst _ _ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    U V₁ V₂ : C
    f : Quiver.Hom X Y
    i : Quiver.Hom U Y
    i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
    i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd ( …
  -/
  conv_rhs => rw [← Category.comp_id (pullback.fst _ _)]
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    U V₁ V₂ : C
    f : Quiver.Hom X Y
    i : Quiver.Hom U Y
    i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
    i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd ( …
  -/
  rw [← diagonal_snd f, pullback.condition_assoc, pullback.lift_snd]
  /-
    🎉 no goals
  -/


/-- The underlying map of `pullbackDiagonalIso` -/
abbrev pullbackDiagonalMapIso.hom :
    pullback (diagonal f)
        (map (i₁ ≫ snd _ _) (i₂ ≫ snd _ _) f f (i₁ ≫ fst _ _) (i₂ ≫ fst _ _) i
              /-
                C : Type u_1
                inst✝² : CategoryTheory.Category.{?u.665791, u_1} C
                X Y Z : C
                inst✝¹ : CategoryTheory.Limits.HasPullbacks C
                U V₁ V₂ : C
                f : Quiver.Hom X Y
                i : Quiver.Hom U Y
                i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
                i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
                inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp i …
              -/
          (by simp only [Category.assoc, condition])
              /-
                🎉 no goals
              -/
              /-
                C : Type u_1
                inst✝² : CategoryTheory.Category.{?u.665791, u_1} C
                X Y Z : C
                inst✝¹ : CategoryTheory.Limits.HasPullbacks C
                U V₁ V₂ : C
                f : Quiver.Hom X Y
                i : Quiver.Hom U Y
                i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
                i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
                inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp i …
              -/
          (by simp only [Category.assoc, condition])) ⟶
              /-
                🎉 no goals
              -/
      pullback i₁ i₂ :=
  pullback.lift (pullback.snd _ _ ≫ pullback.fst _ _) (pullback.snd _ _ ≫ pullback.snd _ _) (by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{?u.665791, u_1} C
    X Y Z : C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    U V₁ V₂ : C
    f : Quiver.Hom X Y
    i : Quiver.Hom U Y
    i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
    i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
    inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  ext
  · simp only [Category.assoc, pullback_diagonal_map_snd_fst_fst,
      pullback_diagonal_map_snd_snd_fst]
    /-
      case h₁
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.665791, u_1} C
      X Y Z : C
      inst✝¹ : CategoryTheory.Limits.HasPullbacks C
      U V₁ V₂ : C
      f : Quiver.Hom X Y
      i : Quiver.Hom U Y
      i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
      i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
      inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · simp only [Category.assoc, condition])
    /-
      🎉 no goals
    -/


/-- The underlying inverse of `pullbackDiagonalIso` -/
abbrev pullbackDiagonalMapIso.inv : pullback i₁ i₂ ⟶
    pullback (diagonal f)
        (map (i₁ ≫ snd _ _) (i₂ ≫ snd _ _) f f (i₁ ≫ fst _ _) (i₂ ≫ fst _ _) i
              /-
                C : Type u_1
                inst✝² : CategoryTheory.Category.{?u.677049, u_1} C
                X Y Z : C
                inst✝¹ : CategoryTheory.Limits.HasPullbacks C
                U V₁ V₂ : C
                f : Quiver.Hom X Y
                i : Quiver.Hom U Y
                i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
                i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
                inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp i …
              -/
          (by simp only [Category.assoc, condition])
              /-
                🎉 no goals
              -/
              /-
                C : Type u_1
                inst✝² : CategoryTheory.Category.{?u.677049, u_1} C
                X Y Z : C
                inst✝¹ : CategoryTheory.Limits.HasPullbacks C
                U V₁ V₂ : C
                f : Quiver.Hom X Y
                i : Quiver.Hom U Y
                i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
                i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
                inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp i …
              -/
          (by simp only [Category.assoc, condition])) :=
              /-
                🎉 no goals
              -/
    pullback.lift (pullback.fst _ _ ≫ i₁ ≫ pullback.fst _ _)
      (pullback.map _ _ _ _ (𝟙 _) (𝟙 _) (pullback.snd _ _) (Category.id_comp _).symm
        (Category.id_comp _).symm) (by
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.677049, u_1} C
          X Y Z : C
          inst✝¹ : CategoryTheory.Limits.HasPullbacks C
          U V₁ V₂ : C
          f : Quiver.Hom X Y
          i : Quiver.Hom U Y
          i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
          i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
          inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
        ext
        · simp only [Category.assoc, diagonal_fst, Category.comp_id, limit.lift_π,
            PullbackCone.mk_pt, PullbackCone.mk_π_app, limit.lift_π_assoc, cospan_left]
        · simp only [condition_assoc, Category.assoc, diagonal_snd, Category.comp_id, limit.lift_π,
            PullbackCone.mk_pt, PullbackCone.mk_π_app, limit.lift_π_assoc, cospan_right])


/-- This iso witnesses the fact that
given `f : X ⟶ Y`, `i : U ⟶ Y`, and `i₁ : V₁ ⟶ X ×[Y] U`, `i₂ : V₂ ⟶ X ×[Y] U`, the diagram

```
V₁ ×[X ×[Y] U] V₂ ⟶ V₁ ×[U] V₂
        |                 |
        |                 |
        ↓                 ↓
        X         ⟶   X ×[Y] X
```

is a pullback square.
Also see `pullback_fst_map_snd_isPullback`.
-/
def pullbackDiagonalMapIso :
    pullback (diagonal f)
        (map (i₁ ≫ snd _ _) (i₂ ≫ snd _ _) f f (i₁ ≫ fst _ _) (i₂ ≫ fst _ _) i
              /-
                C : Type u_1
                inst✝² : CategoryTheory.Category.{?u.689400, u_1} C
                X Y Z : C
                inst✝¹ : CategoryTheory.Limits.HasPullbacks C
                U V₁ V₂ : C
                f : Quiver.Hom X Y
                i : Quiver.Hom U Y
                i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
                i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
                inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp i …
              -/
          (by simp only [Category.assoc, condition])
              /-
                🎉 no goals
              -/
              /-
                C : Type u_1
                inst✝² : CategoryTheory.Category.{?u.689400, u_1} C
                X Y Z : C
                inst✝¹ : CategoryTheory.Limits.HasPullbacks C
                U V₁ V₂ : C
                f : Quiver.Hom X Y
                i : Quiver.Hom U Y
                i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
                i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
                inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp i …
              -/
          (by simp only [Category.assoc, condition])) ≅
              /-
                🎉 no goals
              -/
      pullback i₁ i₂ where
  hom := pullbackDiagonalMapIso.hom f i i₁ i₂
  inv := pullbackDiagonalMapIso.inv f i i₁ i₂


@[reassoc (attr := simp)]
theorem pullbackDiagonalMapIso.hom_fst :
    (pullbackDiagonalMapIso f i i₁ i₂).hom ≫ pullback.fst _ _ =
      pullback.snd _ _ ≫ pullback.fst _ _ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    U V₁ V₂ : C
    f : Quiver.Hom X Y
    i : Quiver.Hom U Y
    i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
    i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
    inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackDiagon …
  -/
  delta pullbackDiagonalMapIso
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    U V₁ V₂ : C
    f : Quiver.Hom X Y
    i : Quiver.Hom U Y
    i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
    i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
    inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp { hom := CategoryTheory.Limits.pullba …
  -/
  simp only [limit.lift_π, PullbackCone.mk_pt, PullbackCone.mk_π_app]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem pullbackDiagonalMapIso.hom_snd :
    (pullbackDiagonalMapIso f i i₁ i₂).hom ≫ pullback.snd _ _ =
      pullback.snd _ _ ≫ pullback.snd _ _ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    U V₁ V₂ : C
    f : Quiver.Hom X Y
    i : Quiver.Hom U Y
    i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
    i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
    inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackDiagon …
  -/
  delta pullbackDiagonalMapIso
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    U V₁ V₂ : C
    f : Quiver.Hom X Y
    i : Quiver.Hom U Y
    i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
    i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
    inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp { hom := CategoryTheory.Limits.pullba …
  -/
  simp only [limit.lift_π, PullbackCone.mk_pt, PullbackCone.mk_π_app]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem pullbackDiagonalMapIso.inv_fst :
    (pullbackDiagonalMapIso f i i₁ i₂).inv ≫ pullback.fst _ _ =
      pullback.fst _ _ ≫ i₁ ≫ pullback.fst _ _ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    U V₁ V₂ : C
    f : Quiver.Hom X Y
    i : Quiver.Hom U Y
    i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
    i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
    inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackDiagon …
  -/
  delta pullbackDiagonalMapIso
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    U V₁ V₂ : C
    f : Quiver.Hom X Y
    i : Quiver.Hom U Y
    i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
    i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
    inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp { hom := CategoryTheory.Limits.pullba …
  -/
  simp only [limit.lift_π, PullbackCone.mk_pt, PullbackCone.mk_π_app]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem pullbackDiagonalMapIso.inv_snd_fst :
    (pullbackDiagonalMapIso f i i₁ i₂).inv ≫ pullback.snd _ _ ≫ pullback.fst _ _ =
      pullback.fst _ _ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    U V₁ V₂ : C
    f : Quiver.Hom X Y
    i : Quiver.Hom U Y
    i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
    i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
    inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackDiagon …
  -/
  delta pullbackDiagonalMapIso
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    U V₁ V₂ : C
    f : Quiver.Hom X Y
    i : Quiver.Hom U Y
    i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
    i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
    inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp { hom := CategoryTheory.Limits.pullba …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem pullbackDiagonalMapIso.inv_snd_snd :
    (pullbackDiagonalMapIso f i i₁ i₂).inv ≫ pullback.snd _ _ ≫ pullback.snd _ _ =
      pullback.snd _ _ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    U V₁ V₂ : C
    f : Quiver.Hom X Y
    i : Quiver.Hom U Y
    i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
    i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
    inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackDiagon …
  -/
  delta pullbackDiagonalMapIso
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    U V₁ V₂ : C
    f : Quiver.Hom X Y
    i : Quiver.Hom U Y
    i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
    i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
    inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
    ⊢ Eq (CategoryTheory.CategoryStruct.comp { hom := CategoryTheory.Limits.pullba …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem pullback_fst_map_snd_isPullback :
    IsPullback (fst _ _ ≫ i₁ ≫ fst _ _)
      (map i₁ i₂ (i₁ ≫ snd _ _) (i₂ ≫ snd _ _) _ _ _
        (Category.id_comp _).symm (Category.id_comp _).symm)
      (diagonal f)
                                                                                 /-
                                                                                   C : Type u_1
                                                                                   inst✝² : CategoryTheory.Category.{?u.739430, u_1} C
                                                                                   X Y Z : C
                                                                                   inst✝¹ : CategoryTheory.Limits.HasPullbacks C
                                                                                   U V₁ V₂ : C
                                                                                   f : Quiver.Hom X Y
                                                                                   i : Quiver.Hom U Y
                                                                                   i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
                                                                                   i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
                                                                                   inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
                                                                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp i …
                                                                                 -/
      (map (i₁ ≫ snd _ _) (i₂ ≫ snd _ _) f f (i₁ ≫ fst _ _) (i₂ ≫ fst _ _) i (by simp [condition])
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
            /-
              C : Type u_1
              inst✝² : CategoryTheory.Category.{?u.739430, u_1} C
              X Y Z : C
              inst✝¹ : CategoryTheory.Limits.HasPullbacks C
              U V₁ V₂ : C
              f : Quiver.Hom X Y
              i : Quiver.Hom U Y
              i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
              i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
              inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp i …
            -/
        (by simp [condition])) :=
            /-
              🎉 no goals
            -/
                                 /-
                                   C : Type u_1
                                   inst✝² : CategoryTheory.Category.{u_2, u_1} C
                                   X Y : C
                                   inst✝¹ : CategoryTheory.Limits.HasPullbacks C
                                   U V₁ V₂ : C
                                   f : Quiver.Hom X Y
                                   i : Quiver.Hom U Y
                                   i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
                                   i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
                                   inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
                                 -/
                                         /-
                                           🎉 no goals
                                         -/
  IsPullback.of_iso_pullback ⟨by ext <;> simp [condition_assoc]⟩
                                         /-
                                           🎉 no goals
                                         -/
    (pullbackDiagonalMapIso f i i₁ i₂).symm (pullbackDiagonalMapIso.inv_fst f i i₁ i₂)
        /-
          C : Type u_1
          inst✝² : CategoryTheory.Category.{u_2, u_1} C
          X Y : C
          inst✝¹ : CategoryTheory.Limits.HasPullbacks C
          U V₁ V₂ : C
          f : Quiver.Hom X Y
          i : Quiver.Hom U Y
          i₁ : Quiver.Hom V₁ (CategoryTheory.Limits.pullback f i)
          i₂ : Quiver.Hom V₂ (CategoryTheory.Limits.pullback f i)
          inst✝ : CategoryTheory.Limits.HasPullback i₁ i₂
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackDiagon …
        -/
    (by aesop_cat)
        /-
          🎉 no goals
        -/


/-- This iso witnesses the fact that
given `f : X ⟶ T`, `g : Y ⟶ T`, and `i : T ⟶ S`, the diagram

```
X ×ₜ Y ⟶ X ×ₛ Y
  |         |
  |         |
  ↓         ↓
  T    ⟶  T ×ₛ T
```

is a pullback square.
Also see `pullback_map_diagonal_isPullback`.
-/
def pullbackDiagonalMapIdIso :
    pullback (diagonal i)
        (pullback.map (f ≫ i) (g ≫ i) i i f g (𝟙 _) (Category.comp_id _) (Category.comp_id _)) ≅
      pullback f g := by
  refine ?_ ≪≫
    pullbackDiagonalMapIso i (𝟙 _) (f ≫ inv (pullback.fst _ _)) (g ≫ inv (pullback.fst _ _)) ≪≫ ?_
  · refine @asIso _ _ _ _ (pullback.map _ _ _ _ (𝟙 T) ((pullback.congrHom ?_ ?_).hom) (𝟙 _) ?_ ?_)
      ?_
      /-
        case refine_1.refine_1
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{?u.759847, u_1} C
        X Y Z : C
        inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
        S T : C
        f : Quiver.Hom X T
        g : Quiver.Hom Y T
        i : Quiver.Hom T S
        inst✝³ : CategoryTheory.Limits.HasPullback i i
        inst✝² : CategoryTheory.Limits.HasPullback f g
        inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
        inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp f i) (CategoryTheory.CategoryStruct.c …
      -/
    · rw [← Category.comp_id (pullback.snd ..), ← condition, Category.assoc, IsIso.inv_hom_id_assoc]
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_2
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{?u.759847, u_1} C
        X Y Z : C
        inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
        S T : C
        f : Quiver.Hom X T
        g : Quiver.Hom Y T
        i : Quiver.Hom T S
        inst✝³ : CategoryTheory.Limits.HasPullback i i
        inst✝² : CategoryTheory.Limits.HasPullback f g
        inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
        inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp g i) (CategoryTheory.CategoryStruct.c …
      -/
    · rw [← Category.comp_id (pullback.snd ..), ← condition, Category.assoc, IsIso.inv_hom_id_assoc]
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_3
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{?u.759847, u_1} C
        X Y Z : C
        inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
        S T : C
        f : Quiver.Hom X T
        g : Quiver.Hom Y T
        i : Quiver.Hom T S
        inst✝³ : CategoryTheory.Limits.HasPullback i i
        inst✝² : CategoryTheory.Limits.HasPullback f g
        inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
        inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.diago …
      -/
    · rw [Category.comp_id, Category.id_comp]
      /-
        🎉 no goals
      -/
      /-
        case refine_1.refine_4
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{?u.759847, u_1} C
        X Y Z : C
        inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
        S T : C
        f : Quiver.Hom X T
        g : Quiver.Hom Y T
        i : Quiver.Hom T S
        inst✝³ : CategoryTheory.Limits.HasPullback i i
        inst✝² : CategoryTheory.Limits.HasPullback f g
        inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
        inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.map ( …
      -/
              /-
                🎉 no goals
              -/
    · ext <;> simp
              /-
                🎉 no goals
              -/
      /-
        case refine_1.refine_5
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{?u.759847, u_1} C
        X Y Z : C
        inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
        S T : C
        f : Quiver.Hom X T
        g : Quiver.Hom Y T
        i : Quiver.Hom T S
        inst✝³ : CategoryTheory.Limits.HasPullback i i
        inst✝² : CategoryTheory.Limits.HasPullback f g
        inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
        inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
        ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pullback.map (CategoryTheory.Lim …
      -/
    · infer_instance
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{?u.759847, u_1} C
      X Y Z : C
      inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
      S T : C
      f : Quiver.Hom X T
      g : Quiver.Hom Y T
      i : Quiver.Hom T S
      inst✝³ : CategoryTheory.Limits.HasPullback i i
      inst✝² : CategoryTheory.Limits.HasPullback f g
      inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
      inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
      ⊢ CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.CategoryS …
    -/
  · refine @asIso _ _ _ _ (pullback.map _ _ _ _ (𝟙 _) (𝟙 _) (pullback.fst _ _) ?_ ?_) ?_
      /-
        case refine_2.refine_1
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{?u.759847, u_1} C
        X Y Z : C
        inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
        S T : C
        f : Quiver.Hom X T
        g : Quiver.Hom Y T
        i : Quiver.Hom T S
        inst✝³ : CategoryTheory.Limits.HasPullback i i
        inst✝² : CategoryTheory.Limits.HasPullback f g
        inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
        inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp f …
      -/
    · rw [Category.assoc, IsIso.inv_hom_id, Category.comp_id, Category.id_comp]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{?u.759847, u_1} C
        X Y Z : C
        inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
        S T : C
        f : Quiver.Hom X T
        g : Quiver.Hom Y T
        i : Quiver.Hom T S
        inst✝³ : CategoryTheory.Limits.HasPullback i i
        inst✝² : CategoryTheory.Limits.HasPullback f g
        inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
        inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp g …
      -/
    · rw [Category.assoc, IsIso.inv_hom_id, Category.comp_id, Category.id_comp]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_3
        C : Type u_1
        inst✝⁵ : CategoryTheory.Category.{?u.759847, u_1} C
        X Y Z : C
        inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
        S T : C
        f : Quiver.Hom X T
        g : Quiver.Hom Y T
        i : Quiver.Hom T S
        inst✝³ : CategoryTheory.Limits.HasPullback i i
        inst✝² : CategoryTheory.Limits.HasPullback f g
        inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
        inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
        ⊢ CategoryTheory.IsIso (CategoryTheory.Limits.pullback.map (CategoryTheory.Cat …
      -/
    · infer_instance
      /-
        🎉 no goals
      -/


@[reassoc (attr := simp)]
theorem pullbackDiagonalMapIdIso_hom_fst :
    (pullbackDiagonalMapIdIso f g i).hom ≫ pullback.fst _ _ =
      pullback.snd _ _ ≫ pullback.fst _ _ := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
    S T : C
    f : Quiver.Hom X T
    g : Quiver.Hom Y T
    i : Quiver.Hom T S
    inst✝³ : CategoryTheory.Limits.HasPullback i i
    inst✝² : CategoryTheory.Limits.HasPullback f g
    inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
    inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackDiagon …
  -/
  delta pullbackDiagonalMapIdIso
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
    S T : C
    f : Quiver.Hom X T
    g : Quiver.Hom Y T
    i : Quiver.Hom T S
    inst✝³ : CategoryTheory.Limits.HasPullback i i
    inst✝² : CategoryTheory.Limits.HasPullback f g
    inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
    inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.asIso (CategoryTheor …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem pullbackDiagonalMapIdIso_hom_snd :
    (pullbackDiagonalMapIdIso f g i).hom ≫ pullback.snd _ _ =
      pullback.snd _ _ ≫ pullback.snd _ _ := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
    S T : C
    f : Quiver.Hom X T
    g : Quiver.Hom Y T
    i : Quiver.Hom T S
    inst✝³ : CategoryTheory.Limits.HasPullback i i
    inst✝² : CategoryTheory.Limits.HasPullback f g
    inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
    inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackDiagon …
  -/
  delta pullbackDiagonalMapIdIso
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
    S T : C
    f : Quiver.Hom X T
    g : Quiver.Hom Y T
    i : Quiver.Hom T S
    inst✝³ : CategoryTheory.Limits.HasPullback i i
    inst✝² : CategoryTheory.Limits.HasPullback f g
    inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
    inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.asIso (CategoryTheor …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem pullbackDiagonalMapIdIso_inv_fst :
    (pullbackDiagonalMapIdIso f g i).inv ≫ pullback.fst _ _ = pullback.fst _ _ ≫ f := by
  rw [Iso.inv_comp_eq, ← Category.comp_id (pullback.fst _ _), ← diagonal_fst i,
    pullback.condition_assoc]
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
    S T : C
    f : Quiver.Hom X T
    g : Quiver.Hom Y T
    i : Quiver.Hom T S
    inst✝³ : CategoryTheory.Limits.HasPullback i i
    inst✝² : CategoryTheory.Limits.HasPullback f g
    inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
    inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd ( …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem pullbackDiagonalMapIdIso_inv_snd_fst :
    (pullbackDiagonalMapIdIso f g i).inv ≫ pullback.snd _ _ ≫ pullback.fst _ _ =
      pullback.fst _ _ := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
    S T : C
    f : Quiver.Hom X T
    g : Quiver.Hom Y T
    i : Quiver.Hom T S
    inst✝³ : CategoryTheory.Limits.HasPullback i i
    inst✝² : CategoryTheory.Limits.HasPullback f g
    inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
    inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackDiagon …
  -/
  rw [Iso.inv_comp_eq]
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
    S T : C
    f : Quiver.Hom X T
    g : Quiver.Hom Y T
    i : Quiver.Hom T S
    inst✝³ : CategoryTheory.Limits.HasPullback i i
    inst✝² : CategoryTheory.Limits.HasPullback f g
    inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
    inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd ( …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem pullbackDiagonalMapIdIso_inv_snd_snd :
    (pullbackDiagonalMapIdIso f g i).inv ≫ pullback.snd _ _ ≫ pullback.snd _ _ =
      pullback.snd _ _ := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
    S T : C
    f : Quiver.Hom X T
    g : Quiver.Hom Y T
    i : Quiver.Hom T S
    inst✝³ : CategoryTheory.Limits.HasPullback i i
    inst✝² : CategoryTheory.Limits.HasPullback f g
    inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
    inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackDiagon …
  -/
  rw [Iso.inv_comp_eq]
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
    S T : C
    f : Quiver.Hom X T
    g : Quiver.Hom Y T
    i : Quiver.Hom T S
    inst✝³ : CategoryTheory.Limits.HasPullback i i
    inst✝² : CategoryTheory.Limits.HasPullback f g
    inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
    inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd ( …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem pullback.diagonal_comp (f : X ⟶ Y) (g : Y ⟶ Z) [HasPullback f f] [HasPullback g g]
    [HasPullback (f ≫ g) (f ≫ g)] :
    diagonal (f ≫ g) = diagonal f ≫ (pullbackDiagonalMapIdIso f f g).inv ≫ pullback.snd _ _ := by
  /-
    C : Type u_1
    inst✝⁴ : CategoryTheory.Category.{u_2, u_1} C
    X Y Z : C
    inst✝³ : CategoryTheory.Limits.HasPullbacks C
    f : Quiver.Hom X Y
    g : Quiver.Hom Y Z
    inst✝² : CategoryTheory.Limits.HasPullback f f
    inst✝¹ : CategoryTheory.Limits.HasPullback g g
    inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp  …
    ⊢ Eq (CategoryTheory.Limits.pullback.diagonal (CategoryTheory.CategoryStruct.c …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


theorem pullback_map_diagonal_isPullback :
    IsPullback (pullback.fst _ _ ≫ f)
      (pullback.map f g (f ≫ i) (g ≫ i) _ _ i (Category.id_comp _).symm (Category.id_comp _).symm)
      (diagonal i)
      (pullback.map (f ≫ i) (g ≫ i) i i f g (𝟙 _) (Category.comp_id _) (Category.comp_id _)) := by
  /-
    C : Type u_1
    inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
    S T : C
    f : Quiver.Hom X T
    g : Quiver.Hom Y T
    i : Quiver.Hom T S
    inst✝³ : CategoryTheory.Limits.HasPullback i i
    inst✝² : CategoryTheory.Limits.HasPullback f g
    inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
    inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
    ⊢ CategoryTheory.IsPullback (CategoryTheory.CategoryStruct.comp (CategoryTheor …
  -/
  apply IsPullback.of_iso_pullback _ (pullbackDiagonalMapIdIso f g i).symm
    /-
      case w₁
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
      S T : C
      f : Quiver.Hom X T
      g : Quiver.Hom Y T
      i : Quiver.Hom T S
      inst✝³ : CategoryTheory.Limits.HasPullback i i
      inst✝² : CategoryTheory.Limits.HasPullback f g
      inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
      inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackDiagon …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case w₂
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
      S T : C
      f : Quiver.Hom X T
      g : Quiver.Hom Y T
      i : Quiver.Hom T S
      inst✝³ : CategoryTheory.Limits.HasPullback i i
      inst✝² : CategoryTheory.Limits.HasPullback f g
      inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
      inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackDiagon …
    -/
            /-
              🎉 no goals
            -/
  · ext <;> simp
            /-
              🎉 no goals
            -/
    /-
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
      S T : C
      f : Quiver.Hom X T
      g : Quiver.Hom Y T
      i : Quiver.Hom T S
      inst✝³ : CategoryTheory.Limits.HasPullback i i
      inst✝² : CategoryTheory.Limits.HasPullback f g
      inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
      inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
      ⊢ CategoryTheory.CommSq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Li …
    -/
  · constructor
    /-
      case w
      C : Type u_1
      inst✝⁵ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
      S T : C
      f : Quiver.Hom X T
      g : Quiver.Hom Y T
      i : Quiver.Hom T S
      inst✝³ : CategoryTheory.Limits.HasPullback i i
      inst✝² : CategoryTheory.Limits.HasPullback f g
      inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
      inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
            /-
              🎉 no goals
            -/
    ext <;> simp [condition]
            /-
              🎉 no goals
            -/


/-- The diagonal object of `X ×[Z] Y ⟶ X` is isomorphic to `Δ_{Y/Z} ×[Z] X`. -/
def diagonalObjPullbackFstIso {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) :
    diagonalObj (pullback.fst f g) ≅
      pullback (pullback.snd _ _ ≫ g : diagonalObj g ⟶ Z) f :=
  pullbackRightPullbackFstIso _ _ _ ≪≫
    pullback.congrHom pullback.condition rfl ≪≫
      pullbackAssoc _ _ _ _ ≪≫ pullbackSymmetry _ _ ≪≫ pullback.congrHom pullback.condition rfl


@[reassoc (attr := simp)]
theorem diagonalObjPullbackFstIso_hom_fst_fst {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) :
    (diagonalObjPullbackFstIso f g).hom ≫ pullback.fst _ _ ≫ pullback.fst _ _ =
      pullback.fst _ _ ≫ pullback.snd _ _ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.diagonalObjPul …
  -/
  delta diagonalObjPullbackFstIso
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.pullbackRight …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem diagonalObjPullbackFstIso_hom_fst_snd {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) :
    (diagonalObjPullbackFstIso f g).hom ≫ pullback.fst _ _ ≫ pullback.snd _ _ =
      pullback.snd _ _ ≫ pullback.snd _ _ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.diagonalObjPul …
  -/
  delta diagonalObjPullbackFstIso
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.pullbackRight …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem diagonalObjPullbackFstIso_hom_snd {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) :
    (diagonalObjPullbackFstIso f g).hom ≫ pullback.snd _ _ =
      pullback.fst _ _ ≫ pullback.fst _ _ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.diagonalObjPul …
  -/
  delta diagonalObjPullbackFstIso
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.pullbackRight …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem diagonalObjPullbackFstIso_inv_fst_fst {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) :
    (diagonalObjPullbackFstIso f g).inv ≫ pullback.fst _ _ ≫ pullback.fst _ _ =
      pullback.snd _ _ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.diagonalObjPul …
  -/
  delta diagonalObjPullbackFstIso
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.pullbackRight …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem diagonalObjPullbackFstIso_inv_fst_snd {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) :
    (diagonalObjPullbackFstIso f g).inv ≫ pullback.fst _ _ ≫ pullback.snd _ _ =
      pullback.fst _ _ ≫ pullback.fst _ _ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.diagonalObjPul …
  -/
  delta diagonalObjPullbackFstIso
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.pullbackRight …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem diagonalObjPullbackFstIso_inv_snd_fst {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) :
    (diagonalObjPullbackFstIso f g).inv ≫ pullback.snd _ _ ≫ pullback.fst _ _ =
      pullback.snd _ _ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.diagonalObjPul …
  -/
  delta diagonalObjPullbackFstIso
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.pullbackRight …
  -/
  simp
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem diagonalObjPullbackFstIso_inv_snd_snd {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) :
    (diagonalObjPullbackFstIso f g).inv ≫ pullback.snd _ _ ≫ pullback.snd _ _ =
      pullback.fst _ _ ≫ pullback.snd _ _ := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.diagonalObjPul …
  -/
  delta diagonalObjPullbackFstIso
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.pullbackRight …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem diagonal_pullback_fst {X Y Z : C} (f : X ⟶ Z) (g : Y ⟶ Z) :
    diagonal (pullback.fst f g) =
      (pullbackSymmetry _ _).hom ≫
        ((Over.pullback f).map
               /-
                 C : Type u_1
                 inst✝⁵ : CategoryTheory.Category.{?u.915077, u_1} C
                 X✝ Y✝ Z✝ : C
                 inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
                 S T : C
                 f✝ : Quiver.Hom X✝ T
                 g✝ : Quiver.Hom Y✝ T
                 i : Quiver.Hom T S
                 inst✝³ : CategoryTheory.Limits.HasPullback i i
                 inst✝² : CategoryTheory.Limits.HasPullback f✝ g✝
                 inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
                 inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
                 X Y Z : C
                 f : Quiver.Hom X Z
                 g : Quiver.Hom Y Z
                 ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.diago …
               -/
              (Over.homMk (diagonal g) : Over.mk g ⟶ Over.mk (pullback.snd _ _ ≫ g))).left ≫
               /-
                 🎉 no goals
               -/
          (diagonalObjPullbackFstIso f g).inv := by
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    X Y Z : C
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ Eq (CategoryTheory.Limits.pullback.diagonal (CategoryTheory.Limits.pullback. …
  -/
                    /-
                      🎉 no goals
                    -/
                    /-
                      🎉 no goals
                    -/
                    /-
                      🎉 no goals
                    -/
  ext <;> dsimp <;> simp
                    /-
                      🎉 no goals
                    -/


/-- Informally, this is a special case of `pullback_map_diagonal_isPullback` for `T = X`. -/
lemma pullback_lift_diagonal_isPullback (g : Y ⟶ X) (f : X ⟶ S) :
                                            /-
                                              C : Type u_1
                                              inst✝⁵ : CategoryTheory.Category.{?u.930113, u_1} C
                                              X Y Z : C
                                              inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
                                              S T : C
                                              f✝ : Quiver.Hom X T
                                              g✝ : Quiver.Hom Y T
                                              i : Quiver.Hom T S
                                              inst✝³ : CategoryTheory.Limits.HasPullback i i
                                              inst✝² : CategoryTheory.Limits.HasPullback f✝ g✝
                                              inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
                                              inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
                                              g : Quiver.Hom Y X
                                              f : Quiver.Hom X S
                                              ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.id Y)  …
                                            -/
    IsPullback g (pullback.lift (𝟙 Y) g (by simp)) (diagonal f)
                                            /-
                                              🎉 no goals
                                            -/
                                                    /-
                                                      C : Type u_1
                                                      inst✝⁵ : CategoryTheory.Category.{?u.930113, u_1} C
                                                      X Y Z : C
                                                      inst✝⁴ : CategoryTheory.Limits.HasPullbacks C
                                                      S T : C
                                                      f✝ : Quiver.Hom X T
                                                      g✝ : Quiver.Hom Y T
                                                      i : Quiver.Hom T S
                                                      inst✝³ : CategoryTheory.Limits.HasPullback i i
                                                      inst✝² : CategoryTheory.Limits.HasPullback f✝ g✝
                                                      inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
                                                      inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.Limits.pullback.diag …
                                                      g : Quiver.Hom Y X
                                                      f : Quiver.Hom X S
                                                      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp g …
                                                    -/
                                                    /-
                                                      🎉 no goals
                                                    -/
      (pullback.map (g ≫ f) f f f g (𝟙 X) (𝟙 S) (by simp) (by simp)) := by
                                                              /-
                                                                🎉 no goals
                                                              -/
  /-
    C : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
    X Y : C
    inst✝ : CategoryTheory.Limits.HasPullbacks C
    S : C
    g : Quiver.Hom Y X
    f : Quiver.Hom X S
    ⊢ CategoryTheory.IsPullback g (CategoryTheory.Limits.pullback.lift (CategoryTh …
  -/
  let i : pullback (g ≫ f) f ≅ pullback (g ≫ f) (𝟙 X ≫ f) := congrHom rfl (by simp)
  let e : pullback (diagonal f) (map (g ≫ f) f f f g (𝟙 X) (𝟙 S) (by simp) (by simp)) ≅
      pullback (diagonal f) (map (g ≫ f) (𝟙 X ≫ f) f f g (𝟙 X) (𝟙 S) (by simp) (by simp)) :=
    (asIso (map _ _ _ _ (𝟙 _) i.inv (𝟙 _) (by simp) (by ext <;> simp [i]))).symm
  apply IsPullback.of_iso_pullback _
      (e ≪≫ pullbackDiagonalMapIdIso (T := X) (S := S) g (𝟙 X) f ≪≫ asIso (pullback.fst _ _)).symm
    /-
      case w₁
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      S : C
      g : Quiver.Hom Y X
      f : Quiver.Hom X S
      i : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Categor …
      e : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Limits. …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.trans ((CategoryTheory.Limits.pull …
    -/
  · simp [e]
    /-
      🎉 no goals
    -/
    /-
      case w₂
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      S : C
      g : Quiver.Hom Y X
      f : Quiver.Hom X S
      i : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Categor …
      e : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Limits. …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (e.trans ((CategoryTheory.Limits.pull …
    -/
            /-
              🎉 no goals
            -/
  · ext <;> simp [e, i]
            /-
              🎉 no goals
            -/
    /-
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      S : C
      g : Quiver.Hom Y X
      f : Quiver.Hom X S
      i : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Categor …
      e : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Limits. …
      ⊢ CategoryTheory.CommSq g (CategoryTheory.Limits.pullback.lift (CategoryTheory …
    -/
  · constructor
    /-
      case w
      C : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} C
      X Y : C
      inst✝ : CategoryTheory.Limits.HasPullbacks C
      S : C
      g : Quiver.Hom Y X
      f : Quiver.Hom X S
      i : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Categor …
      e : CategoryTheory.Iso (CategoryTheory.Limits.pullback (CategoryTheory.Limits. …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.Limits.pullback.dia …
    -/
            /-
              🎉 no goals
            -/
    ext <;> simp [condition]
            /-
              🎉 no goals
            -/


/-- Given the following diagram with `S ⟶ S'` a monomorphism,

```
    X ⟶ X'
      ↘      ↘
        S ⟶ S'
      ↗      ↗
    Y ⟶ Y'
```

This iso witnesses the fact that

```
      X ×[S] Y ⟶ (X' ×[S'] Y') ×[Y'] Y
          |                  |
          |                  |
          ↓                  ↓
(X' ×[S'] Y') ×[X'] X ⟶ X' ×[S'] Y'
```

is a pullback square. The diagonal map of this square is `pullback.map`.
Also see `pullback_lift_map_is_pullback`.
-/
@[simps]
def pullbackFstFstIso {X Y S X' Y' S' : C} (f : X ⟶ S) (g : Y ⟶ S) (f' : X' ⟶ S') (g' : Y' ⟶ S')
    (i₁ : X ⟶ X') (i₂ : Y ⟶ Y') (i₃ : S ⟶ S') (e₁ : f ≫ i₃ = i₁ ≫ f') (e₂ : g ≫ i₃ = i₂ ≫ g')
    [Mono i₃] :
    pullback (pullback.fst _ _ : pullback (pullback.fst _ _ : pullback f' g' ⟶ _) i₁ ⟶ _)
        (pullback.fst _ _ : pullback (pullback.snd _ _ : pullback f' g' ⟶ _) i₂ ⟶ _) ≅
      pullback f g where
  hom :=
    pullback.lift (pullback.fst _ _ ≫ pullback.snd _ _) (pullback.snd _ _ ≫ pullback.snd _ _)
      (by
        rw [← cancel_mono i₃, Category.assoc, Category.assoc, Category.assoc, Category.assoc, e₁,
          e₂, ← pullback.condition_assoc, pullback.condition_assoc, pullback.condition,
          pullback.condition_assoc])
  inv :=
    pullback.lift
      (pullback.lift (pullback.map _ _ _ _ _ _ _ e₁ e₂) (pullback.fst _ _) (pullback.lift_fst ..))
      (pullback.lift (pullback.map _ _ _ _ _ _ _ e₁ e₂) (pullback.snd _ _) (pullback.lift_snd ..))
          /-
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.965259, u_1} C
            X✝ Y✝ Z : C
            inst✝¹ : CategoryTheory.Limits.HasPullbacks C
            X Y S X' Y' S' : C
            f : Quiver.Hom X S
            g : Quiver.Hom Y S
            f' : Quiver.Hom X' S'
            g' : Quiver.Hom Y' S'
            i₁ : Quiver.Hom X X'
            i₂ : Quiver.Hom Y Y'
            i₃ : Quiver.Hom S S'
            e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
            e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
            inst✝ : CategoryTheory.Mono i₃
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
          -/
      (by rw [pullback.lift_fst, pullback.lift_fst])
          /-
            🎉 no goals
          -/
  hom_inv_id := by
    -- We could use `ext` here to immediately descend to the leaf goals,
    -- but it only obscures the structure.
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.965259, u_1} C
      X✝ Y✝ Z : C
      inst✝¹ : CategoryTheory.Limits.HasPullbacks C
      X Y S X' Y' S' : C
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      f' : Quiver.Hom X' S'
      g' : Quiver.Hom Y' S'
      i₁ : Quiver.Hom X X'
      i₂ : Quiver.Hom Y Y'
      i₃ : Quiver.Hom S S'
      e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
      inst✝ : CategoryTheory.Mono i₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
    -/
    apply pullback.hom_ext
      /-
        case h₀
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.965259, u_1} C
        X✝ Y✝ Z : C
        inst✝¹ : CategoryTheory.Limits.HasPullbacks C
        X Y S X' Y' S' : C
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        f' : Quiver.Hom X' S'
        g' : Quiver.Hom Y' S'
        i₁ : Quiver.Hom X X'
        i₂ : Quiver.Hom Y Y'
        i₃ : Quiver.Hom S S'
        e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
        e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
        inst✝ : CategoryTheory.Mono i₃
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · apply pullback.hom_ext
        /-
          case h₀.h₀
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.965259, u_1} C
          X✝ Y✝ Z : C
          inst✝¹ : CategoryTheory.Limits.HasPullbacks C
          X Y S X' Y' S' : C
          f : Quiver.Hom X S
          g : Quiver.Hom Y S
          f' : Quiver.Hom X' S'
          g' : Quiver.Hom Y' S'
          i₁ : Quiver.Hom X X'
          i₂ : Quiver.Hom Y Y'
          i₃ : Quiver.Hom S S'
          e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
          e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
          inst✝ : CategoryTheory.Mono i₃
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
      · apply pullback.hom_ext
          /-
            case h₀.h₀.h₀
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.965259, u_1} C
            X✝ Y✝ Z : C
            inst✝¹ : CategoryTheory.Limits.HasPullbacks C
            X Y S X' Y' S' : C
            f : Quiver.Hom X S
            g : Quiver.Hom Y S
            f' : Quiver.Hom X' S'
            g' : Quiver.Hom Y' S'
            i₁ : Quiver.Hom X X'
            i₂ : Quiver.Hom Y Y'
            i₃ : Quiver.Hom S S'
            e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
            e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
            inst✝ : CategoryTheory.Mono i₃
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
        · simp only [Category.assoc, lift_fst, lift_fst_assoc, Category.id_comp]
          /-
            case h₀.h₀.h₀
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.965259, u_1} C
            X✝ Y✝ Z : C
            inst✝¹ : CategoryTheory.Limits.HasPullbacks C
            X Y S X' Y' S' : C
            f : Quiver.Hom X S
            g : Quiver.Hom Y S
            f' : Quiver.Hom X' S'
            g' : Quiver.Hom Y' S'
            i₁ : Quiver.Hom X X'
            i₂ : Quiver.Hom Y Y'
            i₃ : Quiver.Hom S S'
            e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
            e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
            inst✝ : CategoryTheory.Mono i₃
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
          -/
          rw [condition]
          /-
            🎉 no goals
          -/
          /-
            case h₀.h₀.h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.965259, u_1} C
            X✝ Y✝ Z : C
            inst✝¹ : CategoryTheory.Limits.HasPullbacks C
            X Y S X' Y' S' : C
            f : Quiver.Hom X S
            g : Quiver.Hom Y S
            f' : Quiver.Hom X' S'
            g' : Quiver.Hom Y' S'
            i₁ : Quiver.Hom X X'
            i₂ : Quiver.Hom Y Y'
            i₃ : Quiver.Hom S S'
            e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
            e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
            inst✝ : CategoryTheory.Mono i₃
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
        · simp [Category.assoc, lift_snd, condition_assoc, condition]
          /-
            🎉 no goals
          -/
        /-
          case h₀.h₁
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.965259, u_1} C
          X✝ Y✝ Z : C
          inst✝¹ : CategoryTheory.Limits.HasPullbacks C
          X Y S X' Y' S' : C
          f : Quiver.Hom X S
          g : Quiver.Hom Y S
          f' : Quiver.Hom X' S'
          g' : Quiver.Hom Y' S'
          i₁ : Quiver.Hom X X'
          i₂ : Quiver.Hom Y Y'
          i₃ : Quiver.Hom S S'
          e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
          e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
          inst✝ : CategoryTheory.Mono i₃
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
      · simp only [Category.assoc, lift_fst_assoc, lift_snd, lift_fst, Category.id_comp]
        /-
          🎉 no goals
        -/
      /-
        case h₁
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.965259, u_1} C
        X✝ Y✝ Z : C
        inst✝¹ : CategoryTheory.Limits.HasPullbacks C
        X Y S X' Y' S' : C
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        f' : Quiver.Hom X' S'
        g' : Quiver.Hom Y' S'
        i₁ : Quiver.Hom X X'
        i₂ : Quiver.Hom Y Y'
        i₃ : Quiver.Hom S S'
        e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
        e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
        inst✝ : CategoryTheory.Mono i₃
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · apply pullback.hom_ext
        /-
          case h₁.h₀
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.965259, u_1} C
          X✝ Y✝ Z : C
          inst✝¹ : CategoryTheory.Limits.HasPullbacks C
          X Y S X' Y' S' : C
          f : Quiver.Hom X S
          g : Quiver.Hom Y S
          f' : Quiver.Hom X' S'
          g' : Quiver.Hom Y' S'
          i₁ : Quiver.Hom X X'
          i₂ : Quiver.Hom Y Y'
          i₃ : Quiver.Hom S S'
          e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
          e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
          inst✝ : CategoryTheory.Mono i₃
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
      · apply pullback.hom_ext
          /-
            case h₁.h₀.h₀
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.965259, u_1} C
            X✝ Y✝ Z : C
            inst✝¹ : CategoryTheory.Limits.HasPullbacks C
            X Y S X' Y' S' : C
            f : Quiver.Hom X S
            g : Quiver.Hom Y S
            f' : Quiver.Hom X' S'
            g' : Quiver.Hom Y' S'
            i₁ : Quiver.Hom X X'
            i₂ : Quiver.Hom Y Y'
            i₃ : Quiver.Hom S S'
            e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
            e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
            inst✝ : CategoryTheory.Mono i₃
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
        · simp only [Category.assoc, lift_snd_assoc, lift_fst_assoc, lift_fst, Category.id_comp]
          /-
            case h₁.h₀.h₀
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.965259, u_1} C
            X✝ Y✝ Z : C
            inst✝¹ : CategoryTheory.Limits.HasPullbacks C
            X Y S X' Y' S' : C
            f : Quiver.Hom X S
            g : Quiver.Hom Y S
            f' : Quiver.Hom X' S'
            g' : Quiver.Hom Y' S'
            i₁ : Quiver.Hom X X'
            i₂ : Quiver.Hom Y Y'
            i₃ : Quiver.Hom S S'
            e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
            e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
            inst✝ : CategoryTheory.Mono i₃
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
          -/
          rw [← condition_assoc, condition]
          /-
            🎉 no goals
          -/
          /-
            case h₁.h₀.h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.965259, u_1} C
            X✝ Y✝ Z : C
            inst✝¹ : CategoryTheory.Limits.HasPullbacks C
            X Y S X' Y' S' : C
            f : Quiver.Hom X S
            g : Quiver.Hom Y S
            f' : Quiver.Hom X' S'
            g' : Quiver.Hom Y' S'
            i₁ : Quiver.Hom X X'
            i₂ : Quiver.Hom Y Y'
            i₃ : Quiver.Hom S S'
            e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
            e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
            inst✝ : CategoryTheory.Mono i₃
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
        · simp only [Category.assoc, lift_snd, lift_fst_assoc, lift_snd_assoc, Category.id_comp]
          /-
            case h₁.h₀.h₁
            C : Type u_1
            inst✝² : CategoryTheory.Category.{?u.965259, u_1} C
            X✝ Y✝ Z : C
            inst✝¹ : CategoryTheory.Limits.HasPullbacks C
            X Y S X' Y' S' : C
            f : Quiver.Hom X S
            g : Quiver.Hom Y S
            f' : Quiver.Hom X' S'
            g' : Quiver.Hom Y' S'
            i₁ : Quiver.Hom X X'
            i₂ : Quiver.Hom Y Y'
            i₃ : Quiver.Hom S S'
            e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
            e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
            inst✝ : CategoryTheory.Mono i₃
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd ( …
          -/
          rw [condition]
          /-
            🎉 no goals
          -/
        /-
          case h₁.h₁
          C : Type u_1
          inst✝² : CategoryTheory.Category.{?u.965259, u_1} C
          X✝ Y✝ Z : C
          inst✝¹ : CategoryTheory.Limits.HasPullbacks C
          X Y S X' Y' S' : C
          f : Quiver.Hom X S
          g : Quiver.Hom Y S
          f' : Quiver.Hom X' S'
          g' : Quiver.Hom Y' S'
          i₁ : Quiver.Hom X X'
          i₂ : Quiver.Hom Y Y'
          i₃ : Quiver.Hom S S'
          e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
          e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
          inst✝ : CategoryTheory.Mono i₃
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
      · simp only [Category.assoc, lift_snd_assoc, lift_snd, Category.id_comp]
        /-
          🎉 no goals
        -/
  inv_hom_id := by
    /-
      C : Type u_1
      inst✝² : CategoryTheory.Category.{?u.965259, u_1} C
      X✝ Y✝ Z : C
      inst✝¹ : CategoryTheory.Limits.HasPullbacks C
      X Y S X' Y' S' : C
      f : Quiver.Hom X S
      g : Quiver.Hom Y S
      f' : Quiver.Hom X' S'
      g' : Quiver.Hom Y' S'
      i₁ : Quiver.Hom X X'
      i₂ : Quiver.Hom Y Y'
      i₃ : Quiver.Hom S S'
      e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
      e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
      inst✝ : CategoryTheory.Mono i₃
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
    -/
    apply pullback.hom_ext
      /-
        case h₀
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.965259, u_1} C
        X✝ Y✝ Z : C
        inst✝¹ : CategoryTheory.Limits.HasPullbacks C
        X Y S X' Y' S' : C
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        f' : Quiver.Hom X' S'
        g' : Quiver.Hom Y' S'
        i₁ : Quiver.Hom X X'
        i₂ : Quiver.Hom Y Y'
        i₃ : Quiver.Hom S S'
        e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
        e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
        inst✝ : CategoryTheory.Mono i₃
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp only [Category.assoc, lift_fst, lift_fst_assoc, lift_snd, Category.id_comp]
      /-
        🎉 no goals
      -/
      /-
        case h₁
        C : Type u_1
        inst✝² : CategoryTheory.Category.{?u.965259, u_1} C
        X✝ Y✝ Z : C
        inst✝¹ : CategoryTheory.Limits.HasPullbacks C
        X Y S X' Y' S' : C
        f : Quiver.Hom X S
        g : Quiver.Hom Y S
        f' : Quiver.Hom X' S'
        g' : Quiver.Hom Y' S'
        i₁ : Quiver.Hom X X'
        i₂ : Quiver.Hom Y Y'
        i₃ : Quiver.Hom S S'
        e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
        e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
        inst✝ : CategoryTheory.Mono i₃
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp only [Category.assoc, lift_snd, lift_snd_assoc, Category.id_comp]
      /-
        🎉 no goals
      -/


theorem pullback_map_eq_pullbackFstFstIso_inv {X Y S X' Y' S' : C} (f : X ⟶ S) (g : Y ⟶ S)
    (f' : X' ⟶ S') (g' : Y' ⟶ S') (i₁ : X ⟶ X') (i₂ : Y ⟶ Y') (i₃ : S ⟶ S')
    (e₁ : f ≫ i₃ = i₁ ≫ f') (e₂ : g ≫ i₃ = i₂ ≫ g') [Mono i₃] :
    pullback.map f g f' g' i₁ i₂ i₃ e₁ e₂ =
      (pullbackFstFstIso f g f' g' i₁ i₂ i₃ e₁ e₂).inv ≫ pullback.snd _ _ ≫ pullback.fst _ _ := by
  /-
    C : Type u_1
    inst✝² : CategoryTheory.Category.{u_2, u_1} C
    inst✝¹ : CategoryTheory.Limits.HasPullbacks C
    X Y S X' Y' S' : C
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    f' : Quiver.Hom X' S'
    g' : Quiver.Hom Y' S'
    i₁ : Quiver.Hom X X'
    i₂ : Quiver.Hom Y Y'
    i₃ : Quiver.Hom S S'
    e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
    e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
    inst✝ : CategoryTheory.Mono i₃
    ⊢ Eq (CategoryTheory.Limits.pullback.map f g f' g' i₁ i₂ i₃ e₁ e₂) (CategoryTh …
  -/
  simp only [pullbackFstFstIso_inv, lift_snd_assoc, lift_fst]
  /-
    🎉 no goals
  -/


theorem pullback_lift_map_isPullback {X Y S X' Y' S' : C} (f : X ⟶ S) (g : Y ⟶ S) (f' : X' ⟶ S')
    (g' : Y' ⟶ S') (i₁ : X ⟶ X') (i₂ : Y ⟶ Y') (i₃ : S ⟶ S') (e₁ : f ≫ i₃ = i₁ ≫ f')
    (e₂ : g ≫ i₃ = i₂ ≫ g') [Mono i₃] :
    IsPullback (pullback.lift (pullback.map f g f' g' i₁ i₂ i₃ e₁ e₂) (fst _ _) (lift_fst _ _ _))
      (pullback.lift (pullback.map f g f' g' i₁ i₂ i₃ e₁ e₂) (snd _ _) (lift_snd _ _ _))
      (pullback.fst _ _) (pullback.fst _ _) :=
                                 /-
                                   C : Type u_1
                                   inst✝² : CategoryTheory.Category.{u_2, u_1} C
                                   inst✝¹ : CategoryTheory.Limits.HasPullbacks C
                                   X Y S X' Y' S' : C
                                   f : Quiver.Hom X S
                                   g : Quiver.Hom Y S
                                   f' : Quiver.Hom X' S'
                                   g' : Quiver.Hom Y' S'
                                   i₁ : Quiver.Hom X X'
                                   i₂ : Quiver.Hom Y Y'
                                   i₃ : Quiver.Hom S S'
                                   e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
                                   e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
                                   inst✝ : CategoryTheory.Mono i₃
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
                                 -/
  IsPullback.of_iso_pullback ⟨by rw [lift_fst, lift_fst]⟩
                                 /-
                                   🎉 no goals
                                 -/
                                                          /-
                                                            C : Type u_1
                                                            inst✝² : CategoryTheory.Category.{u_2, u_1} C
                                                            inst✝¹ : CategoryTheory.Limits.HasPullbacks C
                                                            X Y S X' Y' S' : C
                                                            f : Quiver.Hom X S
                                                            g : Quiver.Hom Y S
                                                            f' : Quiver.Hom X' S'
                                                            g' : Quiver.Hom Y' S'
                                                            i₁ : Quiver.Hom X X'
                                                            i₂ : Quiver.Hom Y Y'
                                                            i₃ : Quiver.Hom S S'
                                                            e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
                                                            e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
                                                            inst✝ : CategoryTheory.Mono i₃
                                                            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackFstFst …
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
    (pullbackFstFstIso f g f' g' i₁ i₂ i₃ e₁ e₂).symm (by simp) (by simp)
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


