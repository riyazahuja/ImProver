/-- The intersection of `Uᵢ ×[Z] Y` and `Uⱼ ×[Z] Y` is given by (Uᵢ ×[Z] Y) ×[X] Uⱼ -/
def v (i j : 𝒰.J) : Scheme :=
  pullback ((pullback.fst (𝒰.map i ≫ f) g) ≫ 𝒰.map i) (𝒰.map j)


/-- The canonical transition map `(Uᵢ ×[Z] Y) ×[X] Uⱼ ⟶ (Uⱼ ×[Z] Y) ×[X] Uᵢ` given by the fact
that pullbacks are associative and symmetric. -/
def t (i j : 𝒰.J) : v 𝒰 f g i j ⟶ v 𝒰 f g j i := by
  have : HasPullback (pullback.snd _ _ ≫ 𝒰.map i ≫ f) g :=
    hasPullback_assoc_symm (𝒰.map j) (𝒰.map i) (𝒰.map i ≫ f) g
  have : HasPullback (pullback.snd _ _ ≫ 𝒰.map j ≫ f) g :=
    hasPullback_assoc_symm (𝒰.map i) (𝒰.map j) (𝒰.map j ≫ f) g
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i j : 𝒰.J
    this✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp  …
    this : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp ( …
    ⊢ Quiver.Hom (AlgebraicGeometry.Scheme.Pullback.v 𝒰 f g i j) (AlgebraicGeometr …
  -/
  refine (pullbackSymmetry ..).hom ≫ (pullbackAssoc ..).inv ≫ ?_
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i j : 𝒰.J
    this✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp  …
    this : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp ( …
    ⊢ Quiver.Hom (CategoryTheory.Limits.pullback (CategoryTheory.CategoryStruct.co …
  -/
  refine ?_ ≫ (pullbackAssoc ..).hom ≫ (pullbackSymmetry ..).hom
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i j : 𝒰.J
    this✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp  …
    this : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp ( …
    ⊢ Quiver.Hom (CategoryTheory.Limits.pullback (CategoryTheory.CategoryStruct.co …
  -/
  refine pullback.map _ _ _ _ (pullbackSymmetry _ _).hom (𝟙 _) (𝟙 _) ?_ ?_
    /-
      case refine_1
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      i j : 𝒰.J
      this✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp  …
      this : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp ( …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · rw [pullbackSymmetry_hom_comp_snd_assoc, pullback.condition_assoc, Category.comp_id]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      i j : 𝒰.J
      this✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp  …
      this : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp ( …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.CategoryStruct.id Z …
    -/
  · rw [Category.comp_id, Category.id_comp]
    /-
      🎉 no goals
    -/


@[simp, reassoc]
theorem t_fst_fst (i j : 𝒰.J) : t 𝒰 f g i j ≫ pullback.fst _ _ ≫ pullback.fst _ _ =
    pullback.snd _ _ := by
  simp only [t, Category.assoc, pullbackSymmetry_hom_comp_fst_assoc, pullbackAssoc_hom_snd_fst,
    pullback.lift_fst_assoc, pullbackSymmetry_hom_comp_snd, pullbackAssoc_inv_fst_fst,
    pullbackSymmetry_hom_comp_fst]


@[simp, reassoc]
theorem t_fst_snd (i j : 𝒰.J) :
    t 𝒰 f g i j ≫ pullback.fst _ _ ≫ pullback.snd _ _ = pullback.fst _ _ ≫ pullback.snd _ _ := by
  simp only [t, Category.assoc, pullbackSymmetry_hom_comp_fst_assoc, pullbackAssoc_hom_snd_snd,
    pullback.lift_snd, Category.comp_id, pullbackAssoc_inv_snd, pullbackSymmetry_hom_comp_snd_assoc]


@[simp, reassoc]
theorem t_snd (i j : 𝒰.J) : t 𝒰 f g i j ≫ pullback.snd _ _ =
    pullback.fst _ _ ≫ pullback.fst _ _ := by
  simp only [t, Category.assoc, pullbackSymmetry_hom_comp_snd, pullbackAssoc_hom_fst,
    pullback.lift_fst_assoc, pullbackSymmetry_hom_comp_fst, pullbackAssoc_inv_fst_snd,
    pullbackSymmetry_hom_comp_snd_assoc]


theorem t_id (i : 𝒰.J) : t 𝒰 f g i i = 𝟙 _ := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i : 𝒰.J
    ⊢ Eq (AlgebraicGeometry.Scheme.Pullback.t 𝒰 f g i i) (CategoryTheory.CategoryS …
  -/
  apply pullback.hom_ext <;> rw [Category.id_comp]
    /-
      case h₀
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      i : 𝒰.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.t  …
    -/
  · apply pullback.hom_ext
      /-
        case h₀.h₀
        X Y Z : AlgebraicGeometry.Scheme
        𝒰 : X.OpenCover
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
        i : 𝒰.J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · rw [← cancel_mono (𝒰.map i)]; simp only [pullback.condition, Category.assoc, t_fst_fst]
                                    /-
                                      🎉 no goals
                                    -/
      /-
        case h₀.h₁
        X Y Z : AlgebraicGeometry.Scheme
        𝒰 : X.OpenCover
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
        i : 𝒰.J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp only [Category.assoc, t_fst_snd]
      /-
        🎉 no goals
      -/
    /-
      case h₁
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      i : 𝒰.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.t  …
    -/
  · rw [← cancel_mono (𝒰.map i)]; simp only [pullback.condition, t_snd, Category.assoc]
                                  /-
                                    🎉 no goals
                                  -/


/-- The inclusion map of `V i j = (Uᵢ ×[Z] Y) ×[X] Uⱼ ⟶ Uᵢ ×[Z] Y`-/
abbrev fV (i j : 𝒰.J) : v 𝒰 f g i j ⟶ pullback (𝒰.map i ≫ f) g :=
  pullback.fst _ _


/-- The map `((Xᵢ ×[Z] Y) ×[X] Xⱼ) ×[Xᵢ ×[Z] Y] ((Xᵢ ×[Z] Y) ×[X] Xₖ)` ⟶
  `((Xⱼ ×[Z] Y) ×[X] Xₖ) ×[Xⱼ ×[Z] Y] ((Xⱼ ×[Z] Y) ×[X] Xᵢ)` needed for gluing   -/
def t' (i j k : 𝒰.J) :
    pullback (fV 𝒰 f g i j) (fV 𝒰 f g i k) ⟶ pullback (fV 𝒰 f g j k) (fV 𝒰 f g j i) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i j k : 𝒰.J
    ⊢ Quiver.Hom (CategoryTheory.Limits.pullback (AlgebraicGeometry.Scheme.Pullbac …
  -/
  refine (pullbackRightPullbackFstIso ..).hom ≫ ?_
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i j k : 𝒰.J
    ⊢ Quiver.Hom (CategoryTheory.Limits.pullback (CategoryTheory.CategoryStruct.co …
  -/
  refine ?_ ≫ (pullbackSymmetry _ _).hom
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i j k : 𝒰.J
    ⊢ Quiver.Hom (CategoryTheory.Limits.pullback (CategoryTheory.CategoryStruct.co …
  -/
  refine ?_ ≫ (pullbackRightPullbackFstIso ..).inv
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i j k : 𝒰.J
    ⊢ Quiver.Hom (CategoryTheory.Limits.pullback (CategoryTheory.CategoryStruct.co …
  -/
  refine pullback.map _ _ _ _ (t 𝒰 f g i j) (𝟙 _) (𝟙 _) ?_ ?_
    /-
      case refine_1
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      i j k : 𝒰.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · simp_rw [Category.comp_id, t_fst_fst_assoc, ← pullback.condition]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      i j k : 𝒰.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (𝒰.map k) (CategoryTheory.CategoryStr …
    -/
  · rw [Category.comp_id, Category.id_comp]
    /-
      🎉 no goals
    -/


@[simp, reassoc]
theorem t'_fst_fst_fst (i j k : 𝒰.J) :
    t' 𝒰 f g i j k ≫ pullback.fst _ _ ≫ pullback.fst _ _ ≫ pullback.fst _ _ =
      pullback.fst _ _ ≫ pullback.snd _ _ := by
  simp only [t', Category.assoc, pullbackSymmetry_hom_comp_fst_assoc,
    pullbackRightPullbackFstIso_inv_snd_fst_assoc, pullback.lift_fst_assoc, t_fst_fst,
    pullbackRightPullbackFstIso_hom_fst_assoc]


@[simp, reassoc]
theorem t'_fst_fst_snd (i j k : 𝒰.J) :
    t' 𝒰 f g i j k ≫ pullback.fst _ _ ≫ pullback.fst _ _ ≫ pullback.snd _ _ =
      pullback.fst _ _ ≫ pullback.fst _ _ ≫ pullback.snd _ _ := by
  simp only [t', Category.assoc, pullbackSymmetry_hom_comp_fst_assoc,
    pullbackRightPullbackFstIso_inv_snd_fst_assoc, pullback.lift_fst_assoc, t_fst_snd,
    pullbackRightPullbackFstIso_hom_fst_assoc]


@[simp, reassoc]
theorem t'_fst_snd (i j k : 𝒰.J) :
    t' 𝒰 f g i j k ≫ pullback.fst _ _ ≫ pullback.snd _ _ =
      pullback.snd _ _ ≫ pullback.snd _ _ := by
  simp only [t', Category.assoc, pullbackSymmetry_hom_comp_fst_assoc,
    pullbackRightPullbackFstIso_inv_snd_snd, pullback.lift_snd, Category.comp_id,
    pullbackRightPullbackFstIso_hom_snd]


@[simp, reassoc]
theorem t'_snd_fst_fst (i j k : 𝒰.J) :
    t' 𝒰 f g i j k ≫ pullback.snd _ _ ≫ pullback.fst _ _ ≫ pullback.fst _ _ =
      pullback.fst _ _ ≫ pullback.snd _ _ := by
  simp only [t', Category.assoc, pullbackSymmetry_hom_comp_snd_assoc,
    pullbackRightPullbackFstIso_inv_fst_assoc, pullback.lift_fst_assoc, t_fst_fst,
    pullbackRightPullbackFstIso_hom_fst_assoc]


@[simp, reassoc]
theorem t'_snd_fst_snd (i j k : 𝒰.J) :
    t' 𝒰 f g i j k ≫ pullback.snd _ _ ≫ pullback.fst _ _ ≫ pullback.snd _ _ =
      pullback.fst _ _ ≫ pullback.fst _ _ ≫ pullback.snd _ _ := by
  simp only [t', Category.assoc, pullbackSymmetry_hom_comp_snd_assoc,
    pullbackRightPullbackFstIso_inv_fst_assoc, pullback.lift_fst_assoc, t_fst_snd,
    pullbackRightPullbackFstIso_hom_fst_assoc]


@[simp, reassoc]
theorem t'_snd_snd (i j k : 𝒰.J) :
    t' 𝒰 f g i j k ≫ pullback.snd _ _ ≫ pullback.snd _ _ =
      pullback.fst _ _ ≫ pullback.fst _ _ ≫ pullback.fst _ _ := by
  simp only [t', Category.assoc, pullbackSymmetry_hom_comp_snd_assoc,
    pullbackRightPullbackFstIso_inv_fst_assoc, pullback.lift_fst_assoc, t_snd,
    pullbackRightPullbackFstIso_hom_fst_assoc]


theorem cocycle_fst_fst_fst (i j k : 𝒰.J) :
    t' 𝒰 f g i j k ≫ t' 𝒰 f g j k i ≫ t' 𝒰 f g k i j ≫ pullback.fst _ _ ≫ pullback.fst _ _ ≫
      pullback.fst _ _ = pullback.fst _ _ ≫ pullback.fst _ _ ≫ pullback.fst _ _ := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i j k : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.t' …
  -/
  simp only [t'_fst_fst_fst, t'_fst_snd, t'_snd_snd]
  /-
    🎉 no goals
  -/


theorem cocycle_fst_fst_snd (i j k : 𝒰.J) :
    t' 𝒰 f g i j k ≫ t' 𝒰 f g j k i ≫ t' 𝒰 f g k i j ≫ pullback.fst _ _ ≫ pullback.fst _ _ ≫
      pullback.snd _ _ = pullback.fst _ _ ≫ pullback.fst _ _ ≫ pullback.snd _ _ := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i j k : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.t' …
  -/
  simp only [t'_fst_fst_snd]
  /-
    🎉 no goals
  -/


theorem cocycle_fst_snd (i j k : 𝒰.J) :
    t' 𝒰 f g i j k ≫ t' 𝒰 f g j k i ≫ t' 𝒰 f g k i j ≫ pullback.fst _ _ ≫ pullback.snd _ _ =
      pullback.fst _ _ ≫ pullback.snd _ _ := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i j k : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.t' …
  -/
  simp only [t'_fst_snd, t'_snd_snd, t'_fst_fst_fst]
  /-
    🎉 no goals
  -/


theorem cocycle_snd_fst_fst (i j k : 𝒰.J) :
    t' 𝒰 f g i j k ≫ t' 𝒰 f g j k i ≫ t' 𝒰 f g k i j ≫ pullback.snd _ _ ≫ pullback.fst _ _ ≫
      pullback.fst _ _ = pullback.snd _ _ ≫ pullback.fst _ _ ≫ pullback.fst _ _ := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i j k : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.t' …
  -/
  rw [← cancel_mono (𝒰.map i)]
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i j k : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp only [pullback.condition_assoc, t'_snd_fst_fst, t'_fst_snd, t'_snd_snd]
  /-
    🎉 no goals
  -/


theorem cocycle_snd_fst_snd (i j k : 𝒰.J) :
    t' 𝒰 f g i j k ≫ t' 𝒰 f g j k i ≫ t' 𝒰 f g k i j ≫ pullback.snd _ _ ≫ pullback.fst _ _ ≫
      pullback.snd _ _ = pullback.snd _ _ ≫ pullback.fst _ _ ≫ pullback.snd _ _ := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i j k : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.t' …
  -/
  simp only [pullback.condition_assoc, t'_snd_fst_snd]
  /-
    🎉 no goals
  -/


theorem cocycle_snd_snd (i j k : 𝒰.J) :
    t' 𝒰 f g i j k ≫ t' 𝒰 f g j k i ≫ t' 𝒰 f g k i j ≫ pullback.snd _ _ ≫ pullback.snd _ _ =
      pullback.snd _ _ ≫ pullback.snd _ _ := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i j k : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.t' …
  -/
  simp only [t'_snd_snd, t'_fst_fst_fst, t'_fst_snd]
  /-
    🎉 no goals
  -/

-- `by tidy` should solve it, but it times out.

theorem cocycle (i j k : 𝒰.J) : t' 𝒰 f g i j k ≫ t' 𝒰 f g j k i ≫ t' 𝒰 f g k i j = 𝟙 _ := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i j k : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.t' …
  -/
  apply pullback.hom_ext <;> rw [Category.id_comp]
    /-
      case h₀
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      i j k : 𝒰.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · apply pullback.hom_ext
      /-
        case h₀.h₀
        X Y Z : AlgebraicGeometry.Scheme
        𝒰 : X.OpenCover
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
        i j k : 𝒰.J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · apply pullback.hom_ext
        /-
          case h₀.h₀.h₀
          X Y Z : AlgebraicGeometry.Scheme
          𝒰 : X.OpenCover
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
          i j k : 𝒰.J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
      · simp_rw [Category.assoc, cocycle_fst_fst_fst 𝒰 f g i j k]
        /-
          🎉 no goals
        -/
        /-
          case h₀.h₀.h₁
          X Y Z : AlgebraicGeometry.Scheme
          𝒰 : X.OpenCover
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
          i j k : 𝒰.J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
      · simp_rw [Category.assoc, cocycle_fst_fst_snd 𝒰 f g i j k]
        /-
          🎉 no goals
        -/
      /-
        case h₀.h₁
        X Y Z : AlgebraicGeometry.Scheme
        𝒰 : X.OpenCover
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
        i j k : 𝒰.J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp_rw [Category.assoc, cocycle_fst_snd 𝒰 f g i j k]
      /-
        🎉 no goals
      -/
    /-
      case h₁
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      i j k : 𝒰.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · apply pullback.hom_ext
      /-
        case h₁.h₀
        X Y Z : AlgebraicGeometry.Scheme
        𝒰 : X.OpenCover
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
        i j k : 𝒰.J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · apply pullback.hom_ext
        /-
          case h₁.h₀.h₀
          X Y Z : AlgebraicGeometry.Scheme
          𝒰 : X.OpenCover
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
          i j k : 𝒰.J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
      · simp_rw [Category.assoc, cocycle_snd_fst_fst 𝒰 f g i j k]
        /-
          🎉 no goals
        -/
        /-
          case h₁.h₀.h₁
          X Y Z : AlgebraicGeometry.Scheme
          𝒰 : X.OpenCover
          f : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
          i j k : 𝒰.J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
      · simp_rw [Category.assoc, cocycle_snd_fst_snd 𝒰 f g i j k]
        /-
          🎉 no goals
        -/
      /-
        case h₁.h₁
        X Y Z : AlgebraicGeometry.Scheme
        𝒰 : X.OpenCover
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
        i j k : 𝒰.J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp_rw [Category.assoc, cocycle_snd_snd 𝒰 f g i j k]
      /-
        🎉 no goals
      -/


/-- Given `Uᵢ ×[Z] Y`, this is the glued fibered product `X ×[Z] Y`. -/
@[simps U V f t t', simps (config := .lemmasOnly) J]
def gluing : Scheme.GlueData.{u} where
  J := 𝒰.J
  U i := pullback (𝒰.map i ≫ f) g
  V := fun ⟨i, j⟩ => v 𝒰 f g i j
  -- `p⁻¹(Uᵢ ∩ Uⱼ)` where `p : Uᵢ ×[Z] Y ⟶ Uᵢ ⟶ X`.
  f _ _ := pullback.fst _ _
  f_id _ := inferInstance
  f_open := inferInstance
  t i j := t 𝒰 f g i j
  t_id i := t_id 𝒰 f g i
  t' i j k := t' 𝒰 f g i j k
  t_fac i j k := by
    /-
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      i j k : 𝒰.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((fun i j k => AlgebraicGeometry.Sche …
    -/
    apply pullback.hom_ext
    /-
      case h₀
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      i j k : 𝒰.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    on_goal 1 => apply pullback.hom_ext
    all_goals
      simp only [t'_snd_fst_fst, t'_snd_fst_snd, t'_snd_snd, t_fst_fst, t_fst_snd, t_snd,
        Category.assoc]
  cocycle i j k := cocycle 𝒰 f g i j k


@[simp]
lemma gluing_ι (j : 𝒰.J) :
    (gluing 𝒰 f g).ι j = Multicoequalizer.π (gluing 𝒰 f g).diagram j := rfl


/-- The first projection from the glued scheme into `X`. -/
def p1 : (gluing 𝒰 f g).glued ⟶ X := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    ⊢ Quiver.Hom (AlgebraicGeometry.Scheme.Pullback.gluing 𝒰 f g).glued X
  -/
  apply Multicoequalizer.desc (gluing 𝒰 f g).diagram _ fun i ↦ pullback.fst _ _ ≫ 𝒰.map i
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    ⊢ ∀ (a : (AlgebraicGeometry.Scheme.Pullback.gluing 𝒰 f g).diagram.L), Eq (Cate …
  -/
  simp [t_fst_fst_assoc, ← pullback.condition]
  /-
    🎉 no goals
  -/


/-- The second projection from the glued scheme into `Y`. -/
def p2 : (gluing 𝒰 f g).glued ⟶ Y := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    ⊢ Quiver.Hom (AlgebraicGeometry.Scheme.Pullback.gluing 𝒰 f g).glued Y
  -/
  apply Multicoequalizer.desc _ _ fun i ↦ pullback.snd _ _
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    ⊢ ∀ (a : (AlgebraicGeometry.Scheme.Pullback.gluing 𝒰 f g).diagram.L), Eq (Cate …
  -/
  simp [t_fst_snd]
  /-
    🎉 no goals
  -/


theorem p_comm : p1 𝒰 f g ≫ f = p2 𝒰 f g ≫ g := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.p1 …
  -/
  apply Multicoequalizer.hom_ext
  /-
    case h
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    ⊢ ∀ (b : (AlgebraicGeometry.Scheme.Pullback.gluing 𝒰 f g).diagram.R), Eq (Cate …
  -/
  simp [p1, p2, pullback.condition]
  /-
    🎉 no goals
  -/


/-- (Implementation)
The canonical map `(s.X ×[X] Uᵢ) ×[s.X] (s.X ×[X] Uⱼ) ⟶ (Uᵢ ×[Z] Y) ×[X] Uⱼ`

This is used in `gluedLift`. -/
def gluedLiftPullbackMap (i j : 𝒰.J) :
    pullback ((𝒰.pullbackCover s.fst).map i) ((𝒰.pullbackCover s.fst).map j) ⟶
      (gluing 𝒰 f g).V ⟨i, j⟩ := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    i j : 𝒰.J
    ⊢ Quiver.Hom (CategoryTheory.Limits.pullback ((AlgebraicGeometry.Scheme.Cover. …
  -/
  refine (pullbackRightPullbackFstIso _ _ _).hom ≫ ?_
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    i j : 𝒰.J
    ⊢ Quiver.Hom (CategoryTheory.Limits.pullback (CategoryTheory.CategoryStruct.co …
  -/
  refine pullback.map _ _ _ _ ?_ (𝟙 _) (𝟙 _) ?_ ?_
  · exact (pullbackSymmetry _ _).hom ≫
      pullback.map _ _ _ _ (𝟙 _) s.snd f (Category.id_comp _).symm s.condition
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      s : CategoryTheory.Limits.PullbackCone f g
      i j : 𝒰.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · simpa using pullback.condition
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      s : CategoryTheory.Limits.PullbackCone f g
      i j : 𝒰.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (𝒰.map j) (CategoryTheory.CategoryStr …
    -/
  · simp only [Category.comp_id, Category.id_comp]
    /-
      🎉 no goals
    -/


@[reassoc]
theorem gluedLiftPullbackMap_fst (i j : 𝒰.J) :
    gluedLiftPullbackMap 𝒰 f g s i j ≫ pullback.fst _ _ =
      pullback.fst _ _ ≫
        (pullbackSymmetry _ _).hom ≫
          pullback.map _ _ _ _ (𝟙 _) s.snd f (Category.id_comp _).symm s.condition := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    i j : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.gl …
  -/
  simp [gluedLiftPullbackMap]
  /-
    🎉 no goals
  -/


@[reassoc]
theorem gluedLiftPullbackMap_snd (i j : 𝒰.J) :
    gluedLiftPullbackMap 𝒰 f g s i j ≫ pullback.snd _ _ = pullback.snd _ _ ≫ pullback.snd _ _ := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    i j : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.gl …
  -/
  simp [gluedLiftPullbackMap]
  /-
    🎉 no goals
  -/


/-- The lifted map `s.X ⟶ (gluing 𝒰 f g).glued` in order to show that `(gluing 𝒰 f g).glued` is
indeed the pullback.

Given a pullback cone `s`, we have the maps `s.fst ⁻¹' Uᵢ ⟶ Uᵢ` and
`s.fst ⁻¹' Uᵢ ⟶ s.X ⟶ Y` that we may lift to a map `s.fst ⁻¹' Uᵢ ⟶ Uᵢ ×[Z] Y`.

to glue these into a map `s.X ⟶ Uᵢ ×[Z] Y`, we need to show that the maps agree on
`(s.fst ⁻¹' Uᵢ) ×[s.X] (s.fst ⁻¹' Uⱼ) ⟶ Uᵢ ×[Z] Y`. This is achieved by showing that both of these
maps factors through `gluedLiftPullbackMap`.
-/
def gluedLift : s.pt ⟶ (gluing 𝒰 f g).glued := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ Quiver.Hom s.pt (AlgebraicGeometry.Scheme.Pullback.gluing 𝒰 f g).glued
  -/
  fapply (𝒰.pullbackCover s.fst).glueMorphisms
  · exact fun i ↦ (pullbackSymmetry _ _).hom ≫
      pullback.map _ _ _ _ (𝟙 _) s.snd f (Category.id_comp _).symm s.condition ≫ (gluing 𝒰 f g).ι i
  /-
    case hf
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ ∀ (x y : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).J), Eq (Cate …
  -/
  intro i j
  rw [← gluedLiftPullbackMap_fst_assoc, ← gluing_f, ← (gluing 𝒰 f g).glue_condition i j,
    gluing_t, gluing_f]
  /-
    case hf
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    i j : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.gl …
  -/
  simp_rw [← Category.assoc]
  /-
    case hf
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    i j : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  congr 1
  /-
    case hf.e_a
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    i j : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  apply pullback.hom_ext <;> simp_rw [Category.assoc]
    /-
      case hf.e_a.h₀
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      s : CategoryTheory.Limits.PullbackCone f g
      i j : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.gl …
    -/
  · rw [t_fst_fst, gluedLiftPullbackMap_snd]
    /-
      case hf.e_a.h₀
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      s : CategoryTheory.Limits.PullbackCone f g
      i j : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd ( …
    -/
    congr 1
    /-
      case hf.e_a.h₀.e_a
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      s : CategoryTheory.Limits.PullbackCone f g
      i j : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).J
      ⊢ Eq (CategoryTheory.Limits.pullback.snd s.fst (𝒰.map j)) (CategoryTheory.Cate …
    -/
    rw [← Iso.inv_comp_eq, pullbackSymmetry_inv_comp_snd, pullback.lift_fst, Category.comp_id]
    /-
      🎉 no goals
    -/
    /-
      case hf.e_a.h₁
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      s : CategoryTheory.Limits.PullbackCone f g
      i j : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.gl …
    -/
  · rw [t_fst_snd, gluedLiftPullbackMap_fst_assoc, pullback.lift_snd, pullback.lift_snd]
    /-
      case hf.e_a.h₁
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      s : CategoryTheory.Limits.PullbackCone f g
      i j : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
    -/
    simp_rw [pullbackSymmetry_hom_comp_snd_assoc]
    /-
      case hf.e_a.h₁
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      s : CategoryTheory.Limits.PullbackCone f g
      i j : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
    -/
    exact pullback.condition_assoc _
    /-
      🎉 no goals
    -/


theorem gluedLift_p1 : gluedLift 𝒰 f g s ≫ p1 𝒰 f g = s.fst := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.gl …
  -/
  rw [← cancel_epi (𝒰.pullbackCover s.fst).fromGlued]
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Cover.pullb …
  -/
  apply Multicoequalizer.hom_ext
  /-
    case h
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ ∀ (b : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).gluedCover.dia …
  -/
  intro b
  /-
    case h
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    b : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).gluedCover.diagram.R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multicoequaliz …
  -/
  simp_rw [Cover.fromGlued, Multicoequalizer.π_desc_assoc, gluedLift, ← Category.assoc]
  /-
    case h
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    b : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).gluedCover.diagram.R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp_rw [(𝒰.pullbackCover s.fst).ι_glueMorphisms]
  /-
    case h
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    b : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).gluedCover.diagram.R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [p1, pullback.condition]
  /-
    🎉 no goals
  -/


theorem gluedLift_p2 : gluedLift 𝒰 f g s ≫ p2 𝒰 f g = s.snd := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.gl …
  -/
  rw [← cancel_epi (𝒰.pullbackCover s.fst).fromGlued]
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Cover.pullb …
  -/
  apply Multicoequalizer.hom_ext
  /-
    case h
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ ∀ (b : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).gluedCover.dia …
  -/
  intro b
  /-
    case h
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    b : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).gluedCover.diagram.R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.Multicoequaliz …
  -/
  simp_rw [Cover.fromGlued, Multicoequalizer.π_desc_assoc, gluedLift, ← Category.assoc]
  /-
    case h
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    b : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).gluedCover.diagram.R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp_rw [(𝒰.pullbackCover s.fst).ι_glueMorphisms]
  /-
    case h
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    b : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).gluedCover.diagram.R
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  simp [p2, pullback.condition]
  /-
    🎉 no goals
  -/


/-- (Implementation)
The canonical map `(W ×[X] Uᵢ) ×[W] (Uⱼ ×[Z] Y) ⟶ (Uⱼ ×[Z] Y) ×[X] Uᵢ = V j i` where `W` is
the glued fibred product.

This is used in `lift_comp_ι`. -/
def pullbackFstιToV (i j : 𝒰.J) :
    pullback (pullback.fst (p1 𝒰 f g) (𝒰.map i)) ((gluing 𝒰 f g).ι j) ⟶
      v 𝒰 f g j i :=
  (pullbackSymmetry _ _ ≪≫ pullbackRightPullbackFstIso (p1 𝒰 f g) (𝒰.map i) _).hom ≫
    (pullback.congrHom (Multicoequalizer.π_desc ..) rfl).hom


@[simp, reassoc]
theorem pullbackFstιToV_fst (i j : 𝒰.J) :
    pullbackFstιToV 𝒰 f g i j ≫ pullback.fst _ _ = pullback.snd _ _ := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i j : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.pu …
  -/
  simp [pullbackFstιToV, p1]
  /-
    🎉 no goals
  -/


@[simp, reassoc]
theorem pullbackFstιToV_snd (i j : 𝒰.J) :
    pullbackFstιToV 𝒰 f g i j ≫ pullback.snd _ _ = pullback.fst _ _ ≫ pullback.snd _ _ := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i j : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.pu …
  -/
  simp [pullbackFstιToV, p1]
  /-
    🎉 no goals
  -/


/-- We show that the map `W ×[X] Uᵢ ⟶ Uᵢ ×[Z] Y ⟶ W` is the first projection, where the
first map is given by the lift of `W ×[X] Uᵢ ⟶ Uᵢ` and `W ×[X] Uᵢ ⟶ W ⟶ Y`.

It suffices to show that the two map agrees when restricted onto `Uⱼ ×[Z] Y`. In this case,
both maps factor through `V j i` via `pullback_fst_ι_to_V` -/
theorem lift_comp_ι (i : 𝒰.J) :
    pullback.lift (pullback.snd _ _) (pullback.fst _ _ ≫ p2 𝒰 f g)
              /-
                C : Type u
                inst✝¹ : CategoryTheory.Category.{v, u} C
                X Y Z : AlgebraicGeometry.Scheme
                𝒰 : X.OpenCover
                f : Quiver.Hom X Z
                g : Quiver.Hom Y Z
                inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
                s : CategoryTheory.Limits.PullbackCone f g
                i : 𝒰.J
                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd ( …
              -/
          (by rw [← pullback.condition_assoc, Category.assoc, p_comm]) ≫
              /-
                🎉 no goals
              -/
        (gluing 𝒰 f g).ι i =
      (pullback.fst _ _ : pullback (p1 𝒰 f g) (𝒰.map i) ⟶ _) := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
  -/
  apply ((gluing 𝒰 f g).openCover.pullbackCover (pullback.fst _ _)).hom_ext
  /-
    case h
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i : 𝒰.J
    ⊢ ∀ (x : (AlgebraicGeometry.Scheme.Cover.pullbackCover (AlgebraicGeometry.Sche …
  -/
  intro j
  /-
    case h
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i : 𝒰.J
    j : (AlgebraicGeometry.Scheme.Cover.pullbackCover (AlgebraicGeometry.Scheme.Pu …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Scheme.Cover.pull …
  -/
  dsimp only [Cover.pullbackCover]
  /-
    case h
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i : 𝒰.J
    j : (AlgebraicGeometry.Scheme.Cover.pullbackCover (AlgebraicGeometry.Scheme.Pu …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
  -/
  trans pullbackFstιToV 𝒰 f g i j ≫ fV 𝒰 f g j i ≫ (gluing 𝒰 f g).ι _
    /-
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      i : 𝒰.J
      j : (AlgebraicGeometry.Scheme.Cover.pullbackCover (AlgebraicGeometry.Scheme.Pu …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
    -/
  · rw [← show _ = fV 𝒰 f g j i ≫ _ from (gluing 𝒰 f g).glue_condition j i]
    /-
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      i : 𝒰.J
      j : (AlgebraicGeometry.Scheme.Cover.pullbackCover (AlgebraicGeometry.Scheme.Pu …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
    -/
    simp_rw [← Category.assoc]
    /-
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      i : 𝒰.J
      j : (AlgebraicGeometry.Scheme.Cover.pullbackCover (AlgebraicGeometry.Scheme.Pu …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    congr 1
    /-
      case e_a
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      i : 𝒰.J
      j : (AlgebraicGeometry.Scheme.Cover.pullbackCover (AlgebraicGeometry.Scheme.Pu …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
    -/
    rw [gluing_f, gluing_t]
    /-
      case e_a
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      i : 𝒰.J
      j : (AlgebraicGeometry.Scheme.Cover.pullbackCover (AlgebraicGeometry.Scheme.Pu …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
    -/
    apply pullback.hom_ext <;> simp_rw [Category.assoc]
      /-
        case e_a.h₀
        X Y Z : AlgebraicGeometry.Scheme
        𝒰 : X.OpenCover
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
        i : 𝒰.J
        j : (AlgebraicGeometry.Scheme.Cover.pullbackCover (AlgebraicGeometry.Scheme.Pu …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
      -/
    · simp_rw [t_fst_fst, pullback.lift_fst, pullbackFstιToV_snd, GlueData.openCover_map]
      /-
        🎉 no goals
      -/
    · simp_rw [t_fst_snd, pullback.lift_snd, pullbackFstιToV_fst_assoc, pullback.condition_assoc,
        GlueData.openCover_map, p2]
      /-
        case e_a.h₁
        X Y Z : AlgebraicGeometry.Scheme
        𝒰 : X.OpenCover
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
        i : 𝒰.J
        j : (AlgebraicGeometry.Scheme.Cover.pullbackCover (AlgebraicGeometry.Scheme.Pu …
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd ( …
      -/
      simp
      /-
        🎉 no goals
      -/
    /-
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      i : 𝒰.J
      j : (AlgebraicGeometry.Scheme.Cover.pullbackCover (AlgebraicGeometry.Scheme.Pu …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.pu …
    -/
  · rw [pullback.condition, ← Category.assoc]
    /-
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      i : 𝒰.J
      j : (AlgebraicGeometry.Scheme.Cover.pullbackCover (AlgebraicGeometry.Scheme.Pu …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    simp_rw [pullbackFstιToV_fst, GlueData.openCover_map]
    /-
      🎉 no goals
    -/


/-- The canonical isomorphism between `W ×[X] Uᵢ` and `Uᵢ ×[X] Y`. That is, the preimage of `Uᵢ` in
`W` along `p1` is indeed `Uᵢ ×[X] Y`. -/
def pullbackP1Iso (i : 𝒰.J) : pullback (p1 𝒰 f g) (𝒰.map i) ≅ pullback (𝒰.map i ≫ f) g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    i : 𝒰.J
    ⊢ CategoryTheory.Iso (CategoryTheory.Limits.pullback (AlgebraicGeometry.Scheme …
  -/
  fconstructor
  · exact
      pullback.lift (pullback.snd _ _) (pullback.fst _ _ ≫ p2 𝒰 f g)
        (by rw [← pullback.condition_assoc, Category.assoc, p_comm])
    /-
      case inv
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      s : CategoryTheory.Limits.PullbackCone f g
      i : 𝒰.J
      ⊢ Quiver.Hom (CategoryTheory.Limits.pullback (CategoryTheory.CategoryStruct.co …
    -/
  · apply pullback.lift ((gluing 𝒰 f g).ι i) (pullback.fst _ _)
    /-
      case inv
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      s : CategoryTheory.Limits.PullbackCone f g
      i : 𝒰.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Scheme.Pullback.g …
    -/
    rw [gluing_ι, p1, Multicoequalizer.π_desc]
    /-
      🎉 no goals
    -/
    /-
      case hom_inv_id
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      s : CategoryTheory.Limits.PullbackCone f g
      i : 𝒰.J
      ⊢ autoParam (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pul …
    -/
  · apply pullback.hom_ext
      /-
        case hom_inv_id.h₀
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        X Y Z : AlgebraicGeometry.Scheme
        𝒰 : X.OpenCover
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
        s : CategoryTheory.Limits.PullbackCone f g
        i : 𝒰.J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simpa using lift_comp_ι 𝒰 f g i
      /-
        🎉 no goals
      -/
      /-
        case hom_inv_id.h₁
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        X Y Z : AlgebraicGeometry.Scheme
        𝒰 : X.OpenCover
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
        s : CategoryTheory.Limits.PullbackCone f g
        i : 𝒰.J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp_rw [Category.assoc, pullback.lift_snd, pullback.lift_fst, Category.id_comp]
      /-
        🎉 no goals
      -/
    /-
      case inv_hom_id
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰 : X.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
      s : CategoryTheory.Limits.PullbackCone f g
      i : 𝒰.J
      ⊢ autoParam (Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pul …
    -/
  · apply pullback.hom_ext
      /-
        case inv_hom_id.h₀
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        X Y Z : AlgebraicGeometry.Scheme
        𝒰 : X.OpenCover
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
        s : CategoryTheory.Limits.PullbackCone f g
        i : 𝒰.J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp_rw [Category.assoc, pullback.lift_fst, pullback.lift_snd, Category.id_comp]
      /-
        🎉 no goals
      -/
      /-
        case inv_hom_id.h₁
        C : Type u
        inst✝¹ : CategoryTheory.Category.{v, u} C
        X Y Z : AlgebraicGeometry.Scheme
        𝒰 : X.OpenCover
        f : Quiver.Hom X Z
        g : Quiver.Hom Y Z
        inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
        s : CategoryTheory.Limits.PullbackCone f g
        i : 𝒰.J
        ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
      -/
    · simp [p2]
      /-
        🎉 no goals
      -/


@[simp, reassoc]
theorem pullbackP1Iso_hom_fst (i : 𝒰.J) :
    (pullbackP1Iso 𝒰 f g i).hom ≫ pullback.fst _ _ = pullback.snd _ _ := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.pu …
  -/
  simp_rw [pullbackP1Iso, pullback.lift_fst]
  /-
    🎉 no goals
  -/


@[simp, reassoc]
theorem pullbackP1Iso_hom_snd (i : 𝒰.J) :
    (pullbackP1Iso 𝒰 f g i).hom ≫ pullback.snd _ _ = pullback.fst _ _ ≫ p2 𝒰 f g := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.pu …
  -/
  simp_rw [pullbackP1Iso, pullback.lift_snd]
  /-
    🎉 no goals
  -/


@[simp, reassoc]
theorem pullbackP1Iso_inv_fst (i : 𝒰.J) :
    (pullbackP1Iso 𝒰 f g i).inv ≫ pullback.fst _ _ = (gluing 𝒰 f g).ι i := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.pu …
  -/
  simp_rw [pullbackP1Iso, pullback.lift_fst]
  /-
    🎉 no goals
  -/


@[simp, reassoc]
theorem pullbackP1Iso_inv_snd (i : 𝒰.J) :
    (pullbackP1Iso 𝒰 f g i).inv ≫ pullback.snd _ _ = pullback.fst _ _ := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.pu …
  -/
  simp_rw [pullbackP1Iso, pullback.lift_snd]
  /-
    🎉 no goals
  -/


@[simp, reassoc]
theorem pullbackP1Iso_hom_ι (i : 𝒰.J) :
    (pullbackP1Iso 𝒰 f g i).hom ≫ Multicoequalizer.π (gluing 𝒰 f g).diagram i =
    pullback.fst _ _ := by
  /-
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    i : 𝒰.J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.Scheme.Pullback.pu …
  -/
  rw [← gluing_ι, ← pullbackP1Iso_inv_fst, Iso.hom_inv_id_assoc]
  /-
    🎉 no goals
  -/


/-- The glued scheme (`(gluing 𝒰 f g).glued`) is indeed the pullback of `f` and `g`. -/
def gluedIsLimit : IsLimit (PullbackCone.mk _ _ (p_comm 𝒰 f g)) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk (Algebr …
  -/
  apply PullbackCone.isLimitAux'
  /-
    case create
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f g
    ⊢ (s : CategoryTheory.Limits.PullbackCone f g) → Subtype fun l => And (Eq (Cat …
  -/
  intro s
  /-
    case create
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s✝ s : CategoryTheory.Limits.PullbackCone f g
    ⊢ Subtype fun l => And (Eq (CategoryTheory.CategoryStruct.comp l (CategoryTheo …
  -/
  refine ⟨gluedLift 𝒰 f g s, gluedLift_p1 𝒰 f g s, gluedLift_p2 𝒰 f g s, ?_⟩
  /-
    case create
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s✝ s : CategoryTheory.Limits.PullbackCone f g
    ⊢ ∀ {m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk (AlgebraicGeom …
  -/
  intro m h₁ h₂
  /-
    case create
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s✝ s : CategoryTheory.Limits.PullbackCone f g
    m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk (AlgebraicGeometry. …
    h₁ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.PullbackC …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp m (CategoryTheory.Limits.PullbackC …
    ⊢ Eq m (AlgebraicGeometry.Scheme.Pullback.gluedLift 𝒰 f g s)
  -/
  simp_rw [PullbackCone.mk_pt, PullbackCone.mk_π_app] at h₁ h₂
  /-
    case create
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s✝ s : CategoryTheory.Limits.PullbackCone f g
    m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk (AlgebraicGeometry. …
    h₁ : Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.Scheme.Pullba …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.Scheme.Pullba …
    ⊢ Eq m (AlgebraicGeometry.Scheme.Pullback.gluedLift 𝒰 f g s)
  -/
  apply (𝒰.pullbackCover s.fst).hom_ext
  /-
    case create.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s✝ s : CategoryTheory.Limits.PullbackCone f g
    m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk (AlgebraicGeometry. …
    h₁ : Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.Scheme.Pullba …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.Scheme.Pullba …
    ⊢ ∀ (x : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).J), Eq (Catego …
  -/
  intro i
  /-
    case create.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s✝ s : CategoryTheory.Limits.PullbackCone f g
    m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk (AlgebraicGeometry. …
    h₁ : Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.Scheme.Pullba …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.Scheme.Pullba …
    i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicGeometry.Scheme.Cover.pull …
  -/
  rw [gluedLift, (𝒰.pullbackCover s.fst).ι_glueMorphisms, 𝒰.pullbackCover_map]
  rw [← cancel_epi
    (pullbackRightPullbackFstIso (p1 𝒰 f g) (𝒰.map i) m ≪≫ pullback.congrHom h₁ rfl).hom,
    Iso.trans_hom, Category.assoc, pullback.congrHom_hom, pullback.lift_fst_assoc,
    Category.comp_id, pullbackRightPullbackFstIso_hom_fst_assoc, pullback.condition]
  /-
    case create.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s✝ s : CategoryTheory.Limits.PullbackCone f g
    m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk (AlgebraicGeometry. …
    h₁ : Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.Scheme.Pullba …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.Scheme.Pullba …
    i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd m …
  -/
  conv_lhs => rhs; rw [← pullbackP1Iso_hom_ι]
  /-
    case create.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s✝ s : CategoryTheory.Limits.PullbackCone f g
    m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk (AlgebraicGeometry. …
    h₁ : Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.Scheme.Pullba …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.Scheme.Pullba …
    i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd m …
  -/
  simp_rw [← Category.assoc]
  /-
    case create.h
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s✝ s : CategoryTheory.Limits.PullbackCone f g
    m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk (AlgebraicGeometry. …
    h₁ : Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.Scheme.Pullba …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.Scheme.Pullba …
    i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  congr 1
  /-
    case create.h.e_a
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s✝ s : CategoryTheory.Limits.PullbackCone f g
    m : Quiver.Hom s.pt (CategoryTheory.Limits.PullbackCone.mk (AlgebraicGeometry. …
    h₁ : Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.Scheme.Pullba …
    h₂ : Eq (CategoryTheory.CategoryStruct.comp m (AlgebraicGeometry.Scheme.Pullba …
    i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 s.fst).J
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd m …
  -/
  apply pullback.hom_ext
  · simp_rw [Category.assoc, pullbackP1Iso_hom_fst, pullback.lift_fst, Category.comp_id,
      pullbackSymmetry_hom_comp_fst, pullback.lift_snd, Category.comp_id,
      pullbackRightPullbackFstIso_hom_snd]
  · simp_rw [Category.assoc, pullbackP1Iso_hom_snd, pullback.lift_snd,
      pullbackSymmetry_hom_comp_snd_assoc, pullback.lift_fst_assoc, Category.comp_id,
      pullbackRightPullbackFstIso_hom_fst_assoc, ← pullback.condition_assoc, h₂]


include 𝒰 in
theorem hasPullback_of_cover : HasPullback f g :=
  ⟨⟨⟨_, gluedIsLimit 𝒰 f g⟩⟩⟩


instance affine_hasPullback {A B C : CommRingCat}
    (f : Spec A ⟶ Spec C)
    (g : Spec B ⟶ Spec C) : HasPullback f g := by
  /-
    C✝ : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C✝
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f✝ : Quiver.Hom X Z
    g✝ : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f✝ g✝
    A B C : CommRingCat
    f : Quiver.Hom (AlgebraicGeometry.Spec A) (AlgebraicGeometry.Spec C)
    g : Quiver.Hom (AlgebraicGeometry.Spec B) (AlgebraicGeometry.Spec C)
    ⊢ CategoryTheory.Limits.HasPullback f g
  -/
  rw [← Scheme.Spec.map_preimage f, ← Scheme.Spec.map_preimage g]
  exact ⟨⟨⟨_, isLimitOfHasPullbackOfPreservesLimit
    Scheme.Spec (Scheme.Spec.preimage f) (Scheme.Spec.preimage g)⟩⟩⟩


theorem affine_affine_hasPullback {B C : CommRingCat} {X : Scheme}
    (f : X ⟶ Spec C) (g : Spec B ⟶ Spec C) :
    HasPullback f g :=
  hasPullback_of_cover X.affineCover f g


instance base_affine_hasPullback {C : CommRingCat} {X Y : Scheme} (f : X ⟶ Spec C)
    (g : Y ⟶ Spec C) : HasPullback f g :=
  @hasPullback_symmetry _ _ _ _ _ _ _
    (@hasPullback_of_cover _ _ _ Y.affineCover g f fun _ =>
      @hasPullback_symmetry _ _ _ _ _ _ _ <| affine_affine_hasPullback _ _)


instance left_affine_comp_pullback_hasPullback {X Y Z : Scheme} (f : X ⟶ Z) (g : Y ⟶ Z)
    (i : Z.affineCover.J) : HasPullback ((Z.affineCover.pullbackCover f).map i ≫ f) g := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X✝ Y✝ Z✝ : AlgebraicGeometry.Scheme
    𝒰 : X✝.OpenCover
    f✝ : Quiver.Hom X✝ Z✝
    g✝ : Quiver.Hom Y✝ Z✝
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f✝ g✝
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    i : Z.affineCover.J
    ⊢ CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp ((Alge …
  -/
  simp only [Cover.pullbackCover_obj, Cover.pullbackCover_map, pullback.condition]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X✝ Y✝ Z✝ : AlgebraicGeometry.Scheme
    𝒰 : X✝.OpenCover
    f✝ : Quiver.Hom X✝ Z✝
    g✝ : Quiver.Hom Y✝ Z✝
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f✝ g✝
    X Y Z : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    i : Z.affineCover.J
    ⊢ CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp (Categ …
  -/
  exact hasPullback_assoc_symm f (Z.affineCover.map i) (Z.affineCover.map i) g
  /-
    🎉 no goals
  -/


instance {X Y Z : Scheme} (f : X ⟶ Z) (g : Y ⟶ Z) : HasPullback f g :=
  hasPullback_of_cover (Z.affineCover.pullbackCover f) f g


instance : HasPullbacks Scheme :=
  hasPullbacks_of_hasLimit_cospan _


instance isAffine_of_isAffine_isAffine_isAffine {X Y Z : Scheme}
    (f : X ⟶ Z) (g : Y ⟶ Z) [IsAffine X] [IsAffine Y] [IsAffine Z] :
    IsAffine (pullback f g) :=
  isAffine_of_isIso
    (pullback.map f g (Spec.map (Γ.map f.op)) (Spec.map (Γ.map g.op))
        X.toSpecΓ Y.toSpecΓ Z.toSpecΓ
        (Scheme.toSpecΓ_naturality f) (Scheme.toSpecΓ_naturality g) ≫
      (PreservesPullback.iso Scheme.Spec _ _).inv)


/-- Given an open cover `{ Xᵢ }` of `X`, then `X ×[Z] Y` is covered by `Xᵢ ×[Z] Y`. -/
@[simps! J obj map]
def openCoverOfLeft (𝒰 : OpenCover X) (f : X ⟶ Z) (g : Y ⟶ Z) : OpenCover (pullback f g) := by
  fapply
    ((gluing 𝒰 f g).openCover.pushforwardIso
          (limit.isoLimitCone ⟨_, gluedIsLimit 𝒰 f g⟩).inv).copy
      𝒰.J (fun i => pullback (𝒰.map i ≫ f) g)
      (fun i => pullback.map _ _ _ _ (𝒰.map i) (𝟙 _) (𝟙 _) (Category.comp_id _) (by simp))
      (Equiv.refl 𝒰.J) fun _ => Iso.refl _
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰✝ : X.OpenCover
    f✝ : Quiver.Hom X Z
    g✝ : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
    s : CategoryTheory.Limits.PullbackCone f✝ g✝
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ ∀ (i : 𝒰.J), Eq (CategoryTheory.Limits.pullback.map (CategoryTheory.Category …
  -/
  rintro (i : 𝒰.J)
  simp_rw [Cover.pushforwardIso_J, Cover.pushforwardIso_map, GlueData.openCover_map,
    GlueData.openCover_J, gluing_J]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰✝ : X.OpenCover
    f✝ : Quiver.Hom X Z
    g✝ : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
    s : CategoryTheory.Limits.PullbackCone f✝ g✝
    𝒰 : X.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    i : 𝒰.J
    ⊢ Eq (CategoryTheory.Limits.pullback.map (CategoryTheory.CategoryStruct.comp ( …
  -/
  exact pullback.hom_ext (by simp [p1]) (by simp [p2])
  /-
    🎉 no goals
  -/


/-- Given an open cover `{ Yᵢ }` of `Y`, then `X ×[Z] Y` is covered by `X ×[Z] Yᵢ`. -/
@[simps! J obj map]
def openCoverOfRight (𝒰 : OpenCover Y) (f : X ⟶ Z) (g : Y ⟶ Z) : OpenCover (pullback f g) := by
  fapply
    ((openCoverOfLeft 𝒰 g f).pushforwardIso (pullbackSymmetry _ _).hom).copy 𝒰.J
      (fun i => pullback f (𝒰.map i ≫ g))
      (fun i => pullback.map _ _ _ _ (𝟙 _) (𝒰.map i) (𝟙 _) (by simp) (Category.comp_id _))
      (Equiv.refl _) fun i => pullbackSymmetry _ _
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰✝ : X.OpenCover
    f✝ : Quiver.Hom X Z
    g✝ : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
    s : CategoryTheory.Limits.PullbackCone f✝ g✝
    𝒰 : Y.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ ∀ (i : 𝒰.J), Eq (CategoryTheory.Limits.pullback.map f (CategoryTheory.Catego …
  -/
  intro i
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰✝ : X.OpenCover
    f✝ : Quiver.Hom X Z
    g✝ : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
    s : CategoryTheory.Limits.PullbackCone f✝ g✝
    𝒰 : Y.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    i : 𝒰.J
    ⊢ Eq (CategoryTheory.Limits.pullback.map f (CategoryTheory.CategoryStruct.comp …
  -/
  dsimp [Cover.bind]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰✝ : X.OpenCover
    f✝ : Quiver.Hom X Z
    g✝ : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
    s : CategoryTheory.Limits.PullbackCone f✝ g✝
    𝒰 : Y.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    i : 𝒰.J
    ⊢ Eq (CategoryTheory.Limits.pullback.map f (CategoryTheory.CategoryStruct.comp …
  -/
                             /-
                               🎉 no goals
                             -/
  apply pullback.hom_ext <;> simp
                             /-
                               🎉 no goals
                             -/


/-- Given an open cover `{ Xᵢ }` of `X` and an open cover `{ Yⱼ }` of `Y`, then
`X ×[Z] Y` is covered by `Xᵢ ×[Z] Yⱼ`. -/
@[simps! J obj map]
def openCoverOfLeftRight (𝒰X : X.OpenCover) (𝒰Y : Y.OpenCover) (f : X ⟶ Z) (g : Y ⟶ Z) :
    (pullback f g).OpenCover := by
  fapply
    ((openCoverOfLeft 𝒰X f g).bind fun x => openCoverOfRight 𝒰Y (𝒰X.map x ≫ f) g).copy
      (𝒰X.J × 𝒰Y.J) (fun ij => pullback (𝒰X.map ij.1 ≫ f) (𝒰Y.map ij.2 ≫ g))
      (fun ij =>
        pullback.map _ _ _ _ (𝒰X.map ij.1) (𝒰Y.map ij.2) (𝟙 _) (Category.comp_id _)
          (Category.comp_id _))
      (Equiv.sigmaEquivProd _ _).symm fun _ => Iso.refl _
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f✝ : Quiver.Hom X Z
    g✝ : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f✝ g✝
    𝒰X : X.OpenCover
    𝒰Y : Y.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ ∀ (i : Prod 𝒰X.J 𝒰Y.J), Eq (CategoryTheory.Limits.pullback.map (CategoryTheo …
  -/
  rintro ⟨i, j⟩
  /-
    case mk
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰 : X.OpenCover
    f✝ : Quiver.Hom X Z
    g✝ : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Categor …
    s : CategoryTheory.Limits.PullbackCone f✝ g✝
    𝒰X : X.OpenCover
    𝒰Y : Y.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    i : 𝒰X.J
    j : 𝒰Y.J
    ⊢ Eq (CategoryTheory.Limits.pullback.map (CategoryTheory.CategoryStruct.comp ( …
  -/
                             /-
                               🎉 no goals
                             -/
  apply pullback.hom_ext <;> simp
                             /-
                               🎉 no goals
                             -/


/-- (Implementation). Use `openCoverOfBase` instead. -/
@[simps! map]
def openCoverOfBase' (𝒰 : OpenCover Z) (f : X ⟶ Z) (g : Y ⟶ Z) : OpenCover (pullback f g) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰✝ : X.OpenCover
    f✝ : Quiver.Hom X Z
    g✝ : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
    s : CategoryTheory.Limits.PullbackCone f✝ g✝
    𝒰 : Z.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ (CategoryTheory.Limits.pullback f g).OpenCover
  -/
  apply (openCoverOfLeft (𝒰.pullbackCover f) f g).bind
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰✝ : X.OpenCover
    f✝ : Quiver.Hom X Z
    g✝ : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
    s : CategoryTheory.Limits.PullbackCone f✝ g✝
    𝒰 : Z.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ (x : (AlgebraicGeometry.Scheme.Pullback.openCoverOfLeft (AlgebraicGeometry.S …
  -/
  intro i
  haveI := ((IsPullback.of_hasPullback (pullback.snd g (𝒰.map i))
    (pullback.snd f (𝒰.map i))).paste_horiz (IsPullback.of_hasPullback _ _)).flip
  refine
    @coverOfIsIso _ _ _ _ _
      (f := (pullbackSymmetry (pullback.snd f (𝒰.map i)) (pullback.snd g (𝒰.map i))).hom ≫
        (limit.isoLimitCone ⟨_, this.isLimit⟩).inv ≫
        pullback.map _ _ _ _ (𝟙 _) (𝟙 _) (𝟙 _) ?_ ?_) inferInstance
    /-
      case refine_1
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰✝ : X.OpenCover
      f✝ : Quiver.Hom X Z
      g✝ : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      s : CategoryTheory.Limits.PullbackCone f✝ g✝
      𝒰 : Z.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      i : (AlgebraicGeometry.Scheme.Pullback.openCoverOfLeft (AlgebraicGeometry.Sche …
      this : CategoryTheory.IsPullback (CategoryTheory.Limits.pullback.snd (Category …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · simp [← pullback.condition]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰✝ : X.OpenCover
      f✝ : Quiver.Hom X Z
      g✝ : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      s : CategoryTheory.Limits.PullbackCone f✝ g✝
      𝒰 : Z.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      i : (AlgebraicGeometry.Scheme.Pullback.openCoverOfLeft (AlgebraicGeometry.Sche …
      this : CategoryTheory.IsPullback (CategoryTheory.Limits.pullback.snd (Category …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp g (CategoryTheory.CategoryStruct.id Z …
    -/
  · simp only [Category.comp_id, Category.id_comp]
    /-
      🎉 no goals
    -/


/-- Given an open cover `{ Zᵢ }` of `Z`, then `X ×[Z] Y` is covered by `Xᵢ ×[Zᵢ] Yᵢ`, where
  `Xᵢ = X ×[Z] Zᵢ` and `Yᵢ = Y ×[Z] Zᵢ` is the preimage of `Zᵢ` in `X` and `Y`. -/
@[simps! J obj map]
def openCoverOfBase (𝒰 : OpenCover Z) (f : X ⟶ Z) (g : Y ⟶ Z) : OpenCover (pullback f g) := by
  apply
    (openCoverOfBase'.{u, u} 𝒰 f g).copy 𝒰.J
      (fun i =>
        pullback (pullback.snd _ _ : pullback f (𝒰.map i) ⟶ _)
          (pullback.snd _ _ : pullback g (𝒰.map i) ⟶ _))
      (fun i =>
        pullback.map _ _ _ _ (pullback.fst _ _) (pullback.fst _ _) (𝒰.map i)
          pullback.condition.symm pullback.condition.symm)
      ((Equiv.prodPUnit 𝒰.J).symm.trans (Equiv.sigmaEquivProd 𝒰.J PUnit).symm) fun _ => Iso.refl _
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰✝ : X.OpenCover
    f✝ : Quiver.Hom X Z
    g✝ : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
    s : CategoryTheory.Limits.PullbackCone f✝ g✝
    𝒰 : Z.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    ⊢ ∀ (i : 𝒰.J), Eq (CategoryTheory.Limits.pullback.map (CategoryTheory.Limits.p …
  -/
  intro i
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰✝ : X.OpenCover
    f✝ : Quiver.Hom X Z
    g✝ : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
    s : CategoryTheory.Limits.PullbackCone f✝ g✝
    𝒰 : Z.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    i : 𝒰.J
    ⊢ Eq (CategoryTheory.Limits.pullback.map (CategoryTheory.Limits.pullback.snd f …
  -/
  rw [Iso.refl_hom, Category.id_comp, openCoverOfBase'_map]
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰✝ : X.OpenCover
    f✝ : Quiver.Hom X Z
    g✝ : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
    s : CategoryTheory.Limits.PullbackCone f✝ g✝
    𝒰 : Z.OpenCover
    f : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    i : 𝒰.J
    ⊢ Eq (CategoryTheory.Limits.pullback.map (CategoryTheory.Limits.pullback.snd f …
  -/
  ext : 1 <;>
  · simp only [limit.lift_π, PullbackCone.mk_pt, PullbackCone.mk_π_app, Equiv.trans_apply,
      Equiv.prodPUnit_symm_apply, Category.assoc, limit.lift_π_assoc, cospan_left, Category.comp_id,
      limit.isoLimitCone_inv_π_assoc, PullbackCone.π_app_left, IsPullback.cone_fst,
      pullbackSymmetry_hom_comp_snd_assoc, limit.isoLimitCone_inv_π,
      PullbackCone.π_app_right, IsPullback.cone_snd, pullbackSymmetry_hom_comp_fst_assoc]
    /-
      case h₀
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰✝ : X.OpenCover
      f✝ : Quiver.Hom X Z
      g✝ : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      s : CategoryTheory.Limits.PullbackCone f✝ g✝
      𝒰 : Z.OpenCover
      f : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      i : 𝒰.J
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.fst ( …
    -/
    /-
      🎉 no goals
    -/
    rfl
    /-
      🎉 no goals
    -/


/--
Given `𝒰 i` covering `Y` and `𝒱 i j` covering `𝒰 i`, this is the open cover
`𝒱 i j₁ ×[𝒰 i] 𝒱 i j₂` ranging over all `i`, `j₁`, `j₂`.
-/
noncomputable
def diagonalCover : (pullback.diagonalObj f).OpenCover :=
  (openCoverOfBase 𝒰 f f).bind
    fun i ↦ openCoverOfLeftRight (𝒱 i) (𝒱 i) (𝒰.pullbackHom _ _) (𝒰.pullbackHom _ _)


/-- The image of `𝒱 i j₁ ×[𝒰 i] 𝒱 i j₂` in `diagonalCover` with `j₁ = j₂`  -/
noncomputable
def diagonalCoverDiagonalRange : (pullback.diagonalObj f).Opens :=
  ⨆ i : Σ i, (𝒱 i).J, ((diagonalCover f 𝒰 𝒱).map ⟨i.1, i.2, i.2⟩).opensRange


lemma diagonalCover_map (I) : (diagonalCover f 𝒰 𝒱).map I =
    pullback.map _ _ _ _
    ((𝒱 I.fst).map _ ≫ pullback.fst _ _) ((𝒱 I.fst).map _ ≫ pullback.fst _ _) (𝒰.map _)
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X Y Z : AlgebraicGeometry.Scheme
          𝒰✝ : X.OpenCover
          f✝ : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
          s : CategoryTheory.Limits.PullbackCone f✝ g
          f : Quiver.Hom X Y
          𝒰 : Y.OpenCover
          𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
          I : (AlgebraicGeometry.Scheme.Pullback.diagonalCover f 𝒰 𝒱).J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
    (by simp)
        /-
          🎉 no goals
        -/
        /-
          C : Type u
          inst✝¹ : CategoryTheory.Category.{v, u} C
          X Y Z : AlgebraicGeometry.Scheme
          𝒰✝ : X.OpenCover
          f✝ : Quiver.Hom X Z
          g : Quiver.Hom Y Z
          inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
          s : CategoryTheory.Limits.PullbackCone f✝ g
          f : Quiver.Hom X Y
          𝒰 : Y.OpenCover
          𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
          I : (AlgebraicGeometry.Scheme.Pullback.diagonalCover f 𝒰 𝒱).J
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
        -/
    (by simp) := by
        /-
          🎉 no goals
        -/
  /-
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
    I : (AlgebraicGeometry.Scheme.Pullback.diagonalCover f 𝒰 𝒱).J
    ⊢ Eq ((AlgebraicGeometry.Scheme.Pullback.diagonalCover f 𝒰 𝒱).map I) (Category …
  -/
           /-
             🎉 no goals
           -/
  ext1 <;> simp [diagonalCover, Cover.pullbackHom]
           /-
             🎉 no goals
           -/


/-- The restriction of the diagonal `X ⟶ X ×ₛ X` to `𝒱 i j ×[𝒰 i] 𝒱 i j` is the diagonal
`𝒱 i j ⟶ 𝒱 i j ×[𝒰 i] 𝒱 i j`. -/
noncomputable
def diagonalRestrictIsoDiagonal (i j) :
    Arrow.mk (pullback.diagonal f ∣_ ((diagonalCover f 𝒰 𝒱).map ⟨i, j, j⟩).opensRange) ≅
    Arrow.mk (pullback.diagonal ((𝒱 i).map j ≫ pullback.snd _ _)) := by
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰✝ : X.OpenCover
    f✝ : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
    s : CategoryTheory.Limits.PullbackCone f✝ g
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
    i : (AlgebraicGeometry.Scheme.Pullback.openCoverOfBase 𝒰 f f).J
    j : (𝒱 i).J
    ⊢ CategoryTheory.Iso (CategoryTheory.Arrow.mk (AlgebraicGeometry.morphismRestr …
  -/
  refine (morphismRestrictOpensRange _ _).trans ?_
  /-
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰✝ : X.OpenCover
    f✝ : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
    s : CategoryTheory.Limits.PullbackCone f✝ g
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
    i : (AlgebraicGeometry.Scheme.Pullback.openCoverOfBase 𝒰 f f).J
    j : (𝒱 i).J
    ⊢ CategoryTheory.Iso (CategoryTheory.Arrow.mk (CategoryTheory.Limits.pullback. …
  -/
  refine Arrow.isoMk ?_ (Iso.refl _) ?_
  · exact pullback.congrHom rfl (diagonalCover_map _ _ _ _) ≪≫
      pullbackDiagonalMapIso _ _ _ _ ≪≫ (asIso (pullback.diagonal _)).symm
  have H : pullback.snd (pullback.diagonal f) ((diagonalCover f 𝒰 𝒱).map ⟨i, (j, j)⟩) ≫
      pullback.snd _ _ = pullback.snd _ _ ≫ pullback.fst _ _ := by
    rw [← cancel_mono ((𝒱 i).map _)]
    apply pullback.hom_ext
    · trans pullback.snd (pullback.diagonal f) ((diagonalCover f 𝒰 𝒱).map ⟨i, (j, j)⟩) ≫
        (diagonalCover f 𝒰 𝒱).map _ ≫ pullback.snd _ _
      · simp [diagonalCover_map]
      symm
      trans pullback.snd (pullback.diagonal f) ((diagonalCover f 𝒰 𝒱).map ⟨i, (j, j)⟩) ≫
        (diagonalCover f 𝒰 𝒱).map _ ≫ pullback.fst _ _
      · simp [diagonalCover_map]
      · rw [← pullback.condition_assoc, ← pullback.condition_assoc]
        simp
    · simp [pullback.condition, Cover.pullbackHom]
  /-
    case refine_2
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰✝ : X.OpenCover
    f✝ : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
    s : CategoryTheory.Limits.PullbackCone f✝ g
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
    i : (AlgebraicGeometry.Scheme.Pullback.openCoverOfBase 𝒰 f f).J
    j : (𝒱 i).J
    H : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((CategoryTheory.Limits.pullback.cong …
  -/
  dsimp [Cover.pullbackHom] at H ⊢
  /-
    case refine_2
    C : Type u
    inst✝¹ : CategoryTheory.Category.{v, u} C
    X Y Z : AlgebraicGeometry.Scheme
    𝒰✝ : X.OpenCover
    f✝ : Quiver.Hom X Z
    g : Quiver.Hom Y Z
    inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
    s : CategoryTheory.Limits.PullbackCone f✝ g
    f : Quiver.Hom X Y
    𝒰 : Y.OpenCover
    𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
    i : (AlgebraicGeometry.Scheme.Pullback.openCoverOfBase 𝒰 f f).J
    j : (𝒱 i).J
    H : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
  -/
  apply pullback.hom_ext
    /-
      case refine_2.h₀
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰✝ : X.OpenCover
      f✝ : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      s : CategoryTheory.Limits.PullbackCone f✝ g
      f : Quiver.Hom X Y
      𝒰 : Y.OpenCover
      𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
      i : (AlgebraicGeometry.Scheme.Pullback.openCoverOfBase 𝒰 f f).J
      j : (𝒱 i).J
      H : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · simp only [Category.assoc, pullback.diagonal_fst, Category.comp_id]
    /-
      case refine_2.h₀
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰✝ : X.OpenCover
      f✝ : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      s : CategoryTheory.Limits.PullbackCone f✝ g
      f : Quiver.Hom X Y
      𝒰 : Y.OpenCover
      𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
      i : (AlgebraicGeometry.Scheme.Pullback.openCoverOfBase 𝒰 f f).J
      j : (𝒱 i).J
      H : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.map ( …
    -/
    simp only [← Category.assoc, IsIso.comp_inv_eq]
    /-
      case refine_2.h₀
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰✝ : X.OpenCover
      f✝ : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      s : CategoryTheory.Limits.PullbackCone f✝ g
      f : Quiver.Hom X Y
      𝒰 : Y.OpenCover
      𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
      i : (AlgebraicGeometry.Scheme.Pullback.openCoverOfBase 𝒰 f f).J
      j : (𝒱 i).J
      H : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.map ( …
    -/
                               /-
                                 🎉 no goals
                               -/
    apply pullback.hom_ext <;> simp [H]
                               /-
                                 🎉 no goals
                               -/
    /-
      case refine_2.h₁
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰✝ : X.OpenCover
      f✝ : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      s : CategoryTheory.Limits.PullbackCone f✝ g
      f : Quiver.Hom X Y
      𝒰 : Y.OpenCover
      𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
      i : (AlgebraicGeometry.Scheme.Pullback.openCoverOfBase 𝒰 f f).J
      j : (𝒱 i).J
      H : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
  · simp only [Category.assoc, pullback.diagonal_snd, Category.comp_id]
    /-
      case refine_2.h₁
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰✝ : X.OpenCover
      f✝ : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      s : CategoryTheory.Limits.PullbackCone f✝ g
      f : Quiver.Hom X Y
      𝒰 : Y.OpenCover
      𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
      i : (AlgebraicGeometry.Scheme.Pullback.openCoverOfBase 𝒰 f f).J
      j : (𝒱 i).J
      H : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.map ( …
    -/
    simp only [← Category.assoc, IsIso.comp_inv_eq]
    /-
      case refine_2.h₁
      C : Type u
      inst✝¹ : CategoryTheory.Category.{v, u} C
      X Y Z : AlgebraicGeometry.Scheme
      𝒰✝ : X.OpenCover
      f✝ : Quiver.Hom X Z
      g : Quiver.Hom Y Z
      inst✝ : ∀ (i : 𝒰✝.J), CategoryTheory.Limits.HasPullback (CategoryTheory.Catego …
      s : CategoryTheory.Limits.PullbackCone f✝ g
      f : Quiver.Hom X Y
      𝒰 : Y.OpenCover
      𝒱 : (i : (AlgebraicGeometry.Scheme.Cover.pullbackCover 𝒰 f).J) → ((AlgebraicGe …
      i : (AlgebraicGeometry.Scheme.Pullback.openCoverOfBase 𝒰 f f).J
      j : (𝒱 i).J
      H : Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.snd …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.map ( …
    -/
                               /-
                                 🎉 no goals
                               -/
    apply pullback.hom_ext <;> simp [H]
                               /-
                                 🎉 no goals
                               -/


instance Scheme.pullback_map_isOpenImmersion {X Y S X' Y' S' : Scheme}
    (f : X ⟶ S) (g : Y ⟶ S) (f' : X' ⟶ S') (g' : Y' ⟶ S')
    (i₁ : X ⟶ X') (i₂ : Y ⟶ Y') (i₃ : S ⟶ S') (e₁ : f ≫ i₃ = i₁ ≫ f') (e₂ : g ≫ i₃ = i₂ ≫ g')
    [IsOpenImmersion i₁] [IsOpenImmersion i₂] [Mono i₃] :
    IsOpenImmersion (pullback.map f g f' g' i₁ i₂ i₃ e₁ e₂) := by
  /-
    X Y S X' Y' S' : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    f' : Quiver.Hom X' S'
    g' : Quiver.Hom Y' S'
    i₁ : Quiver.Hom X X'
    i₂ : Quiver.Hom Y Y'
    i₃ : Quiver.Hom S S'
    e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
    e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
    inst✝² : AlgebraicGeometry.IsOpenImmersion i₁
    inst✝¹ : AlgebraicGeometry.IsOpenImmersion i₂
    inst✝ : CategoryTheory.Mono i₃
    ⊢ AlgebraicGeometry.IsOpenImmersion (CategoryTheory.Limits.pullback.map f g f' …
  -/
  rw [pullback_map_eq_pullbackFstFstIso_inv]
  /-
    X Y S X' Y' S' : AlgebraicGeometry.Scheme
    f : Quiver.Hom X S
    g : Quiver.Hom Y S
    f' : Quiver.Hom X' S'
    g' : Quiver.Hom Y' S'
    i₁ : Quiver.Hom X X'
    i₂ : Quiver.Hom Y Y'
    i₃ : Quiver.Hom S S'
    e₁ : Eq (CategoryTheory.CategoryStruct.comp f i₃) (CategoryTheory.CategoryStru …
    e₂ : Eq (CategoryTheory.CategoryStruct.comp g i₃) (CategoryTheory.CategoryStru …
    inst✝² : AlgebraicGeometry.IsOpenImmersion i₁
    inst✝¹ : AlgebraicGeometry.IsOpenImmersion i₂
    inst✝ : CategoryTheory.Mono i₃
    ⊢ AlgebraicGeometry.IsOpenImmersion (CategoryTheory.CategoryStruct.comp (Categ …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- The isomorphism between the fiber product of two schemes `Spec S` and `Spec T`
over a scheme `Spec R` and the `Spec` of the tensor product `S ⊗[R] T`.-/
noncomputable
def pullbackSpecIso :
    pullback (Spec.map (CommRingCat.ofHom (algebraMap R S)))
      (Spec.map (CommRingCat.ofHom (algebraMap R T))) ≅ Spec (.of <| S ⊗[R] T) :=
  letI H := IsLimit.equivIsoLimit (PullbackCone.eta _)
    (PushoutCocone.isColimitEquivIsLimitOp _ (CommRingCat.pushoutCoconeIsColimit R S T))
  limit.isoLimitCone ⟨_, isLimitPullbackConeMapOfIsLimit Scheme.Spec _ H⟩


/--
The composition of the inverse of the isomorphism `pullbackSepcIso R S T` (from the pullback of
`Spec S ⟶ Spec R` and `Spec T ⟶ Spec R` to `Spec (S ⊗[R] T)`) with the first projection is
the morphism `Spec (S ⊗[R] T) ⟶ Spec S` obtained by applying `Spec.map` to the ring morphism
`s ↦ s ⊗ₜ[R] 1`.
-/
@[reassoc (attr := simp)]
lemma pullbackSpecIso_inv_fst :
    (pullbackSpecIso R S T).inv ≫ pullback.fst _ _ = Spec.map (ofHom includeLeftRingHom) :=
  limit.isoLimitCone_inv_π _ _


/--
The composition of the inverse of the isomorphism `pullbackSepcIso R S T` (from the pullback of
`Spec S ⟶ Spec R` and `Spec T ⟶ Spec R` to `Spec (S ⊗[R] T)`) with the second projection is
the morphism `Spec (S ⊗[R] T) ⟶ Spec T` obtained by applying `Spec.map` to the ring morphism
`t ↦ 1 ⊗ₜ[R] t`.
-/
@[reassoc (attr := simp)]
lemma pullbackSpecIso_inv_snd :
    (pullbackSpecIso R S T).inv ≫ pullback.snd _ _ =
      Spec.map (ofHom (R := T) (S := S ⊗[R] T) (toRingHom includeRight)) :=
  limit.isoLimitCone_inv_π _ _


/--
The composition of the isomorphism `pullbackSepcIso R S T` (from the pullback of
`Spec S ⟶ Spec R` and `Spec T ⟶ Spec R` to `Spec (S ⊗[R] T)`) with the morphism
`Spec (S ⊗[R] T) ⟶ Spec S` obtained by applying `Spec.map` to the ring morphism `s ↦ s ⊗ₜ[R] 1`
is the first projection.
-/
@[reassoc (attr := simp)]
lemma pullbackSpecIso_hom_fst :
    (pullbackSpecIso R S T).hom ≫ Spec.map (ofHom includeLeftRingHom) = pullback.fst _ _ := by
  /-
    R S T : Type u
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : CommRing T
    inst✝¹ : Algebra R S
    inst✝ : Algebra R T
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.pullbackSpecIso R  …
  -/
  rw [← pullbackSpecIso_inv_fst, Iso.hom_inv_id_assoc]
  /-
    🎉 no goals
  -/


/--
The composition of the isomorphism `pullbackSepcIso R S T` (from the pullback of
`Spec S ⟶ Spec R` and `Spec T ⟶ Spec R` to `Spec (S ⊗[R] T)`) with the morphism
`Spec (S ⊗[R] T) ⟶ Spec T` obtained by applying `Spec.map` to the ring morphism `t ↦ 1 ⊗ₜ[R] t`
is the second projection.
-/
@[reassoc (attr := simp)]
lemma pullbackSpecIso_hom_snd :
    (pullbackSpecIso R S T).hom ≫ Spec.map (ofHom (toRingHom includeRight)) = pullback.snd _ _ := by
  /-
    R S T : Type u
    inst✝⁴ : CommRing R
    inst✝³ : CommRing S
    inst✝² : CommRing T
    inst✝¹ : Algebra R S
    inst✝ : Algebra R T
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicGeometry.pullbackSpecIso R  …
  -/
  rw [← pullbackSpecIso_inv_snd, Iso.hom_inv_id_assoc]
  /-
    🎉 no goals
  -/


lemma isPullback_Spec_map_isPushout {A B C P : CommRingCat} (f : A ⟶ B) (g : A ⟶ C)
    (inl : B ⟶ P) (inr : C ⟶ P) (h : IsPushout f g inl inr) :
    IsPullback (Spec.map inl) (Spec.map inr) (Spec.map f) (Spec.map g) :=
  IsPullback.map Scheme.Spec h.op.flip


lemma isPullback_Spec_map_pushout {A B C : CommRingCat} (f : A ⟶ B) (g : A ⟶ C) :
    IsPullback (Spec.map (pushout.inl f g))
      (Spec.map (pushout.inr f g)) (Spec.map f) (Spec.map g) := by
  /-
    A B C : CommRingCat
    f : Quiver.Hom A B
    g : Quiver.Hom A C
    ⊢ CategoryTheory.IsPullback (AlgebraicGeometry.Spec.map (CategoryTheory.Limits …
  -/
  apply isPullback_Spec_map_isPushout
  /-
    case h
    A B C : CommRingCat
    f : Quiver.Hom A B
    g : Quiver.Hom A C
    ⊢ CategoryTheory.IsPushout f g (CategoryTheory.Limits.pushout.inl f g) (Catego …
  -/
  exact IsPushout.of_hasPushout f g
  /-
    🎉 no goals
  -/


lemma diagonal_Spec_map :
    pullback.diagonal (Spec.map (CommRingCat.ofHom (algebraMap R S))) =
      Spec.map (CommRingCat.ofHom (Algebra.TensorProduct.lmul' R : S ⊗[R] S →ₐ[R] S).toRingHom) ≫
        (pullbackSpecIso R S S).inv := by
  /-
    R S : Type u
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    ⊢ Eq (CategoryTheory.Limits.pullback.diagonal (AlgebraicGeometry.Spec.map (Com …
  -/
  ext1 <;> simp only [pullback.diagonal_fst, pullback.diagonal_snd, ← Spec.map_comp, ← Spec.map_id,
    AlgHom.toRingHom_eq_coe, Category.assoc, pullbackSpecIso_inv_fst, pullbackSpecIso_inv_snd]
    /-
      case h₀
      R S : Type u
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      ⊢ Eq (AlgebraicGeometry.Spec.map (CategoryTheory.CategoryStruct.id (CommRingCa …
    -/
  · congr 1; ext x; show x = Algebra.TensorProduct.lmul' R (S := S) (x ⊗ₜ[R] 1); simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
    /-
      case h₁
      R S : Type u
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      ⊢ Eq (AlgebraicGeometry.Spec.map (CategoryTheory.CategoryStruct.id (CommRingCa …
    -/
  · congr 1; ext x; show x = Algebra.TensorProduct.lmul' R (S := S) (1 ⊗ₜ[R] x); simp
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


