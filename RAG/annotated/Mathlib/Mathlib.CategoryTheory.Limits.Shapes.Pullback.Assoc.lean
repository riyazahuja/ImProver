local notation "Z₁" => pullback f₁ f₂


local notation "Z₂" => pullback f₃ f₄


local notation "g₁" => (pullback.fst _ _ : Z₁ ⟶ X₁)


local notation "g₂" => (pullback.snd _ _ : Z₁ ⟶ X₂)


local notation "g₃" => (pullback.fst _ _ : Z₂ ⟶ X₂)


local notation "g₄" => (pullback.snd _ _ : Z₂ ⟶ X₃)


local notation "W" => pullback (g₂ ≫ f₃) f₄


local notation "W'" => pullback f₁ (g₃ ≫ f₂)


local notation "l₁" => (pullback.fst _ _ : W ⟶ Z₁)


local notation "l₂" =>
  (pullback.lift (pullback.fst _ _ ≫ g₂) (pullback.snd _ _)
      (Eq.trans (Category.assoc _ _ _) pullback.condition) :
    W ⟶ Z₂)


local notation "l₁'" =>
  (pullback.lift (pullback.fst _ _) (pullback.snd _ _ ≫ g₃)
      (pullback.condition.trans (Eq.symm (Category.assoc _ _ _))) :
    W' ⟶ Z₁)


local notation "l₂'" => (pullback.snd f₁ (g₃ ≫ f₂))


/-- `(X₁ ×[Y₁] X₂) ×[Y₂] X₃` is the pullback `(X₁ ×[Y₁] X₂) ×[X₂] (X₂ ×[Y₂] X₃)`. -/
def pullbackPullbackLeftIsPullback [HasPullback (g₂ ≫ f₃) f₄] : IsLimit (PullbackCone.mk l₁ l₂
    (show l₁ ≫ g₂ = l₂ ≫ g₃ from (pullback.lift_fst _ _ _).symm)) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ Y₂ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₁
    f₃ : Quiver.Hom X₂ Y₂
    f₄ : Quiver.Hom X₃ Y₂
    inst✝² : CategoryTheory.Limits.HasPullback f₁ f₂
    inst✝¹ : CategoryTheory.Limits.HasPullback f₃ f₄
    inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp  …
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk (Catego …
  -/
  apply leftSquareIsPullback _ rfl (pullbackIsPullback f₃ f₄)
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ Y₂ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₁
    f₃ : Quiver.Hom X₂ Y₂
    f₄ : Quiver.Hom X₃ Y₂
    inst✝² : CategoryTheory.Limits.HasPullback f₁ f₂
    inst✝¹ : CategoryTheory.Limits.HasPullback f₃ f₄
    inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp  …
    ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.Limits.PullbackCone.mk (Categ …
  -/
  simpa [PullbackCone.pasteHoriz] using PullbackCone.mkSelfIsLimit (pullbackIsPullback _ f₄)
  /-
    🎉 no goals
  -/


/-- `(X₁ ×[Y₁] X₂) ×[Y₂] X₃` is the pullback `X₁ ×[Y₁] (X₂ ×[Y₂] X₃)`. -/
def pullbackAssocIsPullback [HasPullback (g₂ ≫ f₃) f₄] :
    IsLimit
      (PullbackCone.mk (l₁ ≫ g₁) l₂
        (show (l₁ ≫ g₁) ≫ f₁ = l₂ ≫ g₃ ≫ f₂ by
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            X₁ X₂ X₃ Y₁ Y₂ : C
            f₁ : Quiver.Hom X₁ Y₁
            f₂ : Quiver.Hom X₂ Y₁
            f₃ : Quiver.Hom X₂ Y₂
            f₄ : Quiver.Hom X₃ Y₂
            inst✝² : CategoryTheory.Limits.HasPullback f₁ f₂
            inst✝¹ : CategoryTheory.Limits.HasPullback f₃ f₄
            inst✝ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp  …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
          -/
          rw [pullback.lift_fst_assoc, Category.assoc, Category.assoc, pullback.condition])) := by
          /-
            🎉 no goals
          -/
  simpa using pasteVertIsPullback rfl (pullbackIsPullback _ _)
    (pullbackPullbackLeftIsPullback f₁ f₂ f₃ f₄)


theorem hasPullback_assoc [HasPullback (g₂ ≫ f₃) f₄] : HasPullback f₁ (g₃ ≫ f₂) :=
  ⟨⟨⟨_, pullbackAssocIsPullback f₁ f₂ f₃ f₄⟩⟩⟩


/-- `X₁ ×[Y₁] (X₂ ×[Y₂] X₃)` is the pullback `(X₁ ×[Y₁] X₂) ×[X₂] (X₂ ×[Y₂] X₃)`. -/
def pullbackPullbackRightIsPullback [HasPullback f₁ (g₃ ≫ f₂)] :
    IsLimit (PullbackCone.mk l₁' l₂' (show l₁' ≫ g₂ = l₂' ≫ g₃ from pullback.lift_snd _ _ _)) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ Y₂ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₁
    f₃ : Quiver.Hom X₂ Y₂
    f₄ : Quiver.Hom X₃ Y₂
    inst✝² : CategoryTheory.Limits.HasPullback f₁ f₂
    inst✝¹ : CategoryTheory.Limits.HasPullback f₃ f₄
    inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
    ⊢ CategoryTheory.Limits.IsLimit (CategoryTheory.Limits.PullbackCone.mk (Catego …
  -/
  apply topSquareIsPullback _ rfl (pullbackIsPullback f₁ f₂)
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ Y₂ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₁
    f₃ : Quiver.Hom X₂ Y₂
    f₄ : Quiver.Hom X₃ Y₂
    inst✝² : CategoryTheory.Limits.HasPullback f₁ f₂
    inst✝¹ : CategoryTheory.Limits.HasPullback f₃ f₄
    inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
    ⊢ CategoryTheory.Limits.IsLimit ((CategoryTheory.Limits.PullbackCone.mk (Categ …
  -/
  simpa [PullbackCone.pasteVert] using PullbackCone.mkSelfIsLimit (pullbackIsPullback _ _)
  /-
    🎉 no goals
  -/


/-- `X₁ ×[Y₁] (X₂ ×[Y₂] X₃)` is the pullback `(X₁ ×[Y₁] X₂) ×[Y₂] X₃`. -/
def pullbackAssocSymmIsPullback [HasPullback f₁ (g₃ ≫ f₂)] :
    IsLimit
      (PullbackCone.mk l₁' (l₂' ≫ g₄)
        (show l₁' ≫ g₂ ≫ f₃ = (l₂' ≫ g₄) ≫ f₄ by
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            X₁ X₂ X₃ Y₁ Y₂ : C
            f₁ : Quiver.Hom X₁ Y₁
            f₂ : Quiver.Hom X₂ Y₁
            f₃ : Quiver.Hom X₂ Y₂
            f₄ : Quiver.Hom X₃ Y₂
            inst✝² : CategoryTheory.Limits.HasPullback f₁ f₂
            inst✝¹ : CategoryTheory.Limits.HasPullback f₃ f₄
            inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
          -/
          rw [pullback.lift_snd_assoc, Category.assoc, Category.assoc, pullback.condition])) := by
          /-
            🎉 no goals
          -/
  simpa [PullbackCone.pasteHoriz] using pasteHorizIsPullback rfl
    (pullbackIsPullback f₃ f₄) (pullbackPullbackRightIsPullback _ _ _ _)


theorem hasPullback_assoc_symm [HasPullback f₁ (g₃ ≫ f₂)] : HasPullback (g₂ ≫ f₃) f₄ :=
  ⟨⟨⟨_, pullbackAssocSymmIsPullback f₁ f₂ f₃ f₄⟩⟩⟩


/-- The canonical isomorphism `(X₁ ×[Y₁] X₂) ×[Y₂] X₃ ≅ X₁ ×[Y₁] (X₂ ×[Y₂] X₃)`. -/
noncomputable def pullbackAssoc [HasPullback ((pullback.snd _ _ : Z₁ ⟶ X₂) ≫ f₃) f₄]
    [HasPullback f₁ ((pullback.fst _ _ : Z₂ ⟶ X₂) ≫ f₂)] :
    pullback (pullback.snd _ _ ≫ f₃ : pullback f₁ f₂ ⟶ _) f₄ ≅
      pullback f₁ (pullback.fst _ _ ≫ f₂ : pullback f₃ f₄ ⟶ _) :=
  (pullbackPullbackLeftIsPullback f₁ f₂ f₃ f₄).conePointUniqueUpToIso
    (pullbackPullbackRightIsPullback f₁ f₂ f₃ f₄)


@[reassoc (attr := simp)]
theorem pullbackAssoc_inv_fst_fst [HasPullback ((pullback.snd _ _ : Z₁ ⟶ X₂) ≫ f₃) f₄]
    [HasPullback f₁ ((pullback.fst _ _ : Z₂ ⟶ X₂) ≫ f₂)] :
    (pullbackAssoc f₁ f₂ f₃ f₄).inv ≫ pullback.fst _ _ ≫ pullback.fst _ _ = pullback.fst _ _ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ Y₂ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₁
    f₃ : Quiver.Hom X₂ Y₂
    f₄ : Quiver.Hom X₃ Y₂
    inst✝³ : CategoryTheory.Limits.HasPullback f₁ f₂
    inst✝² : CategoryTheory.Limits.HasPullback f₃ f₄
    inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
    inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackAssoc  …
  -/
  trans l₁' ≫ pullback.fst _ _
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Y₁ Y₂ : C
      f₁ : Quiver.Hom X₁ Y₁
      f₂ : Quiver.Hom X₂ Y₁
      f₃ : Quiver.Hom X₂ Y₂
      f₄ : Quiver.Hom X₃ Y₂
      inst✝³ : CategoryTheory.Limits.HasPullback f₁ f₂
      inst✝² : CategoryTheory.Limits.HasPullback f₃ f₄
      inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
      inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackAssoc  …
    -/
  · rw [← Category.assoc]
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Y₁ Y₂ : C
      f₁ : Quiver.Hom X₁ Y₁
      f₂ : Quiver.Hom X₂ Y₁
      f₃ : Quiver.Hom X₂ Y₂
      f₄ : Quiver.Hom X₃ Y₂
      inst✝³ : CategoryTheory.Limits.HasPullback f₁ f₂
      inst✝² : CategoryTheory.Limits.HasPullback f₃ f₄
      inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
      inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    congr 1
    /-
      case e_a
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Y₁ Y₂ : C
      f₁ : Quiver.Hom X₁ Y₁
      f₂ : Quiver.Hom X₂ Y₁
      f₃ : Quiver.Hom X₂ Y₂
      f₄ : Quiver.Hom X₃ Y₂
      inst✝³ : CategoryTheory.Limits.HasPullback f₁ f₂
      inst✝² : CategoryTheory.Limits.HasPullback f₃ f₄
      inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
      inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackAssoc  …
    -/
    exact IsLimit.conePointUniqueUpToIso_inv_comp _ _ WalkingCospan.left
    /-
      🎉 no goals
    -/
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Y₁ Y₂ : C
      f₁ : Quiver.Hom X₁ Y₁
      f₂ : Quiver.Hom X₂ Y₁
      f₃ : Quiver.Hom X₂ Y₂
      f₄ : Quiver.Hom X₃ Y₂
      inst✝³ : CategoryTheory.Limits.HasPullback f₁ f₂
      inst✝² : CategoryTheory.Limits.HasPullback f₃ f₄
      inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
      inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
    -/
  · exact pullback.lift_fst _ _ _
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
theorem pullbackAssoc_hom_fst [HasPullback ((pullback.snd _ _ : Z₁ ⟶ X₂) ≫ f₃) f₄]
    [HasPullback f₁ ((pullback.fst _ _ : Z₂ ⟶ X₂) ≫ f₂)] :
    (pullbackAssoc f₁ f₂ f₃ f₄).hom ≫ pullback.fst _ _ = pullback.fst _ _ ≫ pullback.fst _ _ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ Y₂ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₁
    f₃ : Quiver.Hom X₂ Y₂
    f₄ : Quiver.Hom X₃ Y₂
    inst✝³ : CategoryTheory.Limits.HasPullback f₁ f₂
    inst✝² : CategoryTheory.Limits.HasPullback f₃ f₄
    inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
    inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackAssoc  …
  -/
  rw [← Iso.eq_inv_comp, pullbackAssoc_inv_fst_fst]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem pullbackAssoc_hom_snd_fst [HasPullback ((pullback.snd _ _ : Z₁ ⟶ X₂) ≫ f₃) f₄]
    [HasPullback f₁ ((pullback.fst _ _ : Z₂ ⟶ X₂) ≫ f₂)] : (pullbackAssoc f₁ f₂ f₃ f₄).hom ≫
    pullback.snd _ _ ≫ pullback.fst _ _ = pullback.fst _ _ ≫ pullback.snd _ _ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ Y₂ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₁
    f₃ : Quiver.Hom X₂ Y₂
    f₄ : Quiver.Hom X₃ Y₂
    inst✝³ : CategoryTheory.Limits.HasPullback f₁ f₂
    inst✝² : CategoryTheory.Limits.HasPullback f₃ f₄
    inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
    inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackAssoc  …
  -/
  trans l₂ ≫ pullback.fst _ _
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Y₁ Y₂ : C
      f₁ : Quiver.Hom X₁ Y₁
      f₂ : Quiver.Hom X₂ Y₁
      f₃ : Quiver.Hom X₂ Y₂
      f₄ : Quiver.Hom X₃ Y₂
      inst✝³ : CategoryTheory.Limits.HasPullback f₁ f₂
      inst✝² : CategoryTheory.Limits.HasPullback f₃ f₄
      inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
      inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackAssoc  …
    -/
  · rw [← Category.assoc]
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Y₁ Y₂ : C
      f₁ : Quiver.Hom X₁ Y₁
      f₂ : Quiver.Hom X₂ Y₁
      f₃ : Quiver.Hom X₂ Y₂
      f₄ : Quiver.Hom X₃ Y₂
      inst✝³ : CategoryTheory.Limits.HasPullback f₁ f₂
      inst✝² : CategoryTheory.Limits.HasPullback f₃ f₄
      inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
      inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    congr 1
    /-
      case e_a
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Y₁ Y₂ : C
      f₁ : Quiver.Hom X₁ Y₁
      f₂ : Quiver.Hom X₂ Y₁
      f₃ : Quiver.Hom X₂ Y₂
      f₄ : Quiver.Hom X₃ Y₂
      inst✝³ : CategoryTheory.Limits.HasPullback f₁ f₂
      inst✝² : CategoryTheory.Limits.HasPullback f₃ f₄
      inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
      inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackAssoc  …
    -/
    exact IsLimit.conePointUniqueUpToIso_hom_comp _ _ WalkingCospan.right
    /-
      🎉 no goals
    -/
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Y₁ Y₂ : C
      f₁ : Quiver.Hom X₁ Y₁
      f₂ : Quiver.Hom X₂ Y₁
      f₃ : Quiver.Hom X₂ Y₂
      f₄ : Quiver.Hom X₃ Y₂
      inst✝³ : CategoryTheory.Limits.HasPullback f₁ f₂
      inst✝² : CategoryTheory.Limits.HasPullback f₃ f₄
      inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
      inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
    -/
  · exact pullback.lift_fst _ _ _
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
theorem pullbackAssoc_hom_snd_snd [HasPullback ((pullback.snd _ _ : Z₁ ⟶ X₂) ≫ f₃) f₄]
    [HasPullback f₁ ((pullback.fst _ _ : Z₂ ⟶ X₂) ≫ f₂)] :
    (pullbackAssoc f₁ f₂ f₃ f₄).hom ≫ pullback.snd _ _ ≫ pullback.snd _ _ = pullback.snd _ _ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ Y₂ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₁
    f₃ : Quiver.Hom X₂ Y₂
    f₄ : Quiver.Hom X₃ Y₂
    inst✝³ : CategoryTheory.Limits.HasPullback f₁ f₂
    inst✝² : CategoryTheory.Limits.HasPullback f₃ f₄
    inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
    inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackAssoc  …
  -/
  trans l₂ ≫ pullback.snd _ _
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Y₁ Y₂ : C
      f₁ : Quiver.Hom X₁ Y₁
      f₂ : Quiver.Hom X₂ Y₁
      f₃ : Quiver.Hom X₂ Y₂
      f₄ : Quiver.Hom X₃ Y₂
      inst✝³ : CategoryTheory.Limits.HasPullback f₁ f₂
      inst✝² : CategoryTheory.Limits.HasPullback f₃ f₄
      inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
      inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackAssoc  …
    -/
  · rw [← Category.assoc]
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Y₁ Y₂ : C
      f₁ : Quiver.Hom X₁ Y₁
      f₂ : Quiver.Hom X₂ Y₁
      f₃ : Quiver.Hom X₂ Y₂
      f₄ : Quiver.Hom X₃ Y₂
      inst✝³ : CategoryTheory.Limits.HasPullback f₁ f₂
      inst✝² : CategoryTheory.Limits.HasPullback f₃ f₄
      inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
      inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp ( …
    -/
    congr 1
    /-
      case e_a
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Y₁ Y₂ : C
      f₁ : Quiver.Hom X₁ Y₁
      f₂ : Quiver.Hom X₂ Y₁
      f₃ : Quiver.Hom X₂ Y₂
      f₄ : Quiver.Hom X₃ Y₂
      inst✝³ : CategoryTheory.Limits.HasPullback f₁ f₂
      inst✝² : CategoryTheory.Limits.HasPullback f₃ f₄
      inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
      inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackAssoc  …
    -/
    exact IsLimit.conePointUniqueUpToIso_hom_comp _ _ WalkingCospan.right
    /-
      🎉 no goals
    -/
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Y₁ Y₂ : C
      f₁ : Quiver.Hom X₁ Y₁
      f₂ : Quiver.Hom X₂ Y₁
      f₃ : Quiver.Hom X₂ Y₂
      f₄ : Quiver.Hom X₃ Y₂
      inst✝³ : CategoryTheory.Limits.HasPullback f₁ f₂
      inst✝² : CategoryTheory.Limits.HasPullback f₃ f₄
      inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
      inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullback.lift  …
    -/
  · exact pullback.lift_snd _ _ _
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
theorem pullbackAssoc_inv_fst_snd [HasPullback ((pullback.snd _ _ : Z₁ ⟶ X₂) ≫ f₃) f₄]
    [HasPullback f₁ ((pullback.fst _ _ : Z₂ ⟶ X₂) ≫ f₂)] :
    (pullbackAssoc f₁ f₂ f₃ f₄).inv ≫ pullback.fst _ _ ≫ pullback.snd _ _ =
                                              /-
                                                C : Type u
                                                inst✝⁴ : CategoryTheory.Category.{v, u} C
                                                X₁ X₂ X₃ Y₁ Y₂ : C
                                                f₁ : Quiver.Hom X₁ Y₁
                                                f₂ : Quiver.Hom X₂ Y₁
                                                f₃ : Quiver.Hom X₂ Y₂
                                                f₄ : Quiver.Hom X₃ Y₂
                                                inst✝³ : CategoryTheory.Limits.HasPullback f₁ f₂
                                                inst✝² : CategoryTheory.Limits.HasPullback f₃ f₄
                                                inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
                                                inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
                                                ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackAssoc  …
                                              -/
    pullback.snd _ _ ≫ pullback.fst _ _ := by rw [Iso.inv_comp_eq, pullbackAssoc_hom_snd_fst]
                                              /-
                                                🎉 no goals
                                              -/


@[reassoc (attr := simp)]
theorem pullbackAssoc_inv_snd [HasPullback ((pullback.snd _ _ : Z₁ ⟶ X₂) ≫ f₃) f₄]
    [HasPullback f₁ ((pullback.fst _ _ : Z₂ ⟶ X₂) ≫ f₂)] :
    (pullbackAssoc f₁ f₂ f₃ f₄).inv ≫ pullback.snd _ _ = pullback.snd _ _ ≫ pullback.snd _ _ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Y₁ Y₂ : C
    f₁ : Quiver.Hom X₁ Y₁
    f₂ : Quiver.Hom X₂ Y₁
    f₃ : Quiver.Hom X₂ Y₂
    f₄ : Quiver.Hom X₃ Y₂
    inst✝³ : CategoryTheory.Limits.HasPullback f₁ f₂
    inst✝² : CategoryTheory.Limits.HasPullback f₃ f₄
    inst✝¹ : CategoryTheory.Limits.HasPullback (CategoryTheory.CategoryStruct.comp …
    inst✝ : CategoryTheory.Limits.HasPullback f₁ (CategoryTheory.CategoryStruct.co …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pullbackAssoc  …
  -/
  rw [Iso.inv_comp_eq, pullbackAssoc_hom_snd_snd]
  /-
    🎉 no goals
  -/


local notation "Y₁" => pushout g₁ g₂


local notation "Y₂" => pushout g₃ g₄


local notation "f₁" => (pushout.inl _ _ : X₁ ⟶ Y₁)


local notation "f₂" => (pushout.inr _ _ : X₂ ⟶ Y₁)


local notation "f₃" => (pushout.inl _ _ : X₂ ⟶ Y₂)


local notation "f₄" => (pushout.inr _ _ : X₃ ⟶ Y₂)


local notation "W" => pushout g₁ (g₂ ≫ f₃)


local notation "W'" => pushout (g₃ ≫ f₂) g₄


local notation "l₁" =>
  (pushout.desc (pushout.inl _ _) (f₃ ≫ pushout.inr _ _)
    (pushout.condition.trans (Category.assoc _ _ _)) : Y₁ ⟶ W)


local notation "l₂" => (pushout.inr _ _ : Y₂ ⟶ W)


local notation "l₁'" => (pushout.inl _ _ : Y₁ ⟶ W')


local notation "l₂'" =>
  (pushout.desc (f₂ ≫ pushout.inl _ _) (pushout.inr _ _)
      (Eq.trans (Eq.symm (Category.assoc _ _ _)) pushout.condition) :
    Y₂ ⟶ W')


/-- `(X₁ ⨿[Z₁] X₂) ⨿[Z₂] X₃` is the pushout `(X₁ ⨿[Z₁] X₂) ×[X₂] (X₂ ⨿[Z₂] X₃)`. -/
def pushoutPushoutLeftIsPushout [HasPushout (g₃ ≫ f₂) g₄] :
    IsColimit
      (PushoutCocone.mk l₁' l₂' (show f₂ ≫ l₁' = f₃ ≫ l₂' from (pushout.inl_desc _ _ _).symm)) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Z₁ Z₂ : C
    g₁ : Quiver.Hom Z₁ X₁
    g₂ : Quiver.Hom Z₁ X₂
    g₃ : Quiver.Hom Z₂ X₂
    g₄ : Quiver.Hom Z₂ X₃
    inst✝² : CategoryTheory.Limits.HasPushout g₁ g₂
    inst✝¹ : CategoryTheory.Limits.HasPushout g₃ g₄
    inst✝ : CategoryTheory.Limits.HasPushout (CategoryTheory.CategoryStruct.comp g …
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk (Cat …
  -/
  apply botSquareIsPushout _ rfl (pushoutIsPushout _ g₄)
  simpa [PushoutCocone.pasteVert] using
    PushoutCocone.mkSelfIsColimit (pushoutIsPushout (g₃ ≫ f₂) g₄)


/-- `(X₁ ⨿[Z₁] X₂) ⨿[Z₂] X₃` is the pushout `X₁ ⨿[Z₁] (X₂ ⨿[Z₂] X₃)`. -/
def pushoutAssocIsPushout [HasPushout (g₃ ≫ f₂) g₄] :
    IsColimit
      (PushoutCocone.mk (f₁ ≫ l₁') l₂'
        (show g₁ ≫ f₁ ≫ l₁' = (g₂ ≫ f₃) ≫ l₂' by
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            X₁ X₂ X₃ Z₁ Z₂ : C
            g₁ : Quiver.Hom Z₁ X₁
            g₂ : Quiver.Hom Z₁ X₂
            g₃ : Quiver.Hom Z₂ X₂
            g₄ : Quiver.Hom Z₂ X₃
            inst✝² : CategoryTheory.Limits.HasPushout g₁ g₂
            inst✝¹ : CategoryTheory.Limits.HasPushout g₃ g₄
            inst✝ : CategoryTheory.Limits.HasPushout (CategoryTheory.CategoryStruct.comp g …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp g₁ (CategoryTheory.CategoryStruct.com …
          -/
          rw [Category.assoc, pushout.inl_desc, pushout.condition_assoc])) := by
          /-
            🎉 no goals
          -/
  simpa using pasteHorizIsPushout rfl (pushoutIsPushout g₁ g₂)
    (pushoutPushoutLeftIsPushout g₁ g₂ g₃ g₄)


theorem hasPushout_assoc [HasPushout (g₃ ≫ f₂) g₄] : HasPushout g₁ (g₂ ≫ f₃) :=
  ⟨⟨⟨_, pushoutAssocIsPushout g₁ g₂ g₃ g₄⟩⟩⟩


/-- `X₁ ⨿[Z₁] (X₂ ⨿[Z₂] X₃)` is the pushout `(X₁ ⨿[Z₁] X₂) ×[X₂] (X₂ ⨿[Z₂] X₃)`. -/
def pushoutPushoutRightIsPushout [HasPushout g₁ (g₂ ≫ f₃)] :
    IsColimit (PushoutCocone.mk l₁ l₂ (show f₂ ≫ l₁ = f₃ ≫ l₂ from pushout.inr_desc _ _ _)) := by
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Z₁ Z₂ : C
    g₁ : Quiver.Hom Z₁ X₁
    g₂ : Quiver.Hom Z₁ X₂
    g₃ : Quiver.Hom Z₂ X₂
    g₄ : Quiver.Hom Z₂ X₃
    inst✝² : CategoryTheory.Limits.HasPushout g₁ g₂
    inst✝¹ : CategoryTheory.Limits.HasPushout g₃ g₄
    inst✝ : CategoryTheory.Limits.HasPushout g₁ (CategoryTheory.CategoryStruct.com …
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk (Cat …
  -/
  apply rightSquareIsPushout _ rfl (pushoutIsPushout _ _)
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Z₁ Z₂ : C
    g₁ : Quiver.Hom Z₁ X₁
    g₂ : Quiver.Hom Z₁ X₂
    g₃ : Quiver.Hom Z₂ X₂
    g₄ : Quiver.Hom Z₂ X₃
    inst✝² : CategoryTheory.Limits.HasPushout g₁ g₂
    inst✝¹ : CategoryTheory.Limits.HasPushout g₃ g₄
    inst✝ : CategoryTheory.Limits.HasPushout g₁ (CategoryTheory.CategoryStruct.com …
    ⊢ CategoryTheory.Limits.IsColimit ((CategoryTheory.Limits.PushoutCocone.mk (Ca …
  -/
  simpa [PushoutCocone.pasteHoriz] using PushoutCocone.mkSelfIsColimit (pushoutIsPushout _ _)
  /-
    🎉 no goals
  -/


/-- `X₁ ⨿[Z₁] (X₂ ⨿[Z₂] X₃)` is the pushout `(X₁ ⨿[Z₁] X₂) ⨿[Z₂] X₃`. -/
def pushoutAssocSymmIsPushout [HasPushout g₁ (g₂ ≫ f₃)] :
    IsColimit
      (PushoutCocone.mk l₁ (f₄ ≫ l₂)
        (show (g₃ ≫ f₂) ≫ l₁ = g₄ ≫ f₄ ≫ l₂ by
          /-
            C : Type u
            inst✝³ : CategoryTheory.Category.{v, u} C
            X₁ X₂ X₃ Z₁ Z₂ : C
            g₁ : Quiver.Hom Z₁ X₁
            g₂ : Quiver.Hom Z₁ X₂
            g₃ : Quiver.Hom Z₂ X₂
            g₄ : Quiver.Hom Z₂ X₃
            inst✝² : CategoryTheory.Limits.HasPushout g₁ g₂
            inst✝¹ : CategoryTheory.Limits.HasPushout g₃ g₄
            inst✝ : CategoryTheory.Limits.HasPushout g₁ (CategoryTheory.CategoryStruct.com …
            ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.CategoryStruct.comp g …
          -/
          rw [Category.assoc, pushout.inr_desc, pushout.condition_assoc])) := by
          /-
            🎉 no goals
          -/
  /-
    C : Type u
    inst✝³ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Z₁ Z₂ : C
    g₁ : Quiver.Hom Z₁ X₁
    g₂ : Quiver.Hom Z₁ X₂
    g₃ : Quiver.Hom Z₂ X₂
    g₄ : Quiver.Hom Z₂ X₃
    inst✝² : CategoryTheory.Limits.HasPushout g₁ g₂
    inst✝¹ : CategoryTheory.Limits.HasPushout g₃ g₄
    inst✝ : CategoryTheory.Limits.HasPushout g₁ (CategoryTheory.CategoryStruct.com …
    ⊢ CategoryTheory.Limits.IsColimit (CategoryTheory.Limits.PushoutCocone.mk (Cat …
  -/
  simpa using pasteVertIsPushout rfl (pushoutIsPushout _ _) (pushoutPushoutRightIsPushout _ _ _ _)
  /-
    🎉 no goals
  -/


theorem hasPushout_assoc_symm [HasPushout g₁ (g₂ ≫ f₃)] : HasPushout (g₃ ≫ f₂) g₄ :=
  ⟨⟨⟨_, pushoutAssocSymmIsPushout g₁ g₂ g₃ g₄⟩⟩⟩

-- Porting note: these are not propagating so moved into statements
-- variable [HasPushout (g₃ ≫ f₂) g₄] [HasPushout g₁ (g₂ ≫ f₃)]


/-- The canonical isomorphism `(X₁ ⨿[Z₁] X₂) ⨿[Z₂] X₃ ≅ X₁ ⨿[Z₁] (X₂ ⨿[Z₂] X₃)`. -/
noncomputable def pushoutAssoc [HasPushout (g₃ ≫ (pushout.inr _ _ : X₂ ⟶ Y₁)) g₄]
    [HasPushout g₁ (g₂ ≫ (pushout.inl _ _ : X₂ ⟶ Y₂))] :
    pushout (g₃ ≫ pushout.inr _ _ : _ ⟶ pushout g₁ g₂) g₄ ≅
      pushout g₁ (g₂ ≫ pushout.inl _ _ : _ ⟶ pushout g₃ g₄) :=
  (pushoutPushoutLeftIsPushout g₁ g₂ g₃ g₄).coconePointUniqueUpToIso
    (pushoutPushoutRightIsPushout g₁ g₂ g₃ g₄)


@[reassoc (attr := simp)]
theorem inl_inl_pushoutAssoc_hom [HasPushout (g₃ ≫ (pushout.inr _ _ : X₂ ⟶ Y₁)) g₄]
    [HasPushout g₁ (g₂ ≫ (pushout.inl _ _ : X₂ ⟶ Y₂))] :
    pushout.inl _ _ ≫ pushout.inl _ _ ≫ (pushoutAssoc g₁ g₂ g₃ g₄).hom = pushout.inl _ _ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Z₁ Z₂ : C
    g₁ : Quiver.Hom Z₁ X₁
    g₂ : Quiver.Hom Z₁ X₂
    g₃ : Quiver.Hom Z₂ X₂
    g₄ : Quiver.Hom Z₂ X₃
    inst✝³ : CategoryTheory.Limits.HasPushout g₁ g₂
    inst✝² : CategoryTheory.Limits.HasPushout g₃ g₄
    inst✝¹ : CategoryTheory.Limits.HasPushout (CategoryTheory.CategoryStruct.comp  …
    inst✝ : CategoryTheory.Limits.HasPushout g₁ (CategoryTheory.CategoryStruct.com …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl g₁ …
  -/
  trans f₁ ≫ l₁
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Z₁ Z₂ : C
      g₁ : Quiver.Hom Z₁ X₁
      g₂ : Quiver.Hom Z₁ X₂
      g₃ : Quiver.Hom Z₂ X₂
      g₄ : Quiver.Hom Z₂ X₃
      inst✝³ : CategoryTheory.Limits.HasPushout g₁ g₂
      inst✝² : CategoryTheory.Limits.HasPushout g₃ g₄
      inst✝¹ : CategoryTheory.Limits.HasPushout (CategoryTheory.CategoryStruct.comp  …
      inst✝ : CategoryTheory.Limits.HasPushout g₁ (CategoryTheory.CategoryStruct.com …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl g₁ …
    -/
  · congr 1
    exact
      (pushoutPushoutLeftIsPushout g₁ g₂ g₃ g₄).comp_coconePointUniqueUpToIso_hom _
        WalkingCospan.left
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Z₁ Z₂ : C
      g₁ : Quiver.Hom Z₁ X₁
      g₂ : Quiver.Hom Z₁ X₂
      g₃ : Quiver.Hom Z₂ X₂
      g₄ : Quiver.Hom Z₂ X₃
      inst✝³ : CategoryTheory.Limits.HasPushout g₁ g₂
      inst✝² : CategoryTheory.Limits.HasPushout g₃ g₄
      inst✝¹ : CategoryTheory.Limits.HasPushout (CategoryTheory.CategoryStruct.comp  …
      inst✝ : CategoryTheory.Limits.HasPushout g₁ (CategoryTheory.CategoryStruct.com …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl g₁ …
    -/
  · exact pushout.inl_desc _ _ _
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
theorem inr_inl_pushoutAssoc_hom [HasPushout (g₃ ≫ (pushout.inr _ _ : X₂ ⟶ Y₁)) g₄]
    [HasPushout g₁ (g₂ ≫ (pushout.inl _ _ : X₂ ⟶ Y₂))] :
    pushout.inr _ _ ≫ pushout.inl _ _ ≫ (pushoutAssoc g₁ g₂ g₃ g₄).hom =
      pushout.inl _ _ ≫ pushout.inr _ _ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Z₁ Z₂ : C
    g₁ : Quiver.Hom Z₁ X₁
    g₂ : Quiver.Hom Z₁ X₂
    g₃ : Quiver.Hom Z₂ X₂
    g₄ : Quiver.Hom Z₂ X₃
    inst✝³ : CategoryTheory.Limits.HasPushout g₁ g₂
    inst✝² : CategoryTheory.Limits.HasPushout g₃ g₄
    inst✝¹ : CategoryTheory.Limits.HasPushout (CategoryTheory.CategoryStruct.comp  …
    inst✝ : CategoryTheory.Limits.HasPushout g₁ (CategoryTheory.CategoryStruct.com …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr g₁ …
  -/
  trans f₂ ≫ l₁
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Z₁ Z₂ : C
      g₁ : Quiver.Hom Z₁ X₁
      g₂ : Quiver.Hom Z₁ X₂
      g₃ : Quiver.Hom Z₂ X₂
      g₄ : Quiver.Hom Z₂ X₃
      inst✝³ : CategoryTheory.Limits.HasPushout g₁ g₂
      inst✝² : CategoryTheory.Limits.HasPushout g₃ g₄
      inst✝¹ : CategoryTheory.Limits.HasPushout (CategoryTheory.CategoryStruct.comp  …
      inst✝ : CategoryTheory.Limits.HasPushout g₁ (CategoryTheory.CategoryStruct.com …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr g₁ …
    -/
  · congr 1
    exact
      (pushoutPushoutLeftIsPushout g₁ g₂ g₃ g₄).comp_coconePointUniqueUpToIso_hom _
        WalkingCospan.left
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Z₁ Z₂ : C
      g₁ : Quiver.Hom Z₁ X₁
      g₂ : Quiver.Hom Z₁ X₂
      g₃ : Quiver.Hom Z₂ X₂
      g₄ : Quiver.Hom Z₂ X₃
      inst✝³ : CategoryTheory.Limits.HasPushout g₁ g₂
      inst✝² : CategoryTheory.Limits.HasPushout g₃ g₄
      inst✝¹ : CategoryTheory.Limits.HasPushout (CategoryTheory.CategoryStruct.comp  …
      inst✝ : CategoryTheory.Limits.HasPushout g₁ (CategoryTheory.CategoryStruct.com …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr g₁ …
    -/
  · exact pushout.inr_desc _ _ _
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
theorem inr_inr_pushoutAssoc_inv [HasPushout (g₃ ≫ (pushout.inr _ _ : X₂ ⟶ Y₁)) g₄]
    [HasPushout g₁ (g₂ ≫ (pushout.inl _ _ : X₂ ⟶ Y₂))] :
    pushout.inr _ _ ≫ pushout.inr _ _ ≫ (pushoutAssoc g₁ g₂ g₃ g₄).inv = pushout.inr _ _ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Z₁ Z₂ : C
    g₁ : Quiver.Hom Z₁ X₁
    g₂ : Quiver.Hom Z₁ X₂
    g₃ : Quiver.Hom Z₂ X₂
    g₄ : Quiver.Hom Z₂ X₃
    inst✝³ : CategoryTheory.Limits.HasPushout g₁ g₂
    inst✝² : CategoryTheory.Limits.HasPushout g₃ g₄
    inst✝¹ : CategoryTheory.Limits.HasPushout (CategoryTheory.CategoryStruct.comp  …
    inst✝ : CategoryTheory.Limits.HasPushout g₁ (CategoryTheory.CategoryStruct.com …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr g₃ …
  -/
  trans f₄ ≫ l₂'
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Z₁ Z₂ : C
      g₁ : Quiver.Hom Z₁ X₁
      g₂ : Quiver.Hom Z₁ X₂
      g₃ : Quiver.Hom Z₂ X₂
      g₄ : Quiver.Hom Z₂ X₃
      inst✝³ : CategoryTheory.Limits.HasPushout g₁ g₂
      inst✝² : CategoryTheory.Limits.HasPushout g₃ g₄
      inst✝¹ : CategoryTheory.Limits.HasPushout (CategoryTheory.CategoryStruct.comp  …
      inst✝ : CategoryTheory.Limits.HasPushout g₁ (CategoryTheory.CategoryStruct.com …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr g₃ …
    -/
  · congr 1
    exact
      (pushoutPushoutLeftIsPushout g₁ g₂ g₃ g₄).comp_coconePointUniqueUpToIso_inv
        (pushoutPushoutRightIsPushout g₁ g₂ g₃ g₄) WalkingCospan.right
    /-
      C : Type u
      inst✝⁴ : CategoryTheory.Category.{v, u} C
      X₁ X₂ X₃ Z₁ Z₂ : C
      g₁ : Quiver.Hom Z₁ X₁
      g₂ : Quiver.Hom Z₁ X₂
      g₃ : Quiver.Hom Z₂ X₂
      g₄ : Quiver.Hom Z₂ X₃
      inst✝³ : CategoryTheory.Limits.HasPushout g₁ g₂
      inst✝² : CategoryTheory.Limits.HasPushout g₃ g₄
      inst✝¹ : CategoryTheory.Limits.HasPushout (CategoryTheory.CategoryStruct.comp  …
      inst✝ : CategoryTheory.Limits.HasPushout g₁ (CategoryTheory.CategoryStruct.com …
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr g₃ …
    -/
  · exact pushout.inr_desc _ _ _
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
theorem inl_pushoutAssoc_inv [HasPushout (g₃ ≫ (pushout.inr _ _ : X₂ ⟶ Y₁)) g₄]
    [HasPushout g₁ (g₂ ≫ (pushout.inl _ _ : X₂ ⟶ Y₂))] :
    pushout.inl _ _ ≫ (pushoutAssoc g₁ g₂ g₃ g₄).inv = pushout.inl _ _ ≫ pushout.inl _ _ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Z₁ Z₂ : C
    g₁ : Quiver.Hom Z₁ X₁
    g₂ : Quiver.Hom Z₁ X₂
    g₃ : Quiver.Hom Z₂ X₂
    g₄ : Quiver.Hom Z₂ X₃
    inst✝³ : CategoryTheory.Limits.HasPushout g₁ g₂
    inst✝² : CategoryTheory.Limits.HasPushout g₃ g₄
    inst✝¹ : CategoryTheory.Limits.HasPushout (CategoryTheory.CategoryStruct.comp  …
    inst✝ : CategoryTheory.Limits.HasPushout g₁ (CategoryTheory.CategoryStruct.com …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl g₁ …
  -/
  rw [Iso.comp_inv_eq, Category.assoc, inl_inl_pushoutAssoc_hom]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem inl_inr_pushoutAssoc_inv [HasPushout (g₃ ≫ (pushout.inr _ _ : X₂ ⟶ Y₁)) g₄]
    [HasPushout g₁ (g₂ ≫ (pushout.inl _ _ : X₂ ⟶ Y₂))] :
    pushout.inl _ _ ≫ pushout.inr _ _ ≫ (pushoutAssoc g₁ g₂ g₃ g₄).inv =
      pushout.inr _ _ ≫ pushout.inl _ _ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Z₁ Z₂ : C
    g₁ : Quiver.Hom Z₁ X₁
    g₂ : Quiver.Hom Z₁ X₂
    g₃ : Quiver.Hom Z₂ X₂
    g₄ : Quiver.Hom Z₂ X₃
    inst✝³ : CategoryTheory.Limits.HasPushout g₁ g₂
    inst✝² : CategoryTheory.Limits.HasPushout g₃ g₄
    inst✝¹ : CategoryTheory.Limits.HasPushout (CategoryTheory.CategoryStruct.comp  …
    inst✝ : CategoryTheory.Limits.HasPushout g₁ (CategoryTheory.CategoryStruct.com …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inl g₃ …
  -/
  rw [← Category.assoc, Iso.comp_inv_eq, Category.assoc, inr_inl_pushoutAssoc_hom]
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem inr_pushoutAssoc_hom [HasPushout (g₃ ≫ (pushout.inr _ _ : X₂ ⟶ Y₁)) g₄]
    [HasPushout g₁ (g₂ ≫ (pushout.inl _ _ : X₂ ⟶ Y₂))] :
    pushout.inr _ _ ≫ (pushoutAssoc g₁ g₂ g₃ g₄).hom = pushout.inr _ _ ≫ pushout.inr _ _ := by
  /-
    C : Type u
    inst✝⁴ : CategoryTheory.Category.{v, u} C
    X₁ X₂ X₃ Z₁ Z₂ : C
    g₁ : Quiver.Hom Z₁ X₁
    g₂ : Quiver.Hom Z₁ X₂
    g₃ : Quiver.Hom Z₂ X₂
    g₄ : Quiver.Hom Z₂ X₃
    inst✝³ : CategoryTheory.Limits.HasPushout g₁ g₂
    inst✝² : CategoryTheory.Limits.HasPushout g₃ g₄
    inst✝¹ : CategoryTheory.Limits.HasPushout (CategoryTheory.CategoryStruct.comp  …
    inst✝ : CategoryTheory.Limits.HasPushout g₁ (CategoryTheory.CategoryStruct.com …
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (CategoryTheory.Limits.pushout.inr (C …
  -/
  rw [← Iso.eq_comp_inv, Category.assoc, inr_inr_pushoutAssoc_inv]
  /-
    🎉 no goals
  -/


