/-- A simplicial set `S` is a *quasicategory* if it satisfies the following horn-filling condition:
for every `n : ℕ` and `0 < i < n`,
every map of simplicial sets `σ₀ : Λ[n, i] → S` can be extended to a map `σ : Δ[n] → S`.

[Kerodon, 003A] -/
class Quasicategory (S : SSet) : Prop where
  hornFilling' : ∀ ⦃n : ℕ⦄ ⦃i : Fin (n+3)⦄ (σ₀ : Λ[n+2, i] ⟶ S)
    (_h0 : 0 < i) (_hn : i < Fin.last (n+2)),
      ∃ σ : Δ[n+2] ⟶ S, σ₀ = hornInclusion (n+2) i ≫ σ


lemma Quasicategory.hornFilling {S : SSet} [Quasicategory S] ⦃n : ℕ⦄ ⦃i : Fin (n+1)⦄
    (h0 : 0 < i) (hn : i < Fin.last n)
    (σ₀ : Λ[n, i] ⟶ S) : ∃ σ : Δ[n] ⟶ S, σ₀ = hornInclusion n i ≫ σ := by
  cases n using Nat.casesAuxOn with
  | zero => simp [Fin.lt_iff_val_lt_val] at hn
  | succ n =>
  cases n using Nat.casesAuxOn with
  | zero =>
    simp only [Fin.lt_iff_val_lt_val, Fin.val_zero, Fin.val_last, zero_add, Nat.lt_one_iff] at h0 hn
    simp [hn] at h0
  | succ n => exact Quasicategory.hornFilling' σ₀ h0 hn


/-- Every Kan complex is a quasicategory.

[Kerodon, 003C] -/
instance (S : SSet) [KanComplex S] : Quasicategory S where
  hornFilling' _ _ σ₀ _ _ := KanComplex.hornFilling σ₀


lemma quasicategory_of_filler (S : SSet)
    (filler : ∀ ⦃n : ℕ⦄ ⦃i : Fin (n+3)⦄ (σ₀ : Λ[n+2, i] ⟶ S)
      (_h0 : 0 < i) (_hn : i < Fin.last (n+2)),
      ∃ σ : S _[n+2], ∀ (j) (h : j ≠ i), S.δ j σ = σ₀.app _ (horn.face i j h)) :
    Quasicategory S where
  hornFilling' n i σ₀ h₀ hₙ := by
    /-
      S : SSet
      filler : ∀ ⦃n : Nat⦄ ⦃i : Fin (HAdd.hAdd n 3)⦄ (σ₀ : Quiver.Hom (SSet.horn (HA …
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) S
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      ⊢ Exists fun σ => Eq σ₀ (CategoryTheory.CategoryStruct.comp (SSet.hornInclusio …
    -/
    obtain ⟨σ, h⟩ := filler σ₀ h₀ hₙ
    /-
      case intro
      S : SSet
      filler : ∀ ⦃n : Nat⦄ ⦃i : Fin (HAdd.hAdd n 3)⦄ (σ₀ : Quiver.Hom (SSet.horn (HA …
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) S
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      σ : S.obj { unop := SimplexCategory.mk (HAdd.hAdd n 2) }
      h : ∀ (j : Fin (HAdd.hAdd n 3)) (h : Ne j i), Eq (CategoryTheory.SimplicialObj …
      ⊢ Exists fun σ => Eq σ₀ (CategoryTheory.CategoryStruct.comp (SSet.hornInclusio …
    -/
    refine ⟨(S.yonedaEquiv _).symm σ, ?_⟩
    /-
      case intro
      S : SSet
      filler : ∀ ⦃n : Nat⦄ ⦃i : Fin (HAdd.hAdd n 3)⦄ (σ₀ : Quiver.Hom (SSet.horn (HA …
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) S
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      σ : S.obj { unop := SimplexCategory.mk (HAdd.hAdd n 2) }
      h : ∀ (j : Fin (HAdd.hAdd n 3)) (h : Ne j i), Eq (CategoryTheory.SimplicialObj …
      ⊢ Eq σ₀ (CategoryTheory.CategoryStruct.comp (SSet.hornInclusion (HAdd.hAdd n 2 …
    -/
    apply horn.hom_ext
    /-
      case intro.h
      S : SSet
      filler : ∀ ⦃n : Nat⦄ ⦃i : Fin (HAdd.hAdd n 3)⦄ (σ₀ : Quiver.Hom (SSet.horn (HA …
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) S
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      σ : S.obj { unop := SimplexCategory.mk (HAdd.hAdd n 2) }
      h : ∀ (j : Fin (HAdd.hAdd n 3)) (h : Ne j i), Eq (CategoryTheory.SimplicialObj …
      ⊢ ∀ (j : Fin (HAdd.hAdd (HAdd.hAdd n 1) 2)) (h : Ne j i), Eq (σ₀.app { unop := …
    -/
    intro j hj
    /-
      case intro.h
      S : SSet
      filler : ∀ ⦃n : Nat⦄ ⦃i : Fin (HAdd.hAdd n 3)⦄ (σ₀ : Quiver.Hom (SSet.horn (HA …
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) S
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      σ : S.obj { unop := SimplexCategory.mk (HAdd.hAdd n 2) }
      h : ∀ (j : Fin (HAdd.hAdd n 3)) (h : Ne j i), Eq (CategoryTheory.SimplicialObj …
      j : Fin (HAdd.hAdd (HAdd.hAdd n 1) 2)
      hj : Ne j i
      ⊢ Eq (σ₀.app { unop := SimplexCategory.mk (HAdd.hAdd n 1) } (SSet.horn.face i  …
    -/
    rw [← h j hj, NatTrans.comp_app]
    /-
      case intro.h
      S : SSet
      filler : ∀ ⦃n : Nat⦄ ⦃i : Fin (HAdd.hAdd n 3)⦄ (σ₀ : Quiver.Hom (SSet.horn (HA …
      n : Nat
      i : Fin (HAdd.hAdd n 3)
      σ₀ : Quiver.Hom (SSet.horn (HAdd.hAdd n 2) i) S
      h₀ : LT.lt 0 i
      hₙ : LT.lt i (Fin.last (HAdd.hAdd n 2))
      σ : S.obj { unop := SimplexCategory.mk (HAdd.hAdd n 2) }
      h : ∀ (j : Fin (HAdd.hAdd n 3)) (h : Ne j i), Eq (CategoryTheory.SimplicialObj …
      j : Fin (HAdd.hAdd (HAdd.hAdd n 1) 2)
      hj : Ne j i
      ⊢ Eq (CategoryTheory.SimplicialObject.δ S j σ) (CategoryTheory.CategoryStruct. …
    -/
    rfl
    /-
      🎉 no goals
    -/


