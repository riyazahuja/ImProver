variable (M) in
/-- Predicate for a set `A` to be covered by at most `K` cosets of another set `B` under the action
by the monoid `M`. -/
@[to_additive "Predicate for a set `A` to be covered by at most `K` cosets of another set `B` under
the action by the monoid `M`."]
def CovBySMul (K : ℝ) (A B : Set X) : Prop := ∃ F : Finset M, #F ≤ K ∧ A ⊆ (F : Set M) • B


@[to_additive (attr := simp, refl)]
                                                  /-
                                                    M : Type u_1
                                                    X : Type u_3
                                                    inst✝¹ : Monoid M
                                                    inst✝ : MulAction M X
                                                    A : Set X
                                                    ⊢ And (LE.le (↑(Finset.card 1)) 1) (HasSubset.Subset A (HSMul.hSMul (↑1) A))
                                                  -/
lemma CovBySMul.rfl : CovBySMul M 1 A A := ⟨1, by simp⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


@[to_additive (attr := simp)]
                                                                      /-
                                                                        M : Type u_1
                                                                        X : Type u_3
                                                                        inst✝¹ : Monoid M
                                                                        inst✝ : MulAction M X
                                                                        A B : Set X
                                                                        hAB : HasSubset.Subset A B
                                                                        ⊢ And (LE.le (↑(Finset.card 1)) 1) (HasSubset.Subset A (HSMul.hSMul (↑1) B))
                                                                      -/
lemma CovBySMul.of_subset (hAB : A ⊆ B) : CovBySMul M 1 A B := ⟨1, by simpa⟩
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[to_additive] lemma CovBySMul.nonneg : CovBySMul M K A B → 0 ≤ K := by
  /-
    M : Type u_1
    X : Type u_3
    inst✝¹ : Monoid M
    inst✝ : MulAction M X
    K : Real
    A B : Set X
    ⊢ CovBySMul M K A B → LE.le 0 K
  -/
  rintro ⟨F, hF, -⟩; exact (#F).cast_nonneg.trans hF
                     /-
                       🎉 no goals
                     -/


@[to_additive (attr := simp)]
                                                       /-
                                                         M : Type u_1
                                                         X : Type u_3
                                                         inst✝¹ : Monoid M
                                                         inst✝ : MulAction M X
                                                         A B : Set X
                                                         ⊢ Iff (CovBySMul M 0 A B) (Eq A EmptyCollection.emptyCollection)
                                                       -/
lemma covBySMul_zero : CovBySMul M 0 A B ↔ A = ∅ := by simp [CovBySMul]
                                                       /-
                                                         🎉 no goals
                                                       -/


@[to_additive]
lemma CovBySMul.mono (hKL : K ≤ L) : CovBySMul M K A B → CovBySMul M L A B := by
  /-
    M : Type u_1
    X : Type u_3
    inst✝¹ : Monoid M
    inst✝ : MulAction M X
    K L : Real
    A B : Set X
    hKL : LE.le K L
    ⊢ CovBySMul M K A B → CovBySMul M L A B
  -/
  rintro ⟨F, hF, hFAB⟩; exact ⟨F, hF.trans hKL, hFAB⟩
                        /-
                          🎉 no goals
                        -/


@[to_additive] lemma CovBySMul.trans [MulAction M N] [IsScalarTower M N X]
    (hAB : CovBySMul M K A B) (hBC : CovBySMul N L B C) : CovBySMul N (K * L) A C := by
  classical
  have := hAB.nonneg
  obtain ⟨F₁, hF₁, hFAB⟩ := hAB
  obtain ⟨F₂, hF₂, hFBC⟩ := hBC
  refine ⟨F₁ • F₂, ?_, ?_⟩
  · calc
      (#(F₁ • F₂) : ℝ) ≤ #F₁ * #F₂ := mod_cast Finset.card_smul_le
      _ ≤ K * L := by gcongr
  · calc
      A ⊆ (F₁ : Set M) • B := hFAB
      _ ⊆ (F₁ : Set M) • (F₂ : Set N) • C := by gcongr
      _ = (↑(F₁ • F₂) : Set N) • C := by simp


@[to_additive]
lemma CovBySMul.subset_left (hA : A₁ ⊆ A₂) (hAB : CovBySMul M K A₂ B) :
                             /-
                               M : Type u_1
                               X : Type u_3
                               inst✝¹ : Monoid M
                               inst✝ : MulAction M X
                               K : Real
                               A₁ A₂ B : Set X
                               hA : HasSubset.Subset A₁ A₂
                               hAB : CovBySMul M K A₂ B
                               ⊢ CovBySMul M K A₁ B
                             -/
    CovBySMul M K A₁ B := by simpa using (CovBySMul.of_subset (M := M) hA).trans hAB
                             /-
                               🎉 no goals
                             -/


@[to_additive]
lemma CovBySMul.subset_right (hB : B₁ ⊆ B₂) (hAB : CovBySMul M K A B₁) :
                             /-
                               M : Type u_1
                               X : Type u_3
                               inst✝¹ : Monoid M
                               inst✝ : MulAction M X
                               K : Real
                               A B₁ B₂ : Set X
                               hB : HasSubset.Subset B₁ B₂
                               hAB : CovBySMul M K A B₁
                               ⊢ CovBySMul M K A B₂
                             -/
    CovBySMul M K A B₂ := by simpa using hAB.trans (.of_subset (M := M) hB)
                             /-
                               🎉 no goals
                             -/


@[to_additive]
lemma CovBySMul.subset (hA : A₁ ⊆ A₂) (hB : B₁ ⊆ B₂) (hAB : CovBySMul M K A₂ B₁) :
    CovBySMul M K A₁ B₂ := (hAB.subset_left hA).subset_right hB

