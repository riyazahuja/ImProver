/-- A Prop-valued class for a bilinear form to be compatible with a root pairing. -/
class IsRootPositive (P : RootPairing ι R M N) (B : M →ₗ[R] M →ₗ[R] R) : Prop where
  zero_lt_apply_root : ∀ i, 0 < B (P.root i) (P.root i)
  symm : ∀ x y, B x y = B y x
  apply_reflection_eq : ∀ i x y, B (P.reflection i x) (P.reflection i y) = B x y


lemma two_mul_apply_root_root :
    2 * B (P.root i) (P.root j) = P.pairing i j * B (P.root j) (P.root j) := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : LinearOrderedCommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    inst✝ : P.IsRootPositive B
    i j : ι
    ⊢ Eq (HMul.hMul 2 ((B (P.root i)) (P.root j))) (HMul.hMul (P.pairing i j) ((B  …
  -/
  rw [two_mul, ← eq_sub_iff_add_eq]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : LinearOrderedCommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    inst✝ : P.IsRootPositive B
    i j : ι
    ⊢ Eq ((B (P.root i)) (P.root j)) (HSub.hSub (HMul.hMul (P.pairing i j) ((B (P. …
  -/
  nth_rw 1 [← IsRootPositive.apply_reflection_eq (P := P) (B := B) j (P.root i) (P.root j)]
  rw [reflection_apply, reflection_apply_self, root_coroot'_eq_pairing, LinearMap.map_sub₂,
    LinearMap.map_smul₂, smul_eq_mul, LinearMap.map_neg, LinearMap.map_neg, mul_neg, neg_sub_neg]


@[simp]
lemma zero_lt_apply_root_root_iff : 0 < B (P.root i) (P.root j) ↔ 0 < P.pairing i j := by
  refine ⟨fun h ↦ (mul_pos_iff_of_pos_right
    (IsRootPositive.zero_lt_apply_root (P := P) (B := B) j)).mp ?_,
      fun h ↦ (mul_pos_iff_of_pos_left zero_lt_two).mp ?_⟩
    /-
      case refine_1
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁵ : LinearOrderedCommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      P : RootPairing ι R M N
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      inst✝ : P.IsRootPositive B
      i j : ι
      h : LT.lt 0 ((B (P.root i)) (P.root j))
      ⊢ LT.lt 0 (HMul.hMul (P.pairing i j) ((B (P.root j)) (P.root j)))
    -/
  · rw [← two_mul_apply_root_root]
    /-
      case refine_1
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁵ : LinearOrderedCommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      P : RootPairing ι R M N
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      inst✝ : P.IsRootPositive B
      i j : ι
      h : LT.lt 0 ((B (P.root i)) (P.root j))
      ⊢ LT.lt 0 (HMul.hMul 2 ((B (P.root i)) (P.root j)))
    -/
    exact mul_pos zero_lt_two h
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁵ : LinearOrderedCommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      P : RootPairing ι R M N
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      inst✝ : P.IsRootPositive B
      i j : ι
      h : LT.lt 0 (P.pairing i j)
      ⊢ LT.lt 0 (HMul.hMul 2 ((B (P.root i)) (P.root j)))
    -/
  · rw [two_mul_apply_root_root]
    /-
      case refine_2
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁵ : LinearOrderedCommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      P : RootPairing ι R M N
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      inst✝ : P.IsRootPositive B
      i j : ι
      h : LT.lt 0 (P.pairing i j)
      ⊢ LT.lt 0 (HMul.hMul (P.pairing i j) ((B (P.root j)) (P.root j)))
    -/
    exact mul_pos h (IsRootPositive.zero_lt_apply_root (P := P) (B := B) j)
    /-
      🎉 no goals
    -/


lemma zero_lt_pairing_iff : 0 < P.pairing i j ↔ 0 < P.pairing j i := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : LinearOrderedCommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    inst✝ : P.IsRootPositive B
    i j : ι
    ⊢ Iff (LT.lt 0 (P.pairing i j)) (LT.lt 0 (P.pairing j i))
  -/
  rw [← zero_lt_apply_root_root_iff B, IsRootPositive.symm P, zero_lt_apply_root_root_iff]
  /-
    🎉 no goals
  -/


lemma coxeterWeight_non_neg : 0 ≤ P.coxeterWeight i j := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : LinearOrderedCommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    inst✝ : P.IsRootPositive B
    i j : ι
    ⊢ LE.le 0 (P.coxeterWeight i j)
  -/
  dsimp [coxeterWeight]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : LinearOrderedCommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    inst✝ : P.IsRootPositive B
    i j : ι
    ⊢ LE.le 0 (HMul.hMul (P.pairing i j) (P.pairing j i))
  -/
  by_cases h : 0 < P.pairing i j
    /-
      case pos
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁵ : LinearOrderedCommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      P : RootPairing ι R M N
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      inst✝ : P.IsRootPositive B
      i j : ι
      h : LT.lt 0 (P.pairing i j)
      ⊢ LE.le 0 (HMul.hMul (P.pairing i j) (P.pairing j i))
    -/
  · exact le_of_lt <| mul_pos h ((zero_lt_pairing_iff B i j).mp h)
    /-
      🎉 no goals
    -/
    /-
      case neg
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁵ : LinearOrderedCommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      P : RootPairing ι R M N
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      inst✝ : P.IsRootPositive B
      i j : ι
      h : Not (LT.lt 0 (P.pairing i j))
      ⊢ LE.le 0 (HMul.hMul (P.pairing i j) (P.pairing j i))
    -/
  · have hn : ¬ 0 < P.pairing j i := fun hc ↦ h ((zero_lt_pairing_iff B i j).mpr hc)
    /-
      case neg
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁵ : LinearOrderedCommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      P : RootPairing ι R M N
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      inst✝ : P.IsRootPositive B
      i j : ι
      h : Not (LT.lt 0 (P.pairing i j))
      hn : Not (LT.lt 0 (P.pairing j i))
      ⊢ LE.le 0 (HMul.hMul (P.pairing i j) (P.pairing j i))
    -/
    simp_all only [not_lt, ge_iff_le]
    /-
      case neg
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁵ : LinearOrderedCommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      P : RootPairing ι R M N
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      inst✝ : P.IsRootPositive B
      i j : ι
      h : LE.le (P.pairing i j) 0
      hn : LE.le (P.pairing j i) 0
      ⊢ LE.le 0 (HMul.hMul (P.pairing i j) (P.pairing j i))
    -/
    exact mul_nonneg_of_nonpos_of_nonpos h hn
    /-
      🎉 no goals
    -/


@[simp]
lemma apply_root_root_zero_iff : B (P.root i) (P.root j) = 0 ↔ P.pairing i j = 0 := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : LinearOrderedCommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    inst✝ : P.IsRootPositive B
    i j : ι
    ⊢ Iff (Eq ((B (P.root i)) (P.root j)) 0) (Eq (P.pairing i j) 0)
  -/
  refine ⟨fun hB => ?_, fun hP => ?_⟩
    /-
      case refine_1
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁵ : LinearOrderedCommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      P : RootPairing ι R M N
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      inst✝ : P.IsRootPositive B
      i j : ι
      hB : Eq ((B (P.root i)) (P.root j)) 0
      ⊢ Eq (P.pairing i j) 0
    -/
  · have h2 : 2 * (B (P.root i)) (P.root j) = 0 := mul_eq_zero_of_right 2 hB
    /-
      case refine_1
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁵ : LinearOrderedCommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      P : RootPairing ι R M N
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      inst✝ : P.IsRootPositive B
      i j : ι
      hB : Eq ((B (P.root i)) (P.root j)) 0
      h2 : Eq (HMul.hMul 2 ((B (P.root i)) (P.root j))) 0
      ⊢ Eq (P.pairing i j) 0
    -/
    rw [two_mul_apply_root_root] at h2
    /-
      case refine_1
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁵ : LinearOrderedCommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      P : RootPairing ι R M N
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      inst✝ : P.IsRootPositive B
      i j : ι
      hB : Eq ((B (P.root i)) (P.root j)) 0
      h2 : Eq (HMul.hMul (P.pairing i j) ((B (P.root j)) (P.root j))) 0
      ⊢ Eq (P.pairing i j) 0
    -/
    exact eq_zero_of_ne_zero_of_mul_right_eq_zero (IsRootPositive.zero_lt_apply_root j).ne' h2
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁵ : LinearOrderedCommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      P : RootPairing ι R M N
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      inst✝ : P.IsRootPositive B
      i j : ι
      hP : Eq (P.pairing i j) 0
      ⊢ Eq ((B (P.root i)) (P.root j)) 0
    -/
  · have h2 : 2 * B (P.root i) (P.root j) = 0 := by rw [two_mul_apply_root_root, hP, zero_mul]
    /-
      case refine_2
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁵ : LinearOrderedCommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      P : RootPairing ι R M N
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      inst✝ : P.IsRootPositive B
      i j : ι
      hP : Eq (P.pairing i j) 0
      h2 : Eq (HMul.hMul 2 ((B (P.root i)) (P.root j))) 0
      ⊢ Eq ((B (P.root i)) (P.root j)) 0
    -/
    exact (mul_eq_zero.mp h2).resolve_left two_ne_zero
    /-
      🎉 no goals
    -/


lemma pairing_zero_iff : P.pairing i j = 0 ↔ P.pairing j i = 0 := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : LinearOrderedCommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    inst✝ : P.IsRootPositive B
    i j : ι
    ⊢ Iff (Eq (P.pairing i j) 0) (Eq (P.pairing j i) 0)
  -/
  rw [← apply_root_root_zero_iff B, IsRootPositive.symm P, apply_root_root_zero_iff B]
  /-
    🎉 no goals
  -/


lemma coxeterWeight_zero_iff_isOrthogonal : P.coxeterWeight i j = 0 ↔ P.IsOrthogonal i j := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : LinearOrderedCommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    inst✝ : P.IsRootPositive B
    i j : ι
    ⊢ Iff (Eq (P.coxeterWeight i j) 0) (P.IsOrthogonal i j)
  -/
  rw [coxeterWeight, mul_eq_zero]
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : LinearOrderedCommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    inst✝ : P.IsRootPositive B
    i j : ι
    ⊢ Iff (Or (Eq (P.pairing i j) 0) (Eq (P.pairing j i) 0)) (P.IsOrthogonal i j)
  -/
  refine ⟨fun h => ?_, fun h => Or.inl h.1⟩
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : LinearOrderedCommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : Module R M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R N
    P : RootPairing ι R M N
    B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
    inst✝ : P.IsRootPositive B
    i j : ι
    h : Or (Eq (P.pairing i j) 0) (Eq (P.pairing j i) 0)
    ⊢ P.IsOrthogonal i j
  -/
  rcases h with h | h
    /-
      case inl
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁵ : LinearOrderedCommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      P : RootPairing ι R M N
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      inst✝ : P.IsRootPositive B
      i j : ι
      h : Eq (P.pairing i j) 0
      ⊢ P.IsOrthogonal i j
    -/
  · exact ⟨h, (pairing_zero_iff B i j).mp h⟩
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁵ : LinearOrderedCommRing R
      inst✝⁴ : AddCommGroup M
      inst✝³ : Module R M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R N
      P : RootPairing ι R M N
      B : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) M R)
      inst✝ : P.IsRootPositive B
      i j : ι
      h : Eq (P.pairing j i) 0
      ⊢ P.IsOrthogonal i j
    -/
  · exact ⟨(pairing_zero_iff B j i).mp h, h⟩
    /-
      🎉 no goals
    -/


